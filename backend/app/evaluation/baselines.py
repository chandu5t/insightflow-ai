"""Implementation of Baseline A, Baseline B, and Baseline C runners matching Contract v1.1 §4."""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import UUID

from app.core.config import Settings, get_settings
from app.evaluation.schemas import CaseStageResult, EvaluatorInput
from app.services.dataset_repository import DatasetRepository
from app.services.gemini_client import LlmClient
from app.services.metric_retriever import MetricRetriever
from app.services.query_service import answer_question
from app.utils.dataframe_utils import load_dataframe


class BaselineRunner:
    """Base class for baseline condition runners."""

    def __init__(
        self,
        repository: DatasetRepository,
        settings: Settings | None = None,
        llm_client: LlmClient | None = None,
        retriever: MetricRetriever | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings or get_settings()
        self.llm_client = llm_client
        self.retriever = retriever

    def run(self, case_input: EvaluatorInput, dataset_id: str | UUID) -> dict[str, Any]:
        raise NotImplementedError


class CountingLlmClient:
    """Count actual generate_json calls without changing provider behavior."""

    def __init__(self, client: LlmClient) -> None:
        self.client = client
        self.model_name = client.model_name
        self.call_count = 0
        self._input_tokens_by_call: list[int | None] = []
        self._output_tokens_by_call: list[int | None] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        usage_call = getattr(self.client, "generate_json_with_usage", None)
        if callable(usage_call):
            try:
                result = usage_call(system_prompt=system_prompt, user_prompt=user_prompt)
            except Exception:
                self._input_tokens_by_call.append(None)
                self._output_tokens_by_call.append(None)
                raise
            self._input_tokens_by_call.append(result.input_tokens)
            self._output_tokens_by_call.append(result.output_tokens)
            return result.text
        self._input_tokens_by_call.append(None)
        self._output_tokens_by_call.append(None)
        return self.client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)

    @property
    def input_tokens(self) -> int | None:
        if not self.call_count:
            return 0
        if any(value is None for value in self._input_tokens_by_call):
            return None
        return sum(value for value in self._input_tokens_by_call if value is not None)

    @property
    def output_tokens(self) -> int | None:
        if not self.call_count:
            return 0
        if any(value is None for value in self._output_tokens_by_call):
            return None
        return sum(value for value in self._output_tokens_by_call if value is not None)


class BaselineARunner(BaselineRunner):
    """Baseline A: LLM-Only (Question -> LLM -> Answer without deterministic tools)."""

    def run(self, case_input: EvaluatorInput, dataset_id: str | UUID) -> dict[str, Any]:
        start = time.perf_counter()
        stage_results: list[CaseStageResult] = []

        # If LLM client is available, simulate direct prompting
        if self.llm_client and hasattr(self.llm_client, "generate_json"):
            try:
                prompt = f"Answer the following business question about dataset {case_input.dataset}: {case_input.question}"
                raw = self.llm_client.generate_json(system_prompt="Answer directly with JSON {answer: value}", user_prompt=prompt)
                payload = json.loads(raw)
                if not isinstance(payload, dict) or "answer" not in payload:
                    raise ValueError("LLM-only baseline returned no structured answer field.")
                answer = payload["answer"]
                success = True
            except Exception as exc:
                answer = None
                success = False
        else:
            # An unavailable model is not an observed LLM-only answer.
            answer = None
            success = False

        latency_ms = (time.perf_counter() - start) * 1000
        stage_results.append(
            CaseStageResult(
                stage_name="direct_llm",
                status="completed" if success else "failed",
                latency_ms=latency_ms,
                output_summary=answer,
            )
        )
        return {
            "success": success,
            "raw_answer": answer,
            "latency_ms": latency_ms,
            "llm_calls": getattr(self.llm_client, "call_count", 1 if self.llm_client else 0),
            "input_tokens": _usage_value(self.llm_client, "input_tokens"),
            "output_tokens": _usage_value(self.llm_client, "output_tokens"),
            "stage_results": stage_results,
        }


class BaselineBRunner(BaselineRunner):
    """Baseline B: Tool-Augmented LLM (Question -> Tool Selection -> Single Tool Execution -> Answer)."""

    def run(self, case_input: EvaluatorInput, dataset_id: str | UUID) -> dict[str, Any]:
        start = time.perf_counter()
        stage_results: list[CaseStageResult] = []

        # Tool execution using deterministic tools directly without planner decomposition or verification
        dataset_uuid = UUID(str(dataset_id))
        path = self.repository.get_csv_path(dataset_uuid)
        frame = load_dataframe(path)

        success = False
        raw_answer = None
        error_msg = ""

        try:
            if self.llm_client is None:
                raise RuntimeError("Baseline B requires a configured LLM classifier.")
            from app.services.gemini_classifier import classify_question
            from app.services.query_dispatcher import dispatch

            classification = classify_question(
                case_input.question, frame.columns.tolist(), self.llm_client, self.settings
            )
            if classification.info.used != "gemini":
                raise RuntimeError("Baseline B requires successful LLM tool selection; fallback was used.")
            result = dispatch(frame, classification.plan)
            raw_answer = getattr(result, "value", result)
            success = True
        except Exception as exc:
            error_msg = str(exc)
            success = False

        latency_ms = (time.perf_counter() - start) * 1000
        stage_results.append(
            CaseStageResult(
                stage_name="single_tool_execution",
                status="completed" if success else "failed",
                latency_ms=latency_ms,
                output_summary=raw_answer if success else error_msg,
            )
        )
        return {
            "success": success,
            "raw_answer": raw_answer,
            "latency_ms": latency_ms,
            "llm_calls": getattr(self.llm_client, "call_count", 1 if self.llm_client else 0),
            "input_tokens": _usage_value(self.llm_client, "input_tokens"),
            "output_tokens": _usage_value(self.llm_client, "output_tokens"),
            "stage_results": stage_results,
        }


class BaselineCRunner(BaselineRunner):
    """Baseline C: InsightFlow V1 (Question -> V1 LangGraph -> Pandas Tools -> Validation -> Explanation)."""

    def run(self, case_input: EvaluatorInput, dataset_id: str | UUID) -> dict[str, Any]:
        start = time.perf_counter()
        stage_results: list[CaseStageResult] = []

        try:
            v1_response = answer_question(
                raw_dataset_id=str(dataset_id),
                question=case_input.question,
                settings=self.settings,
                repository=self.repository,
                llm_client=self.llm_client,
                retriever=self.retriever,
            )
            success = v1_response.status == "success"
            result = v1_response.result
            raw_answer = getattr(result, "value", result)
            raw_explanation = v1_response.explanation
        except Exception as exc:
            success = False
            raw_answer = None
            raw_explanation = str(exc)

        latency_ms = (time.perf_counter() - start) * 1000
        stage_results.append(
            CaseStageResult(
                stage_name="v1_workflow",
                status="completed" if success else "failed",
                latency_ms=latency_ms,
                output_summary=raw_answer,
                metadata={"explanation": raw_explanation},
            )
        )
        return {
            "success": success,
            "raw_answer": raw_answer,
            "raw_explanation": raw_explanation,
            "latency_ms": latency_ms,
            "llm_calls": getattr(self.llm_client, "call_count", 0) if self.llm_client else 0,
            "input_tokens": _usage_value(self.llm_client, "input_tokens"),
            "output_tokens": _usage_value(self.llm_client, "output_tokens"),
            "stage_results": stage_results,
        }


def _usage_value(client: Any, field: str) -> int | None:
    if client is None:
        return 0
    return getattr(client, field, None)
