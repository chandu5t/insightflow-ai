"""Unit tests for Baseline A, Baseline B, and Baseline C runners."""

import uuid
from pathlib import Path

from app.evaluation.baselines import BaselineARunner, BaselineBRunner, BaselineCRunner, CountingLlmClient
from app.evaluation.schemas import BenchmarkCategory, DifficultyLevel, EvaluatorInput
from app.services.dataset_repository import JsonDatasetRepository
from tests.evaluation_helpers import seed_evaluation_dataset
from tests.fakes import FakeLlm, plan_json


def test_counting_client_preserves_observed_provider_usage_and_leaves_cost_out():
    class UsageAwareFake:
        model_name = "usage-fake"

        def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
            raise AssertionError("usage-aware method should be selected")

        def generate_json_with_usage(self, *, system_prompt: str, user_prompt: str):
            from app.services.gemini_client import LlmCallResult

            return LlmCallResult(text='{"answer": 1}', input_tokens=9, output_tokens=4)

    counted = CountingLlmClient(UsageAwareFake())
    assert counted.generate_json(system_prompt="s", user_prompt="u") == '{"answer": 1}'
    assert counted.call_count == 1
    assert counted.input_tokens == 9
    assert counted.output_tokens == 4


def test_baseline_runners(tmp_path: Path):
    repo = JsonDatasetRepository(tmp_path)
    dataset_id = uuid.uuid4()
    seed_evaluation_dataset(repo, dataset_id, filename="sales.csv")

    case_input = EvaluatorInput(
        case_id="Q1",
        dataset="sales.csv",
        question="What is the total revenue?",
        category=BenchmarkCategory.A_AGGREGATION,
        difficulty=DifficultyLevel.EASY,
    )

    # 1. Baseline A
    runner_a = BaselineARunner(repo, llm_client=FakeLlm('{"answer": 122500.0}'))
    res_a = runner_a.run(case_input, dataset_id)
    assert res_a["success"] is True
    assert res_a["raw_answer"] == 122500.0
    assert res_a["llm_calls"] == 1

    # 2. Baseline B (Tool augmented)
    runner_b = BaselineBRunner(repo, llm_client=FakeLlm(plan_json()))
    res_b = runner_b.run(case_input, dataset_id)
    assert res_b["success"] is True
    # In SALES_CSV: (2*55000) + (10*500) + (5*1500) = 110000 + 5000 + 7500 = 122500
    assert res_b["raw_answer"] == 122500.0
    assert res_b["llm_calls"] == 1

    unavailable_b = BaselineBRunner(repo).run(case_input, dataset_id)
    assert unavailable_b["success"] is False
    assert unavailable_b["raw_answer"] is None

    # 3. Baseline C (InsightFlow V1)
    runner_c = BaselineCRunner(repo, llm_client=FakeLlm(plan_json()))
    res_c = runner_c.run(case_input, dataset_id)
    assert res_c["success"] is True
    assert res_c["raw_answer"] is not None
