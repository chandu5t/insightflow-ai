"""The three controlled V2.3 agents and Analysis Agent operation handlers."""

from typing import Any

import pandas as pd

from app.multi_agent.schemas import AgentResult, WorkflowError, WorkflowState
from app.planner.operations import OPERATION_REGISTRY
from app.services.dataset_repository import DatasetRepository
from app.services.metric_retriever import MetricRetriever
from app.utils.dataframe_utils import load_dataframe


class OperationFailure(ValueError):
    pass


def _plain(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {"columns": list(value.columns), "rows": value.where(pd.notna(value), None).to_dict(orient="records")}
    if isinstance(value, pd.Series):
        return value.where(pd.notna(value), None).tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _dependency_results(step, state: WorkflowState) -> list[Any]:
    """Resolve only depends_on IDs; inputs contains operation data names/values."""
    try:
        return [state["step_results"][step_id] for step_id in step.depends_on]
    except KeyError as exc:
        raise OperationFailure(f"Dependency step '{exc.args[0]}' has no result.") from exc


def _frame_input(step, state: WorkflowState) -> pd.DataFrame:
    values = _dependency_results(step, state)
    if len(values) > 1:
        raise OperationFailure("The operation does not define how to combine multiple dependency results.")
    frame = values[0] if values else state.get("frame")
    if not isinstance(frame, pd.DataFrame):
        raise OperationFailure("The operation requires a prior table result or dataset context.")
    return frame


def _safe_number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OperationFailure("The operation requires numeric values.") from exc
    if not pd.notna(result) or result in (float("inf"), float("-inf")):
        raise OperationFailure("The operation requires finite numeric values.")
    return result


def execute_operation(step, state: WorkflowState) -> Any:
    """Execute a bounded operation when the frozen V2.2 metadata is sufficient.

    V2.2 does not define safe predicate/formula grammars for filter_rows or
    derive_metric, nor a denominator convention for percentage difference. Those
    operations therefore fail explicitly instead of interpreting arbitrary text.
    """
    operation = step.operation
    if operation not in OPERATION_REGISTRY:
        raise OperationFailure(f"Unknown operation: {operation}")
    params = step.parameters
    values = _dependency_results(step, state)

    if operation == "select_columns":
        columns = step.inputs or params.get("columns")
        if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
            raise OperationFailure("columns must be a non-empty list of column names.")
        if step.inputs and params.get("columns") and list(params["columns"]) != columns:
            raise OperationFailure("inputs and parameters.columns must identify the same columns.")
        frame = _frame_input(step, state)
        if any(column not in frame.columns for column in columns):
            raise OperationFailure("One or more selected columns do not exist.")
        return frame.loc[:, columns].copy()
    if operation in {"filter_rows", "derive_metric"}:
        raise OperationFailure(f"{operation} has no frozen safe parameter grammar in V2.2.")
    if operation == "group_by":
        frame = _frame_input(step, state)
        column = params.get("column")
        if column is None and len(step.inputs) == 1:
            column = step.inputs[0]
        if not isinstance(column, str) or column not in frame.columns:
            raise OperationFailure("group_by requires an existing column parameter.")
        if step.inputs and column not in step.inputs:
            raise OperationFailure("The grouping column must be present in the named operation inputs.")
        if frame[column].isna().any():
            raise OperationFailure("V2.2 does not define how missing grouping values are handled.")
        return {str(key): group.copy() for key, group in frame.groupby(column, dropna=False, sort=False)}
    if operation == "aggregate":
        raise OperationFailure("V2.2 does not freeze the supported aggregate function values.")
    if operation in {"sort", "rank", "top_n", "bottom_n"}:
        if not values and operation in {"top_n", "bottom_n"}:
            raise OperationFailure(f"{operation} requires an input from a prior step.")
        raw = values[0] if values else state.get("frame")
        if operation in {"sort", "rank"}:
            by, order = params.get("by"), params.get("order")
            if order not in {"asc", "desc"} or not isinstance(by, str):
                raise OperationFailure(f"{operation} requires by and order=asc|desc.")
            if step.inputs and by not in step.inputs:
                raise OperationFailure("The sort/rank key must be present in the named operation inputs.")
            ascending = order == "asc"
            if isinstance(raw, pd.DataFrame) and by in raw.columns:
                if raw[by].isna().any():
                    raise OperationFailure("V2.2 does not define sort/rank ordering for missing key values.")
                result = raw.sort_values(by=by, ascending=ascending, kind="stable")
                if operation == "rank":
                    if result[by].duplicated().any():
                        raise OperationFailure("V2.2 does not define how tied values receive ranks.")
                    result = result.assign(rank=range(1, len(result) + 1))
                return result
            if isinstance(raw, dict) and all(isinstance(v, (int, float)) for v in raw.values()):
                if operation == "rank" and len(set(raw.values())) != len(raw):
                    raise OperationFailure("V2.2 does not define how tied values receive ranks.")
                ordered = sorted(raw.items(), key=lambda pair: pair[1], reverse=not ascending)
                return ([{"key": k, "value": v, "rank": i + 1} for i, (k, v) in enumerate(ordered)]
                        if operation == "rank" else dict(ordered))
            raise OperationFailure(f"{operation} input does not contain the requested key.")
        n = params.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise OperationFailure(f"{operation} requires a positive integer n.")
        if isinstance(raw, pd.DataFrame):
            return raw.head(n).copy() if operation == "top_n" else raw.tail(n).copy()
        if isinstance(raw, (list, tuple)):
            return list(raw[:n] if operation == "top_n" else raw[-n:])
        raise OperationFailure(f"{operation} requires an ordered table or list.")
    if operation == "count":
        if "column" in params:
            raise OperationFailure("V2.2 does not define count(column) null/value semantics.")
        if step.inputs and not values:
            raise OperationFailure("Named count inputs are not executable values without a prior-step dependency.")
        if values:
            raw = values[0]
            if isinstance(raw, (pd.DataFrame, pd.Series, list, tuple, dict)):
                return len(raw)
            raise OperationFailure("count input is not a countable table or collection.")
        frame = state.get("frame")
        if isinstance(frame, pd.DataFrame):
            return len(frame)
        raise OperationFailure("count requires data context.")
    if operation == "distinct_count":
        frame, column = _frame_input(step, state), params.get("column")
        if not isinstance(column, str) or column not in frame.columns:
            raise OperationFailure("distinct_count requires an existing column parameter.")
        if step.inputs and column not in step.inputs:
            raise OperationFailure("The distinct-count column must be present in the named operation inputs.")
        if frame[column].isna().any():
            raise OperationFailure("V2.2 does not define whether missing values count as distinct.")
        return int(frame[column].nunique(dropna=True))
    if operation in {"calculate_difference", "calculate_percentage_difference"}:
        if len(values) != 2:
            raise OperationFailure(f"{operation} requires exactly two prior step inputs.")
        left, right = _safe_number(values[0]), _safe_number(values[1])
        if operation == "calculate_difference":
            return left - right
        if right == 0:
            raise OperationFailure("percentage difference reference value cannot be zero.")
        raise OperationFailure("V2.2 does not define the percentage-difference denominator convention.")
    if operation == "compare_groups":
        raise OperationFailure("V2.2 does not define a comparison measure or comparison semantics.")
    raise OperationFailure(f"No controlled handler is available for {operation}.")


class DataUnderstandingAgent:
    name = "data_understanding"

    def __init__(self, repository: DatasetRepository) -> None:
        self.repository = repository

    def run(self, state: WorkflowState) -> tuple[AgentResult, dict[str, Any]]:
        try:
            metadata = self.repository.get(state["dataset_id"])
            frame = load_dataframe(self.repository.get_csv_path(state["dataset_id"]))
            context = {"columns": list(frame.columns), "row_count": len(frame), "column_count": len(frame.columns)}
            result = AgentResult(agent_name=self.name, status="completed", result=context,
                                 metadata={"dataset_id": str(state["dataset_id"])})
            return result, {"frame": frame, "dataset_context": context}
        except Exception:
            error = WorkflowError(category="agent_failure", code="DATASET_CONTEXT_FAILED",
                                  message="Dataset context could not be prepared.")
            return AgentResult(agent_name=self.name, status="failed", error=error), {}


class AnalysisAgent:
    name = "analysis"

    def run(self, step, state: WorkflowState) -> tuple[AgentResult, Any]:
        try:
            result = execute_operation(step, state)
            return (AgentResult(agent_name=self.name, step_id=step.step_id, status="completed",
                                result=_plain(result), metadata={"operation": step.operation}), result)
        except (OperationFailure, KeyError, TypeError, ValueError) as exc:
            return (AgentResult(agent_name=self.name, step_id=step.step_id, status="failed",
                                metadata={"operation": step.operation},
                                error=WorkflowError(category="agent_failure", code="OPERATION_FAILED",
                                                    message=str(exc), step_id=step.step_id)), None)


class KnowledgeAgent:
    name = "knowledge"

    def __init__(self, retriever: MetricRetriever) -> None:
        self.retriever = retriever

    def run(self, state: WorkflowState) -> AgentResult:
        try:
            found = self.retriever.lookup(state["question"])
            return AgentResult(agent_name=self.name, status="completed", result=found.model_dump())
        except Exception:
            error = WorkflowError(category="agent_failure", code="KNOWLEDGE_RETRIEVAL_FAILED",
                                  message="Knowledge retrieval failed.")
            return AgentResult(agent_name=self.name, status="failed", error=error)
