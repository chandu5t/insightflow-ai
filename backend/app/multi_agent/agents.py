"""The three controlled V2.3 agents and Analysis Agent operation handlers."""

from typing import Any
from dataclasses import dataclass

import numpy as np
import numpy as np
import pandas as pd

from app.multi_agent.schemas import AgentResult, WorkflowError, WorkflowState
from app.planner.operations import OPERATION_REGISTRY
from app.services.dataset_repository import DatasetRepository
from app.services.metric_retriever import MetricRetriever
from app.utils.dataframe_utils import load_dataframe
from app.planner.semantic_manifest import load_semantic_manifest


class OperationFailure(ValueError):
    pass


@dataclass
class OperationOutput:
    value: Any
    semantic_role: str = "unknown"
    lineage: list[dict[str, Any]] | None = None


def _successful_step_agent(state: WorkflowState, step_id: str):
    return next((item for item in state.get("agent_outputs", [])
                 if item.agent_name == "analysis" and item.step_id == step_id
                 and item.status == "completed"), None)


def _role(state: WorkflowState, column: str) -> str:
    manifest = state.get("semantic_manifest")
    return manifest.role_for(column) if manifest is not None else "unknown"


def _eval_expression(expr: dict[str, Any], frame: pd.DataFrame, state: WorkflowState):
    if "literal" in expr:
        return float(expr["literal"]), "unknown", []
    if "column" in expr:
        name = expr["column"]
        if name not in frame:
            raise OperationFailure("Formula references an unknown column.")
        source = frame[name]
        numeric = pd.to_numeric(source, errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise OperationFailure("Required formula values must be numeric, finite, and non-missing.")
        return numeric.astype(float), _role(state, name), [{"kind": "column", "name": name}]
    if "step_id" in expr:
        step_id = expr["step_id"]
        if step_id not in state["step_results"]:
            raise OperationFailure("Formula step reference is unavailable.")
        prior = _successful_step_agent(state, step_id)
        if prior is None:
            raise OperationFailure("Formula step reference is not a successful Analysis Agent result.")
        raw = state["step_results"][step_id]
        if isinstance(raw, pd.Series):
            number = pd.to_numeric(raw, errors="coerce")
            if number.isna().any() or not np.isfinite(number.to_numpy(dtype=float)).all():
                raise OperationFailure("Referenced formula values must be finite and non-missing.")
            return number.astype(float), prior.metadata.get("semantic_role", "unknown"), [{"kind": "step", "step_id": step_id}]
        return _safe_number(raw), prior.metadata.get("semantic_role", "unknown"), [{"kind": "step", "step_id": step_id}]
    if expr["op"] == "negate":
        val, role, lineage = _eval_expression(expr["value"], frame, state)
        return -val, role, lineage
    left, left_role, left_lineage = _eval_expression(expr["left"], frame, state)
    right, right_role, right_lineage = _eval_expression(expr["right"], frame, state)
    op = expr["op"]
    if op == "divide" and ((isinstance(right, (int, float)) and right == 0)
                            or (isinstance(right, pd.Series) and (right == 0).any())):
        raise OperationFailure("Division by zero is not allowed.")
    value = {"add": lambda: left + right, "subtract": lambda: left - right,
             "multiply": lambda: left * right, "divide": lambda: left / right}[op]()
    role = "unknown"
    if op in {"add", "subtract"} and left_role == right_role == "additive_measure":
        role = "additive_measure"
    if op == "multiply" and left_lineage and right_lineage:
        manifest = state.get("semantic_manifest")
        pairs = set(manifest.additive_products) if manifest else set()
        left_name, right_name = left_lineage[0].get("name"), right_lineage[0].get("name")
        if (left_name, right_name) in pairs or (right_name, left_name) in pairs:
            role = "additive_measure"
    return value, role, left_lineage + right_lineage


def _comparison_value(agent: AgentResult, selector: Any) -> float:
    value = agent.result
    if selector == "scalar":
        if isinstance(value, dict) and value.get("kind") == "scalar":
            value = value.get("value")
    elif selector.get("kind") == "group_key":
        key = selector["key"]
        groups = value.get("groups") if isinstance(value, dict) else None
        if isinstance(groups, list):
            matches = [item.get("value") for item in groups if item.get("key") == key]
            if len(matches) != 1:
                raise OperationFailure("Comparison group reference is missing or ambiguous.")
            value = matches[0]
        elif isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise OperationFailure("Comparison group reference is unavailable.")
    elif selector.get("kind") == "list_item":
        if agent.metadata.get("operation") not in {"rank", "top_n", "bottom_n"}:
            raise OperationFailure("List-item comparisons require a ranked/selected prior result.")
        try:
            value = value[selector["index"]][selector["field"]]
        except (IndexError, KeyError, TypeError) as exc:
            raise OperationFailure("Comparison list-item reference is unavailable.") from exc
    elif selector.get("kind") == "leader":
        if agent.metadata.get("operation") not in {"rank", "top_n"}:
            raise OperationFailure("A leader must come from a successful rank or top_n result.")
        try:
            first = value[0]
            value = first["value"]
        except (IndexError, KeyError, TypeError) as exc:
            raise OperationFailure("The upstream ranking result has no explicit leader value.") from exc
    return _safe_number(value)


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


def _derive_frame_input(step, state: WorkflowState) -> pd.DataFrame:
    """Resolve at most one table dependency; numeric dependencies are expression operands."""
    values = _dependency_results(step, state)
    frames = [value for value in values if isinstance(value, pd.DataFrame)]
    if len(frames) > 1:
        raise OperationFailure("derive_metric accepts at most one table dependency.")
    frame = frames[0] if frames else state.get("frame")
    if not isinstance(frame, pd.DataFrame):
        raise OperationFailure("derive_metric requires a dataset or prior table result.")
    return frame


def _safe_number(value: Any) -> float:
    if isinstance(value, dict) and value.get("kind") == "scalar":
        value = value.get("value")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
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
    if operation == "filter_rows":
        frame = _frame_input(step, state)
        if params.get("mode", "predicate") == "exact_duplicate_rows":
            if params.get("keep") != "first":
                raise OperationFailure("Exact duplicate removal supports keep='first' only.")
            return frame.drop_duplicates(keep="first").copy()
        mask = pd.Series(True, index=frame.index)
        for condition in params["conditions"]:
            name, operator = condition["column"], condition["operator"]
            if name not in frame:
                raise OperationFailure("Predicate references an unknown column.")
            series = frame[name]
            missing = series.isna()
            if operator == "is_missing":
                selected = missing
            elif operator == "is_not_missing":
                selected = ~missing
            else:
                valid = ~missing
                literal = condition["value"]
                role = _role(state, name)
                if role in {"additive_measure", "non_additive_measure"}:
                    numeric_literal = (isinstance(literal, (int, float)) and not isinstance(literal, bool))
                    numeric_list = isinstance(literal, list) and all(
                        isinstance(item, (int, float)) and not isinstance(item, bool) for item in literal
                    )
                    if ((operator in {"in", "not_in"} and not numeric_list)
                            or (operator not in {"in", "not_in"} and not numeric_literal)):
                        raise OperationFailure("Numeric predicates require numeric literals.")
                    values = pd.to_numeric(series, errors="coerce")
                    if values[valid].isna().any() or not np.isfinite(values[valid].to_numpy(dtype=float)).all():
                        raise OperationFailure("Numeric predicate source contains invalid values.")
                    right = [float(item) for item in literal] if isinstance(literal, list) else float(literal)
                    left = values
                else:
                    if operator in {"in", "not_in"}:
                        valid_literal = isinstance(literal, list) and all(isinstance(item, str) for item in literal)
                    else:
                        valid_literal = isinstance(literal, str)
                    if not valid_literal:
                        raise OperationFailure("Predicate literal type does not match the column role.")
                    if role == "date" and operator in {"lt", "lte", "gt", "gte"}:
                        raise OperationFailure("Ordered date predicates are unsupported without a frozen date format.")
                    right, left = literal, series
                if operator in {"eq", "in"}:
                    selected = left.isin(right if isinstance(right, list) else [right])
                elif operator in {"ne", "not_in"}:
                    selected = ~left.isin(right if isinstance(right, list) else [right])
                else:
                    if role not in {"additive_measure", "non_additive_measure"}:
                        raise OperationFailure("Ordered predicates require a trusted numeric column role.")
                    selected = {"lt": left < right, "lte": left <= right, "gt": left > right, "gte": left >= right}[operator]
                selected &= valid
            mask &= selected.fillna(False)
        return frame.loc[mask].copy()
    if operation == "derive_metric":
        frame = _derive_frame_input(step, state)
        value, role, lineage = _eval_expression(params["formula"], frame, state)
        if isinstance(value, pd.Series):
            if not value.index.equals(frame.index) or value.isna().any() or not np.isfinite(value.to_numpy(dtype=float)).all():
                raise OperationFailure("Derived values must be finite, non-missing, and row-aligned.")
            value = value.astype(float)
            value.name = params["output_name"]
        elif not np.isfinite(float(value)):
            raise OperationFailure("Derived value must be finite.")
        return OperationOutput(value, role, lineage)
    if operation == "group_by":
        frame = _frame_input(step, state)
        column = params.get("column")
        if column is None and len(step.inputs) == 1:
            column = step.inputs[0]
        if not isinstance(column, str) or column not in frame.columns:
            raise OperationFailure("group_by requires an existing column parameter.")
        if step.inputs and column not in step.inputs:
            raise OperationFailure("The grouping column must be present in the named operation inputs.")
        missing = frame[column].isna()
        if missing.any() and frame.loc[~missing, column].astype(str).eq("(missing)").any():
            raise OperationFailure("Literal '(missing)' conflicts with the missing-group label.")
        keyed = frame.copy()
        keyed[column] = keyed[column].where(~missing, "(missing)")
        return {str(key): group.copy() for key, group in keyed.groupby(column, dropna=False, sort=True)}
    if operation == "aggregate":
        if params.get("function") != "sum" or len(step.inputs) != 1:
            raise OperationFailure("aggregate supports sum over exactly one explicit value input.")
        name = step.inputs[0]
        manifest = state.get("semantic_manifest")
        raw = values[0] if values else state.get("frame")
        if isinstance(raw, dict):
            if not raw:
                raise OperationFailure("Empty grouped aggregate input is unsupported.")
            groups = []
            upstream = _successful_step_agent(state, step.depends_on[0]) if step.depends_on else None
            lineage = list(upstream.metadata.get("lineage", [])) if upstream else []
            if step.depends_on:
                lineage.append({"kind": "step", "step_id": step.depends_on[0]})
            lineage.append({"kind": "column", "name": name})
            for key in sorted(raw):
                group = raw[key]
                if not isinstance(group, pd.DataFrame) or name not in group:
                    raise OperationFailure("Grouped aggregate source is unavailable.")
                if manifest is None or manifest.role_for(name) != "additive_measure":
                    raise OperationFailure("Only trusted additive measures may be summed.")
                numbers = pd.to_numeric(group[name], errors="coerce")
                if numbers.empty:
                    raise OperationFailure("Empty aggregate input is unsupported.")
                if numbers.isna().any() or not np.isfinite(numbers.to_numpy(dtype=float)).all():
                    raise OperationFailure("Missing, invalid, or nonfinite aggregate values are not allowed.")
                group_total = float(numbers.sum())
                if not np.isfinite(group_total):
                    raise OperationFailure("Aggregate result must be finite.")
                groups.append({"key": str(key), "value": group_total})
            return OperationOutput({"kind": "grouped", "groups": groups}, "additive_measure", lineage)
        if isinstance(raw, pd.DataFrame) and name in raw:
            series = raw[name]
            role = manifest.role_for(name) if manifest else "unknown"
            prior = _successful_step_agent(state, step.depends_on[0]) if step.depends_on else None
            lineage = list(prior.metadata.get("lineage", [])) if prior else []
            if step.depends_on:
                lineage.append({"kind": "step", "step_id": step.depends_on[0]})
            lineage.append({"kind": "column", "name": name})
        elif isinstance(raw, pd.Series):
            series = raw
            prior = _successful_step_agent(state, step.depends_on[0]) if step.depends_on else None
            role = prior.metadata.get("semantic_role", "unknown") if prior else "unknown"
            lineage = list(prior.metadata.get("lineage", [])) if prior else []
            if step.depends_on:
                lineage.append({"kind": "step", "step_id": step.depends_on[0]})
        else:
            raise OperationFailure("aggregate requires a trusted column or derived measure.")
        if role != "additive_measure":
            raise OperationFailure("Only trusted additive measures may be summed.")
        numbers = pd.to_numeric(series, errors="coerce")
        if numbers.empty:
            raise OperationFailure("Empty aggregate input is unsupported.")
        if numbers.isna().any() or not np.isfinite(numbers.to_numpy(dtype=float)).all():
            raise OperationFailure("Missing, invalid, or nonfinite aggregate values are not allowed.")
        total = float(numbers.sum())
        if not np.isfinite(total):
            raise OperationFailure("Aggregate result must be finite.")
        return OperationOutput({"kind": "scalar", "value": total}, role, lineage)
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
            if isinstance(raw, dict) and raw.get("kind") == "grouped" and isinstance(raw.get("groups"), list):
                items = raw["groups"]
                if any(not isinstance(item, dict) or not isinstance(item.get("key"), str)
                       or not isinstance(item.get("value"), (int, float)) for item in items):
                    raise OperationFailure(f"{operation} requires numeric grouped values.")
                if operation == "rank" and len({item["value"] for item in items}) != len(items):
                    raise OperationFailure("V2.2 does not define how tied values receive ranks.")
                ordered_items = sorted(items, key=lambda item: item["value"], reverse=not ascending)
                if operation == "rank":
                    return [{"key": item["key"], "value": item["value"], "rank": i + 1}
                            for i, item in enumerate(ordered_items)]
                return {"kind": "grouped", "groups": ordered_items}
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
            result = left - right
            if not np.isfinite(result):
                raise OperationFailure("Difference result must be finite.")
            return result
        if right == 0:
            raise OperationFailure("percentage difference reference value cannot be zero.")
        result = ((left - right) / right) * 100.0
        if not np.isfinite(result):
            raise OperationFailure("Percentage difference result must be finite.")
        if operation == "calculate_percentage_difference":
            return OperationOutput(result, "non_additive_measure",
                                  [{"kind": "step", "step_id": step.depends_on[0]},
                                   {"kind": "step", "step_id": step.depends_on[1]}])
        return result
    if operation == "compare_groups":
        operands = []
        for side in ("left", "right"):
            specification = params[side]
            prior = _successful_step_agent(state, specification["step_id"])
            if prior is None:
                raise OperationFailure("Comparison operands must reference successful prior Analysis Agent results.")
            operand = _comparison_value(prior, specification["selector"]) * float(specification["multiplier"])
            if not np.isfinite(operand):
                raise OperationFailure("Comparison operands must remain finite after multiplication.")
            operands.append(operand)
        op = params["comparator"]
        outcome = {"lt": lambda: operands[0] < operands[1], "lte": lambda: operands[0] <= operands[1],
                   "eq": lambda: operands[0] == operands[1], "gte": lambda: operands[0] >= operands[1],
                   "gt": lambda: operands[0] > operands[1]}[op]()
        return OperationOutput({"left_value": operands[0], "comparator": op,
                                "right_value": operands[1], "result": outcome},
                               "boolean", [{"kind": "step", "step_id": params["left"]["step_id"]},
                                           {"kind": "step", "step_id": params["right"]["step_id"]}])
    raise OperationFailure(f"No controlled handler is available for {operation}.")


class DataUnderstandingAgent:
    name = "data_understanding"

    def __init__(self, repository: DatasetRepository) -> None:
        self.repository = repository

    def run(self, state: WorkflowState) -> tuple[AgentResult, dict[str, Any]]:
        try:
            metadata = self.repository.get(state["dataset_id"])
            csv_path = self.repository.get_csv_path(state["dataset_id"])
            frame = load_dataframe(csv_path)
            semantic_manifest = load_semantic_manifest(csv_path)
            context = {"columns": list(frame.columns), "row_count": len(frame), "column_count": len(frame.columns)}
            result = AgentResult(agent_name=self.name, status="completed", result=context,
                                 metadata={"dataset_id": str(state["dataset_id"]),
                                           "semantic_manifest_version": semantic_manifest.manifest_version if semantic_manifest else None,
                                           "dataset_sha256": semantic_manifest.dataset_sha256 if semantic_manifest else None})
            return result, {"frame": frame, "dataset_context": context,
                            "semantic_manifest": semantic_manifest}
        except Exception:
            error = WorkflowError(category="agent_failure", code="DATASET_CONTEXT_FAILED",
                                  message="Dataset context could not be prepared.")
            return AgentResult(agent_name=self.name, status="failed", error=error), {}


class AnalysisAgent:
    name = "analysis"

    def run(self, step, state: WorkflowState) -> tuple[AgentResult, Any]:
        try:
            result = execute_operation(step, state)
            semantic_role = result.semantic_role if isinstance(result, OperationOutput) else "unknown"
            lineage = result.lineage if isinstance(result, OperationOutput) else (
                [{"kind": "step", "step_id": dependency} for dependency in step.depends_on]
                or [{"kind": "dataset", "dataset_id": str(state["dataset_id"])}]
            )
            raw_result = result.value if isinstance(result, OperationOutput) else result
            return (AgentResult(agent_name=self.name, step_id=step.step_id, status="completed",
                                result=_plain(raw_result), metadata={"operation": step.operation,
                                    "semantic_role": semantic_role, "lineage": lineage or [],
                                    **({"output_name": step.parameters.get("output_name")}
                                       if step.operation == "derive_metric" else {})}), raw_result)
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
