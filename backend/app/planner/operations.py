"""Metadata-only registry of the 14 operations allowed in V2.2 plans."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationMetadata:
    name: str
    description: str
    required_inputs: tuple[str, ...]
    allowed_parameters: tuple[str, ...]
    required_parameters: tuple[str, ...]
    output_type: str
    dependency_information: str


_OPERATIONS = (
    OperationMetadata("select_columns", "Select named columns from the analytical context.", ("columns",), ("columns",), ("columns",), "table", "May precede row filtering or metric derivation."),
    OperationMetadata("filter_rows", "Filter rows using explicit predicates.", ("table",), ("conditions",), ("conditions",), "table", "Consumes a table and may precede analytical operations."),
    OperationMetadata("derive_metric", "Derive a named metric from supplied columns and a documented formula.", ("source_columns",), ("formula", "output_name"), ("formula", "output_name"), "metric", "Must precede operations that consume the derived metric."),
    OperationMetadata("group_by", "Partition records by one or more grouping fields.", ("table", "grouping_column"), ("column",), ("column",), "groups", "May consume a table or a derived metric."),
    OperationMetadata("aggregate", "Apply a supported aggregate to values.", ("values",), ("function",), ("function",), "scalar_or_grouped_values", "Consumes selected, filtered, or grouped values."),
    OperationMetadata("sort", "Order values or groups by an explicit key and direction.", ("values",), ("by", "order"), ("by", "order"), "ordered_values", "Usually follows aggregation or grouping."),
    OperationMetadata("rank", "Assign ranks to values or groups.", ("values",), ("by", "order"), ("by", "order"), "ranked_values", "Usually follows aggregation or grouping."),
    OperationMetadata("top_n", "Select the first N items from an ordered or ranked result.", ("ordered_values",), ("n",), ("n",), "selected_values", "Consumes an ordered or ranked result."),
    OperationMetadata("bottom_n", "Select the last N items from an ordered or ranked result.", ("ordered_values",), ("n",), ("n",), "selected_values", "Consumes an ordered or ranked result."),
    OperationMetadata("count", "Count rows or supplied items.", ("table_or_values",), ("column",), (), "count", "May consume a table, filtered table, or selected values."),
    OperationMetadata("distinct_count", "Count unique values in a named field.", ("table", "column"), ("column",), ("column",), "count", "Consumes a table or filtered table."),
    OperationMetadata("calculate_difference", "Calculate the difference between two supplied values.", ("left_value", "right_value"), (), (), "number", "Consumes two prior values or aggregates."),
    OperationMetadata("calculate_percentage_difference", "Calculate percentage difference between two supplied values.", ("value", "reference_value"), (), (), "percentage", "Consumes two prior values or aggregates."),
    OperationMetadata("compare_groups", "Compare explicitly identified groups.", ("groups",), ("left_group", "right_group"), ("left_group", "right_group"), "comparison", "Consumes grouped or ranked results."),
)

OPERATION_REGISTRY: dict[str, OperationMetadata] = {operation.name: operation for operation in _OPERATIONS}

# Explicit aliases only. Unknown values are preserved so validation can reject them.
OPERATION_ALIASES: dict[str, str] = {}
for _name in OPERATION_REGISTRY:
    OPERATION_ALIASES[_name] = _name
    OPERATION_ALIASES[_name.upper()] = _name
    OPERATION_ALIASES["".join(part.title() for part in _name.split("_"))] = _name
OPERATION_ALIASES.update({"GroupBy": "group_by", "GROUP_BY": "group_by"})


def operation_metadata_for_prompt() -> list[dict[str, object]]:
    """Return stable JSON-compatible registry metadata for the planner prompt."""
    return [
        {
            "name": item.name,
            "description": item.description,
            "required_inputs": list(item.required_inputs),
            "allowed_parameters": list(item.allowed_parameters),
            "required_parameters": list(item.required_parameters),
            "output_type": item.output_type,
            "dependency_information": item.dependency_information,
        }
        for item in OPERATION_REGISTRY.values()
    ]
