// These types mirror backend/app/schemas/query_schema.py and the tool result schemas.

export type QueryStatus = "success" | "unsupported" | "insufficient_data" | "error";

export interface QueryPlan {
  intent: string;
  tool_name: string | null;
  metric: string | null;
  aggregation: string | null;
  group_by: string | null;
  sort_order: "asc" | "desc" | null;
  limit: number | null;
  confidence: number;
  reasoning: string;
}

export interface ClassifierInfo {
  used: "gemini" | "rule_based";
  model: string | null;
  fallback_reason: string | null;
}

export interface ValidationCheck {
  name: string;
  passed: boolean;
  detail: string | null;
  code?: string | null;
}

export interface ValidationInfo {
  status: "passed" | "failed" | "not_run";
  checks: ValidationCheck[];
}

export interface QueryErrorInfo {
  code: string;
  message: string;
  details: Record<string, unknown>;
}

export interface AggregationResult {
  tool: "aggregation_tool";
  metric: string;
  value: number;
  columns_used: string[];
  calculation_method: string;
  rows_used: number;
  rows_excluded_missing: number;
  total_revenue: number | null;
  order_count: number | null;
  order_count_method: string | null;
  notes: string[];
}

export interface GroupRow {
  group: string | null;
  value: number | null;
  row_count: number;
}

export interface GroupingResult {
  tool: "grouping_tool";
  group_by: string;
  metric: string;
  value_columns_used: string[];
  calculation_method: string;
  groups: GroupRow[];
  group_count: number;
  rows_used: number;
  rows_missing_group: number;
  rows_missing_value: number;
  notes: string[];
}

export interface RankedItem {
  rank: number;
  group: string | null;
  value: number;
  row_count: number;
}

export interface RankingResult {
  tool: "ranking_tool";
  group_by: string;
  metric: string;
  order: "asc" | "desc";
  value_columns_used: string[];
  calculation_method: string;
  n_requested: number;
  n_returned: number;
  total_groups_ranked: number;
  truncated_tie: boolean;
  items: RankedItem[];
  notes: string[];
}

export interface MissingValueResult {
  tool: "missing_value_tool";
  total_rows: number;
  total_columns: number;
  total_cells: number;
  total_missing_cells: number;
  missing_percentage: number;
  columns: { name: string; missing_count: number; missing_percentage: number }[];
  columns_with_missing: string[];
  fully_empty_columns: string[];
  rows_with_missing_count: number;
  rows_with_missing_sample: number[];
  duplicate_row_count: number;
}

export type ToolResult =
  | AggregationResult
  | GroupingResult
  | RankingResult
  | MissingValueResult;

// Response of POST /analysis/query
export interface QueryResponse {
  status: QueryStatus;
  question: string;
  dataset_id: string;
  classifier: ClassifierInfo;
  query_plan: QueryPlan | null;
  tool_used: string | null;
  result: ToolResult | null;
  explanation: string | null;
  calculation_method: string | null;
  assumptions: string[];
  validation: ValidationInfo;
  message: string | null;
  error: QueryErrorInfo | null;
}