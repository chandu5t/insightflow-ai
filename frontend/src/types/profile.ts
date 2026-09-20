// These types mirror backend/app/schemas/profile_schema.py.
// Numbers that could not be calculated (or were not finite) arrive as null.

import type { SourceFormat } from "./dataset";

export type DataType =
  | "integer"
  | "float"
  | "boolean"
  | "datetime"
  | "text"
  | "empty";

export interface NumericStats {
  count: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  p25: number | null;
  median: number | null;
  p75: number | null;
  max: number | null;
}

export interface DatetimeStats {
  min: string;
  max: string;
}

export interface ValueCount {
  value: string;
  count: number;
}

export interface CategoricalInfo {
  unique_count: number;
  top_values: ValueCount[];
  values_truncated: boolean;
}

export interface ColumnProfile {
  name: string;
  data_type: DataType;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  numeric_stats: NumericStats | null;
  datetime_stats: DatetimeStats | null;
  categorical: CategoricalInfo | null;
}

export interface ColumnMappingInfo {
  resolved: Record<string, string>;
  ambiguous: Record<string, string[]>;
  missing: string[];
}

// Response of GET /datasets/{dataset_id}/profile
export interface DatasetProfileResponse {
  dataset_id: string;
  filename: string;
  source_format: SourceFormat;
  row_count: number;
  column_count: number;
  column_names: string[];
  duplicate_row_count: number;
  total_missing_cells: number;
  columns: ColumnProfile[];
  column_mapping: ColumnMappingInfo;
  warnings: string[];
}