// These types mirror the backend Pydantic schemas in backend/app/schemas/dataset_schema.py.

export type SourceFormat = "csv" | "xlsx";

// Response of POST /datasets/upload
export interface DatasetSummary {
  dataset_id: string;
  filename: string;
  source_format: SourceFormat;
  row_count: number;
  column_count: number;
  column_names: string[];
  uploaded_at: string;
  status: "uploaded";
  warnings: string[];
}

// Response of GET /datasets/{dataset_id}/preview
export interface DatasetPreviewResponse {
  dataset_id: string;
  filename: string;
  source_format: SourceFormat;
  row_count: number;
  column_count: number;
  column_names: string[];
  requested_rows: number;
  preview_rows: (string | null)[][];
  cells_truncated: boolean;
  max_cell_length: number;
}