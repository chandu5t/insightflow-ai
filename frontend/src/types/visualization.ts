export type ChartType = "bar" | "line" | "pie" | "scatter";
export type VerificationStatus = "passed" | "failed" | "unsupported";
export type VisualizationStatus = "generated" | "unsupported" | "failed";

export interface VisualizationAxis {
  field: string;
  label: string;
  data_type: "categorical" | "numeric" | "ordered" | "temporal";
}

export interface VisualizationTraceability {
  question: string;
  dataset_id: string;
  source_step_id: string;
  source_operation: string | null;
  verification_status: VerificationStatus;
  correction_status: string | null;
}

export interface VisualizationSpec {
  visualization_id: string;
  chart_type: ChartType;
  title: string;
  description: string;
  source_step_id: string;
  x_axis: VisualizationAxis;
  y_axis: VisualizationAxis | null;
  series: Array<{ name: string; field: string }>;
  data: Array<Record<string, unknown>>;
  metadata: Record<string, unknown>;
  traceability: VisualizationTraceability;
}

export interface VisualizationError {
  category: string;
  code: string;
  message: string;
}

export interface VisualizationResponse {
  status: VisualizationStatus;
  visualization: VisualizationSpec | null;
  errors: VisualizationError[];
  metadata: Record<string, unknown>;
}

export interface VisualizationRequest {
  question: string;
  dataset_id: string;
  analysis_plan: unknown;
  execution_result: unknown;
  verification_result: unknown;
  correction_result?: unknown | null;
}
