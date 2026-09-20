// Central place for all backend calls.

import type { ApiErrorBody } from "../types/api";
import type { DatasetPreviewResponse, DatasetSummary } from "../types/dataset";
import type { DatasetProfileResponse } from "../types/profile";

const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

// An error that carries the details sent by the backend.
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;

  constructor(
    message: string,
    status: number,
    code: string,
    details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.error === "string" && typeof candidate.code === "string"
  );
}

// Turn a failed response into an ApiError. Use the backend message when it exists.
async function buildApiError(response: Response): Promise<ApiError> {
  try {
    const body: unknown = await response.json();
    if (isApiErrorBody(body)) {
      const details =
        typeof body.details === "object" && body.details !== null
          ? body.details
          : {};
      return new ApiError(body.error, response.status, body.code, details);
    }
  } catch {
    // The body was not JSON. Fall through to the generic message.
  }
  return new ApiError(
    `The backend returned an error (HTTP ${response.status}).`,
    response.status,
    "HTTP_ERROR",
  );
}

async function request(
  path: string,
  options: RequestInit = {},
): Promise<unknown> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new Error(
      `Cannot reach the backend at ${API_BASE_URL}. Is the server running?`,
    );
  }

  if (!response.ok) {
    throw await buildApiError(response);
  }

  return response.json();
}

export interface HealthResponse {
  status: "healthy";
  app: string;
  version: string;
  environment: string;
}

export function getHealth(): Promise<HealthResponse> {
  return request("/health") as Promise<HealthResponse>;
}

export function uploadDataset(file: File): Promise<DatasetSummary> {
  const formData = new FormData();
  formData.append("file", file);
  // Do not set the Content-Type header here. The browser adds it, together with
  // the multipart boundary. If we set it ourselves, the upload breaks.
  return request("/datasets/upload", {
    method: "POST",
    body: formData,
  }) as Promise<DatasetSummary>;
}

export function getDatasetPreview(
  datasetId: string,
  rows?: number,
): Promise<DatasetPreviewResponse> {
  const query = rows === undefined ? "" : `?rows=${rows}`;
  return request(
    `/datasets/${encodeURIComponent(datasetId)}/preview${query}`,
  ) as Promise<DatasetPreviewResponse>;
}

export function getDatasetProfile(
  datasetId: string,
): Promise<DatasetProfileResponse> {
  return request(
    `/datasets/${encodeURIComponent(datasetId)}/profile`,
  ) as Promise<DatasetProfileResponse>;
}