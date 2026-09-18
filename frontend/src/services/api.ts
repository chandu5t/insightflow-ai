// Central place for all backend calls.

const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

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
    throw new Error(
      `The backend returned an error (HTTP ${response.status}).`,
    );
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