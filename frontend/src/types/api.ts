// Shape of every error returned by the backend.
export interface ApiErrorBody {
  error: string;
  code: string;
  details: Record<string, unknown>;
}