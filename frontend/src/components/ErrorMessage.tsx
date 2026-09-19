import { ApiError } from "../services/api";

type ErrorMessageProps = {
  error: Error;
};

function formatDetail(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((item) => formatDetail(item)).join(", ");
  }
  if (typeof value === "object" && value !== null) {
    return JSON.stringify(value);
  }
  return String(value);
}

function ErrorMessage({ error }: ErrorMessageProps) {
  const details = error instanceof ApiError ? Object.entries(error.details) : [];

  return (
    <div className="error-box" role="alert">
      <p className="error-message">{error.message}</p>

      {error instanceof ApiError && (
        <p className="error-code">Error code: {error.code}</p>
      )}

      {details.length > 0 && (
        <ul className="error-details">
          {details.map(([key, value]) => (
            <li key={key}>
              {key}: {formatDetail(value)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default ErrorMessage;