import { useEffect, useState } from "react";
import { getHealth, type HealthResponse } from "../services/api";

type HealthState = {
  phase: "loading" | "success" | "error";
  data: HealthResponse | null;
  error: string | null;
};

function HealthStatus() {
  const [state, setState] = useState<HealthState>({
    phase: "loading",
    data: null,
    error: null,
  });

  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    // Prevent state updates if the component is removed.
    let cancelled = false;

    getHealth()
      .then((data) => {
        if (!cancelled) {
          setState({
            phase: "success",
            data,
            error: null,
          });
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            phase: "error",
            data: null,
            error:
              error instanceof Error
                ? error.message
                : "An unknown error occurred.",
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [attempt]);

  function handleRetry() {
    setState({
      phase: "loading",
      data: null,
      error: null,
    });

    setAttempt((current) => current + 1);
  }

  if (state.phase === "loading") {
    return (
      <p className="status status-loading">
        Checking backend connection...
      </p>
    );
  }

  if (state.phase === "error") {
    return (
      <div>
        <p className="status status-error">Backend: unreachable</p>

        <p className="error-message">{state.error}</p>

        <button type="button" onClick={handleRetry}>
          Retry
        </button>
      </div>
    );
  }

  const { data } = state;

  if (!data) {
    return null;
  }

  return (
    <div>
      <p className="status status-healthy">
        Backend: {data.status}
      </p>

      <dl className="details">
        <dt>Application</dt>
        <dd>{data.app}</dd>

        <dt>Version</dt>
        <dd>{data.version}</dd>

        <dt>Environment</dt>
        <dd>{data.environment}</dd>
      </dl>
    </div>
  );
}

export default HealthStatus;