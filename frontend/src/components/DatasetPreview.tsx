import { useEffect, useState, type ChangeEvent } from "react";
import { getDatasetPreview } from "../services/api";
import type { DatasetPreviewResponse } from "../types/dataset";
import ErrorMessage from "./ErrorMessage";

type PreviewState =
  | { phase: "loading" }
  | { phase: "success"; preview: DatasetPreviewResponse }
  | { phase: "error"; error: Error };

type DatasetPreviewProps = {
  datasetId: string;
};

const ROW_OPTIONS = [5, 10, 20];

function DatasetPreview({ datasetId }: DatasetPreviewProps) {
  const [state, setState] = useState<PreviewState>({ phase: "loading" });
  const [rows, setRows] = useState(5);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    // Ignore the answer if the component was removed or the request became old.
    let cancelled = false;

    getDatasetPreview(datasetId, rows)
      .then((preview) => {
        if (!cancelled) {
          setState({ phase: "success", preview });
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            phase: "error",
            error:
              error instanceof Error
                ? error
                : new Error("An unknown error occurred."),
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [datasetId, rows, attempt]);

  function handleRowsChange(event: ChangeEvent<HTMLSelectElement>) {
    setState({ phase: "loading" });
    setRows(Number(event.target.value));
  }

  function handleRetry() {
    setState({ phase: "loading" });
    setAttempt((current) => current + 1);
  }

  if (state.phase === "loading") {
    return <p className="status status-loading">Loading preview...</p>;
  }

  if (state.phase === "error") {
    return (
      <div>
        <ErrorMessage error={state.error} />
        <button type="button" onClick={handleRetry}>
          Retry
        </button>
      </div>
    );
  }

  const { preview } = state;

  return (
    <div>
      <dl className="details">
        <dt>File name</dt>
        <dd>{preview.filename}</dd>
        <dt>Format</dt>
        <dd>{preview.source_format.toUpperCase()}</dd>
        <dt>Rows</dt>
        <dd>{preview.row_count.toLocaleString()}</dd>
        <dt>Columns</dt>
        <dd>{preview.column_count}</dd>
      </dl>

      <div className="preview-toolbar">
        <label>
          Rows to show:{" "}
          <select value={rows} onChange={handleRowsChange}>
            {ROW_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <span className="hint">
          Showing {preview.preview_rows.length} of{" "}
          {preview.row_count.toLocaleString()} rows
        </span>
      </div>

      <div className="table-wrapper">
        <table className="preview-table">
          <thead>
            <tr>
              {preview.column_names.map((name) => (
                <th key={name}>{name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.preview_rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex}>
                    {cell === null ? (
                      <span className="empty-cell">empty</span>
                    ) : (
                      cell
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {preview.cells_truncated && (
        <p className="hint">
          Long values are shortened to {preview.max_cell_length} characters in
          this preview. The stored data is not changed.
        </p>
      )}
    </div>
  );
}

export default DatasetPreview;