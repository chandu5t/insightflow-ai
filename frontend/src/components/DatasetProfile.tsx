import { useEffect, useState } from "react";
import { getDatasetProfile } from "../services/api";
import type { DatasetProfileResponse } from "../types/profile";
import ErrorMessage from "./ErrorMessage";

type ProfileState =
  | { phase: "loading" }
  | { phase: "success"; profile: DatasetProfileResponse }
  | { phase: "error"; error: Error };

type DatasetProfileProps = {
  datasetId: string;
};

// Display only. All numbers are calculated by the backend.
function formatNumber(value: number | null): string {
  return value === null
    ? "-"
    : value.toLocaleString(undefined, { maximumFractionDigits: 6 });
}

function DatasetProfile({ datasetId }: DatasetProfileProps) {
  const [state, setState] = useState<ProfileState>({ phase: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    // Ignore the answer if the component was removed before it arrived.
    let cancelled = false;

    getDatasetProfile(datasetId)
      .then((profile) => {
        if (!cancelled) {
          setState({ phase: "success", profile });
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
  }, [datasetId, attempt]);

  function handleRetry() {
    setState({ phase: "loading" });
    setAttempt((current) => current + 1);
  }

  if (state.phase === "loading") {
    return <p className="status status-loading">Loading profile...</p>;
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

  const { profile } = state;

  if (profile.row_count === 0 || profile.column_count === 0) {
    return (
      <p className="hint">This dataset has no data to profile.</p>
    );
  }

  const numericColumns = profile.columns.flatMap((column) =>
    column.numeric_stats
      ? [{ name: column.name, stats: column.numeric_stats }]
      : [],
  );
  const dateColumns = profile.columns.flatMap((column) =>
    column.datetime_stats
      ? [{ name: column.name, dates: column.datetime_stats }]
      : [],
  );
  const categoricalColumns = profile.columns.flatMap((column) =>
    column.categorical
      ? [{ name: column.name, info: column.categorical }]
      : [],
  );
  const { resolved, ambiguous, missing } = profile.column_mapping;

  return (
    <div className="profile">
      <dl className="details">
        <dt>File name</dt>
        <dd>{profile.filename}</dd>
        <dt>Dataset ID</dt>
        <dd className="mono">{profile.dataset_id}</dd>
        <dt>Rows</dt>
        <dd>{profile.row_count.toLocaleString()}</dd>
        <dt>Columns</dt>
        <dd>{profile.column_count}</dd>
        <dt>Duplicate rows</dt>
        <dd>{profile.duplicate_row_count.toLocaleString()}</dd>
        <dt>Missing cells</dt>
        <dd>{profile.total_missing_cells.toLocaleString()}</dd>
      </dl>

      {profile.warnings.length > 0 && (
        <ul className="warning-list">
          {profile.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      <h3>Columns</h3>
      <div className="table-wrapper">
        <table className="preview-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Type</th>
              <th>Missing</th>
              <th>Missing %</th>
              <th>Unique values</th>
            </tr>
          </thead>
          <tbody>
            {profile.columns.map((column) => (
              <tr key={column.name}>
                <td>{column.name}</td>
                <td>
                  <span className={`type-badge type-${column.data_type}`}>
                    {column.data_type}
                  </span>
                </td>
                <td>{column.missing_count.toLocaleString()}</td>
                <td>{column.missing_percentage}%</td>
                <td>{column.unique_count.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {numericColumns.length > 0 && (
        <>
          <h3>Numeric statistics</h3>
          <div className="table-wrapper">
            <table className="preview-table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Count</th>
                  <th>Mean</th>
                  <th>Std dev</th>
                  <th>Min</th>
                  <th>25%</th>
                  <th>Median</th>
                  <th>75%</th>
                  <th>Max</th>
                </tr>
              </thead>
              <tbody>
                {numericColumns.map(({ name, stats }) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{stats.count.toLocaleString()}</td>
                    <td>{formatNumber(stats.mean)}</td>
                    <td>{formatNumber(stats.std)}</td>
                    <td>{formatNumber(stats.min)}</td>
                    <td>{formatNumber(stats.p25)}</td>
                    <td>{formatNumber(stats.median)}</td>
                    <td>{formatNumber(stats.p75)}</td>
                    <td>{formatNumber(stats.max)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {dateColumns.length > 0 && (
        <>
          <h3>Date columns</h3>
          <ul className="value-list">
            {dateColumns.map(({ name, dates }) => (
              <li key={name}>
                <strong>{name}:</strong> {dates.min} to {dates.max}
              </li>
            ))}
          </ul>
        </>
      )}

      {categoricalColumns.length > 0 && (
        <>
          <h3>Text and yes/no columns</h3>
          <div className="category-grid">
            {categoricalColumns.map(({ name, info }) => (
              <div className="category-card" key={name}>
                <h4>
                  {name} <span className="hint">({info.unique_count} unique)</span>
                </h4>
                <ul className="value-list">
                  {info.top_values.map((item) => (
                    <li key={item.value}>
                      {item.value} <span className="hint">({item.count})</span>
                    </li>
                  ))}
                </ul>
                {info.values_truncated && (
                  <p className="hint">
                    Showing the {info.top_values.length} most common of{" "}
                    {info.unique_count} unique values.
                  </p>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      <h3>Detected column roles</h3>
      <dl className="details">
        {Object.entries(resolved).map(([role, column]) => (
          <div className="role-row" key={role}>
            <dt>{role}</dt>
            <dd>{column}</dd>
          </div>
        ))}
        {Object.entries(ambiguous).map(([role, candidates]) => (
          <div className="role-row" key={role}>
            <dt>{role}</dt>
            <dd className="role-ambiguous">
              unclear: {candidates.join(", ")}
            </dd>
          </div>
        ))}
      </dl>
      {missing.length > 0 && (
        <p className="hint">Not found in this dataset: {missing.join(", ")}</p>
      )}
    </div>
  );
}

export default DatasetProfile;