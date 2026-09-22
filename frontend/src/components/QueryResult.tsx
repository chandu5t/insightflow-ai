import { ApiError } from "../services/api";
import type { QueryResponse, ToolResult } from "../types/query";
import ErrorMessage from "./ErrorMessage";

type QueryResultProps = {
  response: QueryResponse;
};

// Display only. All numbers come from the backend.
function formatNumber(value: number | null): string {
  return value === null
    ? "-"
    : value.toLocaleString(undefined, { maximumFractionDigits: 6 });
}

function describeClassifier(response: QueryResponse): string {
  const { used, model, fallback_reason } = response.classifier;
  if (used === "gemini") {
    return model ? `Understood by Gemini (${model})` : "Understood by Gemini";
  }
  return fallback_reason
    ? `Understood by the rule-based fallback (${fallback_reason})`
    : "Understood by the rule-based classifier";
}

// NEW in Module 5: the checks that failed, with their machine-readable codes.
function FailedChecks({ response }: { response: QueryResponse }) {
  const failed = response.validation.checks.filter((check) => !check.passed);
  if (failed.length === 0) {
    return null;
  }
  return (
    <div>
      <h4>Failed validation checks</h4>
      <ul className="error-details">
        {failed.map((check) => (
          <li key={check.name}>
            {check.name}
            {check.code ? ` [${check.code}]` : ""}
            {check.detail ? `: ${check.detail}` : ""}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ResultDetails({ result }: { result: ToolResult }) {
  if (result.tool === "aggregation_tool") {
    return (
      <dl className="details">
        <dt>Value</dt>
        <dd>{formatNumber(result.value)}</dd>
        <dt>Rows used</dt>
        <dd>{result.rows_used.toLocaleString()}</dd>
        <dt>Rows left out (empty)</dt>
        <dd>{result.rows_excluded_missing.toLocaleString()}</dd>
        {result.order_count !== null && (
          <>
            <dt>Orders counted</dt>
            <dd>
              {result.order_count.toLocaleString()} ({result.order_count_method})
            </dd>
          </>
        )}
      </dl>
    );
  }

  if (result.tool === "grouping_tool") {
    return (
      <div className="table-wrapper">
        <table className="preview-table">
          <thead>
            <tr>
              <th>{result.group_by}</th>
              <th>Value</th>
              <th>Rows</th>
            </tr>
          </thead>
          <tbody>
            {result.groups.map((group) => (
              <tr key={group.group ?? "(empty)"}>
                <td>{group.group ?? "(empty)"}</td>
                <td>{formatNumber(group.value)}</td>
                <td>{group.row_count.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  if (result.tool === "ranking_tool") {
    return (
      <div className="table-wrapper">
        <table className="preview-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>{result.group_by}</th>
              <th>Value</th>
              <th>Rows</th>
            </tr>
          </thead>
          <tbody>
            {result.items.map((item) => (
              <tr key={`${item.rank}-${item.group ?? "(empty)"}`}>
                <td>{item.rank}</td>
                <td>{item.group ?? "(empty)"}</td>
                <td>{formatNumber(item.value)}</td>
                <td>{item.row_count.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  const withMissing = result.columns.filter((column) => column.missing_count > 0);
  return withMissing.length === 0 ? (
    <p className="hint">No column has missing values.</p>
  ) : (
    <div className="table-wrapper">
      <table className="preview-table">
        <thead>
          <tr>
            <th>Column</th>
            <th>Missing</th>
            <th>Missing %</th>
          </tr>
        </thead>
        <tbody>
          {withMissing.map((column) => (
            <tr key={column.name}>
              <td>{column.name}</td>
              <td>{column.missing_count.toLocaleString()}</td>
              <td>{column.missing_percentage}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function QueryResult({ response }: QueryResultProps) {
  if (response.status === "unsupported") {
    const isDefinition = response.error?.details["reason"] === "definition_not_available";
    return (
      <div className="query-result">
        <p className="query-question">Question: {response.question}</p>
        <p className="status status-warning">
          {isDefinition ? "Definitions are coming in Module 7" : "Not supported yet"}
        </p>
        <p>{response.message}</p>
        <p className="hint">{describeClassifier(response)}</p>
      </div>
    );
  }

  if (response.status === "insufficient_data" || response.status === "error") {
    const problem = response.error;
    return (
      <div className="query-result">
        <p className="query-question">Question: {response.question}</p>
        <p className="status status-error">
          {response.status === "insufficient_data"
            ? "This dataset cannot answer that question"
            : "The result could not be produced"}
        </p>
        {problem && (
          <ErrorMessage
            error={new ApiError(problem.message, 200, problem.code, problem.details)}
          />
        )}
        <FailedChecks response={response} />
        <p className="hint">{describeClassifier(response)}</p>
      </div>
    );
  }

  const checks = response.validation.checks;
  const passedChecks = checks.filter((check) => check.passed).length;
  const isDefinitionAnswer = response.calculation_method === "metric_definition";

  return (
    <div className="query-result">
      <p className="query-question">Question: {response.question}</p>
      <p className="answer">{response.explanation}</p>

      <div className="badge-row">
        <span className="badge badge-passed">Outcome: {response.status}</span>
        <span className={`badge badge-${response.validation.status}`}>
          Validation: {response.validation.status}
          {checks.length > 0 ? ` (${passedChecks}/${checks.length} checks)` : ""}
        </span>
        <span className="badge">{describeClassifier(response)}</span>
        {response.tool_used && <span className="badge">Tool: {response.tool_used}</span>}
      </div>

      {isDefinitionAnswer && (
        <p className="hint">Definition only. No calculation was performed.</p>
      )}

      {response.result && <ResultDetails result={response.result} />}

      {response.calculation_method && !isDefinitionAnswer && (
        <p className="hint">Calculation method: {response.calculation_method}</p>
      )}

      {response.assumptions.length > 0 && (
        <div>
          <h4>Assumptions</h4>
          <ul className="warning-list">
            {response.assumptions.map((assumption) => (
              <li key={assumption}>{assumption}</li>
            ))}
          </ul>
        </div>
      )}

      <details className="plan-details">
        <summary>How the question was understood and validated</summary>
        <pre>{JSON.stringify(response.query_plan, null, 2)}</pre>
        <ul className="value-list">
          {checks.map((check) => (
            <li key={check.name}>
              {check.passed ? "Passed" : "Failed"}: {check.name}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

export default QueryResult;