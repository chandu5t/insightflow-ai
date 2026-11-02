import { useState, type FormEvent } from "react";
import { runDemoWorkflow, type DemoWorkflowResponse } from "../services/api";
import { VisualizationRenderer } from "./VisualizationRenderer";

type Props = { datasetId: string };

function artifact(value: unknown): string {
  return value === null || value === undefined ? "Not produced" : JSON.stringify(value, null, 2);
}

function hasPassedFinalVerification(value: unknown): boolean {
  return typeof value === "object" && value !== null
    && (value as { status?: unknown }).status === "passed";
}

function traceMatchesDataset(response: DemoWorkflowResponse, datasetId: string): boolean {
  const visualization = response.visualization_result?.visualization;
  if (!visualization) return true;
  return visualization.traceability.dataset_id === datasetId
    && visualization.traceability.verification_status === "passed"
    && visualization.traceability.source_step_id === visualization.source_step_id;
}

export default function VisualizationRequestPanel({ datasetId }: Props) {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<DemoWorkflowResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);
    const text = question.trim();
    if (!text) {
      setError("Enter a question about the uploaded dataset.");
      return;
    }
    setBusy(true);
    try {
      setResult(await runDemoWorkflow(datasetId, text));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The V2 workflow request failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="visualization-request">
      <p className="hint">
        This runs the V2 planner, controlled analysis workflow, independent verification, and bounded correction when applicable.
        A chart is requested only after final verification passes.
      </p>
      <form onSubmit={submit}>
        <label htmlFor="v2-question">Analysis question</label>
        <textarea
          id="v2-question"
          rows={3}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="For example: Show revenue by region"
          disabled={busy}
        />
        <div className="example-list">
          {["What is the total revenue?", "Show revenue by region."].map((example) => (
            <button key={example} type="button" className="secondary-button example-button"
              disabled={busy} onClick={() => setQuestion(example)}>
              {example}
            </button>
          ))}
        </div>
        <button type="submit" disabled={busy}>
          {busy ? "Running verified V2 workflow…" : "Run V2 analysis"}
        </button>
      </form>
      <p className="hint data-integrity-note">
        Data handling: uploaded rows are preserved, including exact duplicates. Missing grouping values are shown as “(missing)”;
        the workflow does not invent or impute a category.
      </p>
      {busy && <p role="status" className="status status-loading">Planning, executing, and verifying this question…</p>}
      {error && <p role="alert" className="status-error">{error}</p>}
      {result && (
        <section className="v2-workflow-result" aria-label="V2 workflow result">
          <h3>Workflow: {result.status}</h3>
          <dl className="workflow-stages">
            {Object.entries(result.stages).map(([stage, status]) => (
              <div key={stage}><dt>{stage.replaceAll("_", " ")}</dt><dd>{status}</dd></div>
            ))}
          </dl>
          {result.errors.length > 0 && (
            <div role="alert" className="workflow-errors">
              <h4>What needs attention</h4>
              <ul>{result.errors.map((item, index) => (
                <li key={`${item.stage}-${item.code}-${index}`}><strong>{item.stage}: {item.code}</strong> — {item.message}</li>
              ))}</ul>
            </div>
          )}
          <details><summary>Validated plan</summary><pre>{artifact(result.final_plan ?? result.analysis_plan)}</pre></details>
          <details><summary>Execution result</summary><pre>{artifact(result.final_execution_result ?? result.execution_result)}</pre></details>
          <details><summary>Verification result</summary><pre>{artifact(result.final_verification_result ?? result.verification_result)}</pre></details>
          {result.correction_result !== null && (
            <details><summary>Correction trace</summary><pre>{artifact(result.correction_result)}</pre></details>
          )}
          {result.visualization_result && hasPassedFinalVerification(result.final_verification_result)
            && traceMatchesDataset(result, datasetId)
            ? <VisualizationRenderer response={result.visualization_result} />
            : result.visualization_result && <p role="status" className="status-warning">
              Visualization withheld because the final verification or dataset trace does not pass the display checks.
            </p>}
        </section>
      )}
    </div>
  );
}
