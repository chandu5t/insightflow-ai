import { useState } from "react";
import { ApiError } from "../services/api";
import { visualizeAnalysis } from "../services/api";
import type { VisualizationRequest, VisualizationResponse } from "../types/visualization";
import { VisualizationRenderer } from "./VisualizationRenderer";

const EXAMPLE = `{
  "question": "Show the verified comparison",
  "dataset_id": "00000000-0000-0000-0000-000000000000",
  "analysis_plan": {},
  "execution_result": {},
  "verification_result": {}
}`;

export default function VisualizationRequestPanel() {
  const [payload, setPayload] = useState(EXAMPLE);
  const [result, setResult] = useState<VisualizationResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    setError("");
    setResult(null);
    let request: VisualizationRequest;
    try {
      request = JSON.parse(payload) as VisualizationRequest;
    } catch {
      setError("Enter a valid JSON request containing the verified V2.2–V2.5 artifacts.");
      return;
    }
    setBusy(true);
    try {
      setResult(await visualizeAnalysis(request));
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "The visualization request failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="visualization-request">
      <p className="hint">Paste a V2.2–V2.5 artifact request. The visualization service does not run analysis or correction.</p>
      <label htmlFor="visualization-payload">Verified analysis artifacts (JSON)</label>
      <textarea id="visualization-payload" rows={10} value={payload} onChange={(event) => setPayload(event.target.value)} />
      <button type="button" onClick={submit} disabled={busy}>{busy ? "Preparing visualization…" : "Visualize verified result"}</button>
      {error && <p role="alert" className="status-error">{error}</p>}
      {result && <VisualizationRenderer response={result} />}
    </div>
  );
}
