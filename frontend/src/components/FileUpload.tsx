import { useState, type ChangeEvent, type FormEvent } from "react";
import { uploadDataset } from "../services/api";
import type { DatasetSummary } from "../types/dataset";
import ErrorMessage from "./ErrorMessage";

type UploadState =
  | { phase: "idle" }
  | { phase: "uploading" }
  | { phase: "success"; dataset: DatasetSummary }
  | { phase: "error"; error: Error };

type FileUploadProps = {
  onUploaded: (dataset: DatasetSummary) => void;
};

function FileUpload({ onUploaded }: FileUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [state, setState] = useState<UploadState>({ phase: "idle" });

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    setSelectedFile(event.target.files?.[0] ?? null);
    setState({ phase: "idle" });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFile) {
      return;
    }

    setState({ phase: "uploading" });

    try {
      const dataset = await uploadDataset(selectedFile);
      setState({ phase: "success", dataset });
      onUploaded(dataset);
    } catch (error: unknown) {
      setState({
        phase: "error",
        error:
          error instanceof Error
            ? error
            : new Error("An unknown error occurred."),
      });
    }
  }

  const isUploading = state.phase === "uploading";

  return (
    <div>
      <form
        className="upload-form"
        onSubmit={(event) => {
          void handleSubmit(event);
        }}
      >
        <input
          type="file"
          accept=".csv,.xlsx"
          onChange={handleFileChange}
          disabled={isUploading}
          aria-label="Choose a CSV or XLSX file"
        />
        <button type="submit" disabled={!selectedFile || isUploading}>
          {isUploading ? "Uploading..." : "Upload"}
        </button>
      </form>

      <p className="hint">
        Supported files: .csv and .xlsx (first sheet only). The first row must
        contain the column names.
      </p>

      {state.phase === "uploading" && (
        <p className="status status-loading">Uploading and validating...</p>
      )}

      {state.phase === "error" && <ErrorMessage error={state.error} />}

      {state.phase === "success" && (
        <div>
          <p className="status status-healthy">
            Upload successful ({state.dataset.status})
          </p>
          <p className="hint">Dataset ID: {state.dataset.dataset_id}</p>

          {state.dataset.warnings.length > 0 && (
            <ul className="warning-list">
              {state.dataset.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

export default FileUpload;