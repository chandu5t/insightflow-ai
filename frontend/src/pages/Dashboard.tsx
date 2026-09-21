import { useState } from "react";
import DatasetPreview from "../components/DatasetPreview";
import DatasetProfile from "../components/DatasetProfile";
import FileUpload from "../components/FileUpload";
import HealthStatus from "../components/HealthStatus";
import QuestionPanel from "../components/QuestionPanel";
import type { DatasetSummary } from "../types/dataset";
import "../styles/dataset.css";
import "../styles/profile.css";
import "../styles/query.css";

function Dashboard() {
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [showProfile, setShowProfile] = useState(false);

  function handleUploaded(uploaded: DatasetSummary) {
    setDataset(uploaded);
    setShowProfile(false); // a new file starts with a closed profile
  }

  function handleUploadStart() {
  setDataset(null);
  setShowProfile(false);
  }

  return (
    <main className="container">
      <header>
        <h1>InsightFlow AI</h1>
        <p className="subtitle">Business Data Analyst Agent</p>
      </header>

      <section className="card">
        <h2>System status</h2>
        <HealthStatus />
      </section>

      <section className="card">
        <h2>1. Upload data</h2>
        <FileUpload
          onUploaded={handleUploaded}
          onUploadStart={handleUploadStart}
        />
      </section>

      {dataset && (
        <section className="card">
          <h2>2. Data preview</h2>
          {/* key makes React start fresh when a new file is uploaded */}
          <DatasetPreview key={dataset.dataset_id} datasetId={dataset.dataset_id} />
        </section>
      )}

      {dataset && (
        <section className="card">
          <h2>3. Dataset profile</h2>
          {showProfile ? (
            <div>
              <button
                type="button"
                className="secondary-button"
                onClick={() => setShowProfile(false)}
              >
                Hide profile
              </button>
              <DatasetProfile
                key={dataset.dataset_id}
                datasetId={dataset.dataset_id}
              />
            </div>
          ) : (
            <div>
              <p className="hint">
                See column types, missing values, statistics and duplicate rows.
              </p>
              <button type="button" onClick={() => setShowProfile(true)}>
                View profile
              </button>
            </div>
          )}
        </section>
      )}

      {dataset && (
        <section className="card">
          <h2>4. Ask a question</h2>
          {/* key resets the question, the answer and any error when a new dataset is uploaded */}
          <QuestionPanel key={dataset.dataset_id} datasetId={dataset.dataset_id} />
        </section>
      )}
    </main>
  );
}

export default Dashboard;