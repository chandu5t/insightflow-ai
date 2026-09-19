import { useState } from "react";
import DatasetPreview from "../components/DatasetPreview";
import FileUpload from "../components/FileUpload";
import HealthStatus from "../components/HealthStatus";
import type { DatasetSummary } from "../types/dataset";
import "../styles/dataset.css";

function Dashboard() {
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);

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
        <FileUpload onUploaded={setDataset} />
      </section>

      {dataset && (
        <section className="card">
          <h2>2. Data preview</h2>
          {/* key makes React start fresh when a new file is uploaded */}
          <DatasetPreview key={dataset.dataset_id} datasetId={dataset.dataset_id} />
        </section>
      )}
    </main>
  );
}

export default Dashboard;