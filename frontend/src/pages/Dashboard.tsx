import HealthStatus from "../components/HealthStatus";

function Dashboard() {
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
    </main>
  );
}

export default Dashboard;