import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import VisualizationRequestPanel from "./VisualizationRequestPanel";
import { runDemoWorkflow, type DemoWorkflowResponse } from "../services/api";
import type { VisualizationResponse } from "../types/visualization";

vi.mock("../services/api", () => ({ runDemoWorkflow: vi.fn() }));

const chart: VisualizationResponse = {
  status: "generated",
  visualization: {
    visualization_id: "viz-demo", chart_type: "bar", title: "Revenue by region", description: "Verified totals",
    source_step_id: "sum_revenue", x_axis: { field: "key", label: "Region", data_type: "categorical" },
    y_axis: { field: "value", label: "Revenue", data_type: "numeric" }, series: [{ name: "Revenue", field: "value" }],
    data: [{ key: "North", value: 10 }], metadata: {},
    traceability: { question: "Show revenue by region", dataset_id: "dataset-current", source_step_id: "sum_revenue",
      source_operation: "aggregate", verification_status: "passed", correction_status: null },
  },
  errors: [], metadata: {},
};

function response(overrides: Partial<DemoWorkflowResponse> = {}): DemoWorkflowResponse {
  return {
    status: "completed", question: "Show revenue by region", dataset_id: "dataset-current",
    stages: { planning: "completed", execution: "completed", verification: "passed", correction: "not_needed", visualization: "generated" },
    planner_response: {}, analysis_plan: {}, execution_result: {}, verification_result: { status: "passed" },
    correction_result: null, final_plan: {}, final_execution_result: {}, final_verification_result: { status: "passed" },
    visualization_result: chart, errors: [], ...overrides,
  };
}

describe("V2 workflow dashboard panel", () => {
  afterEach(() => { cleanup(); vi.clearAllMocks(); });

  it("sends the active uploaded dataset ID and displays a verified chart", async () => {
    vi.mocked(runDemoWorkflow).mockResolvedValue(response());
    render(<VisualizationRequestPanel datasetId="dataset-current" />);
    fireEvent.change(screen.getByLabelText("Analysis question"), { target: { value: "Show revenue by region" } });
    fireEvent.click(screen.getByRole("button", { name: "Run V2 analysis" }));

    await waitFor(() => expect(runDemoWorkflow).toHaveBeenCalledWith("dataset-current", "Show revenue by region"));
    expect(await screen.findByRole("img", { name: "Revenue by region" })).toBeTruthy();
    expect(screen.getByText("passed")).toBeTruthy();
  });

  it("shows a loading state while the workflow request is pending", async () => {
    let finish!: (value: DemoWorkflowResponse) => void;
    vi.mocked(runDemoWorkflow).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    render(<VisualizationRequestPanel datasetId="dataset-current" />);
    fireEvent.change(screen.getByLabelText("Analysis question"), { target: { value: "What is the total revenue?" } });
    fireEvent.click(screen.getByRole("button", { name: "Run V2 analysis" }));

    expect(screen.getByRole("status").textContent).toContain("Planning, executing, and verifying");
    expect(screen.getByRole("button", { name: "Running verified V2 workflow…" }).hasAttribute("disabled")).toBe(true);
    finish(response({ question: "What is the total revenue?", visualization_result: null }));
    expect(await screen.findByText("Workflow: completed")).toBeTruthy();
  });

  it("withholds chart data when final verification did not pass", async () => {
    vi.mocked(runDemoWorkflow).mockResolvedValue(response({
      status: "failed", stages: { planning: "completed", execution: "completed", verification: "failed", correction: "uncorrectable", visualization: "blocked_by_verification" },
      final_verification_result: { status: "failed" }, visualization_result: chart,
    }));
    render(<VisualizationRequestPanel datasetId="dataset-current" />);
    fireEvent.change(screen.getByLabelText("Analysis question"), { target: { value: "Show revenue by region" } });
    fireEvent.click(screen.getByRole("button", { name: "Run V2 analysis" }));

    expect(await screen.findByText(/Visualization withheld/)).toBeTruthy();
    expect(screen.queryByRole("img", { name: "Revenue by region" })).toBeNull();
  });

  it("withholds a chart whose trace points to another dataset", async () => {
    const wrongDataset = { ...chart, visualization: { ...chart.visualization!, traceability: {
      ...chart.visualization!.traceability, dataset_id: "old-dataset",
    } } };
    vi.mocked(runDemoWorkflow).mockResolvedValue(response({ visualization_result: wrongDataset }));
    render(<VisualizationRequestPanel datasetId="dataset-current" />);
    fireEvent.change(screen.getByLabelText("Analysis question"), { target: { value: "Show revenue by region" } });
    fireEvent.click(screen.getByRole("button", { name: "Run V2 analysis" }));

    expect(await screen.findByText(/Visualization withheld/)).toBeTruthy();
    expect(screen.queryByRole("img", { name: "Revenue by region" })).toBeNull();
  });

  it("shows backend unsupported outcomes and network errors", async () => {
    vi.mocked(runDemoWorkflow).mockResolvedValue(response({
      status: "unsupported", stages: { planning: "unsupported", execution: "pending", verification: "pending", correction: "not_needed", visualization: "not_ready" },
      final_verification_result: null, visualization_result: null,
      errors: [{ stage: "planning", code: "UNSUPPORTED_ANALYSIS", message: "The requested metric is not registered." }],
    }));
    const { rerender } = render(<VisualizationRequestPanel datasetId="dataset-current" />);
    fireEvent.change(screen.getByLabelText("Analysis question"), { target: { value: "Unknown metric" } });
    fireEvent.click(screen.getByRole("button", { name: "Run V2 analysis" }));
    expect(await screen.findByText(/UNSUPPORTED_ANALYSIS/)).toBeTruthy();
    expect(screen.queryByRole("img")).toBeNull();

    vi.mocked(runDemoWorkflow).mockRejectedValueOnce(new Error("Cannot reach backend"));
    rerender(<VisualizationRequestPanel datasetId="dataset-current" />);
    fireEvent.change(screen.getByLabelText("Analysis question"), { target: { value: "Try again" } });
    fireEvent.click(screen.getByRole("button", { name: "Run V2 analysis" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Cannot reach backend");
  });
});
