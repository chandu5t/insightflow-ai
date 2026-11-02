import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { BarRenderer, LineRenderer, PieRenderer, ScatterRenderer, VisualizationRenderer } from "./VisualizationRenderer";
import type { VisualizationResponse, VisualizationSpec } from "../types/visualization";

function spec(chart_type: VisualizationSpec["chart_type"] = "bar"): VisualizationSpec {
  const scatter = chart_type === "scatter";
  const pie = chart_type === "pie";
  const rows = scatter
    ? [{ x: 1, y: 2 }, { x: 2, y: 4 }]
    : [{ label: "A", value: 10 }, { label: "B", value: 20 }];
  return {
    visualization_id: "viz-test",
    chart_type,
    title: "Test values",
    description: "A structured test chart",
    source_step_id: "s1",
    x_axis: { field: scatter ? "x" : "label", label: scatter ? "x" : "Label", data_type: scatter ? "numeric" : pie ? "categorical" : "ordered" },
    y_axis: { field: scatter ? "y" : "value", label: scatter ? "y" : "Value", data_type: "numeric" },
    series: [{ name: "Value", field: scatter ? "y" : "value" }],
    data: rows,
    metadata: pie ? { part_to_whole_whole_value: 30 } : {},
    traceability: {
      question: "Show values", dataset_id: "dataset-test", source_step_id: "s1",
      source_operation: "select_columns", verification_status: "passed", correction_status: null,
    },
  };
}

const response = (value: VisualizationSpec): VisualizationResponse => ({
  status: "generated", visualization: value, errors: [], metadata: {},
});

describe("V2.6 fixed SVG renderers", () => {
  afterEach(() => {
    cleanup();
  });
  it("renders bar category labels and exact supplied values", () => {
    render(<BarRenderer spec={spec("bar")} />);
    expect(screen.getByText("A")).toBeTruthy();
    expect(screen.getByText("10")).toBeTruthy();
    expect(document.querySelector('[data-value="20"]')).toBeTruthy();
  });

  it("preserves negative bar direction around a zero baseline", () => {
    const negative = spec("bar");
    negative.data = [{ label: "Gain", value: 10 }, { label: "Loss", value: -5 }];
    render(<BarRenderer spec={negative} />);
    const bars = [...document.querySelectorAll("rect.chart-bar")];
    expect(bars).toHaveLength(2);
    const axisY = Number(document.querySelector("line.chart-axis")?.getAttribute("y1"));
    expect(Number(bars[0].getAttribute("y")) + Number(bars[0].getAttribute("height"))).toBeCloseTo(axisY);
    expect(Number(bars[1].getAttribute("y"))).toBeCloseTo(axisY);
    expect(document.querySelector('[data-value="-5"]')).toBeTruthy();
  });

  it("renders ordered line observations without changing their labels", () => {
    render(<LineRenderer spec={spec("line")} />);
    expect(document.querySelectorAll("polyline")).toHaveLength(1);
    expect(document.querySelector('[data-x="A"][data-y="10"]')).toBeTruthy();
    expect(document.querySelector('[data-x="B"][data-y="20"]')).toBeTruthy();
  });

  it("renders pie slices from the supplied category/value pairs", () => {
    render(<PieRenderer spec={spec("pie")} />);
    expect(document.querySelectorAll("path.chart-slice")).toHaveLength(2);
    expect(document.querySelector('[data-category="A"][data-value="10"]')).toBeTruthy();
    expect(screen.getByText("B: 20")).toBeTruthy();
  });

  it("renders a single complete pie component without changing its value", () => {
    const single = spec("pie");
    single.data = [{ label: "All", value: 30 }];
    render(<PieRenderer spec={single} />);
    expect(document.querySelector("circle.chart-slice[data-value='30']")).toBeTruthy();
  });

  it("renders scatter points from existing paired observations", () => {
    render(<ScatterRenderer spec={spec("scatter")} />);
    expect(document.querySelectorAll("circle.chart-point")).toHaveLength(2);
    expect(document.querySelector('[data-x="1"][data-y="2"]')).toBeTruthy();
  });

  it.each(["bar", "line", "pie", "scatter"] as const)("dispatches controlled chart type %s", (chartType) => {
    render(<VisualizationRenderer response={response(spec(chartType))} />);
    expect(screen.getByRole("img", { name: "Test values" })).toBeTruthy();
  });

  it("renders unsupported state without attempting a renderer", () => {
    render(<VisualizationRenderer response={{ status: "unsupported", visualization: null,
      errors: [{ category: "unsupported_visualization", code: "NO_SEMANTICS", message: "No supported chart." }], metadata: {} }} />);
    expect(screen.getByRole("status").textContent).toBe("No supported chart.");
  });

  it("renders failed state safely", () => {
    render(<VisualizationRenderer response={{ status: "failed", visualization: null, errors: [], metadata: {} }} />);
    expect(screen.getByRole("status").textContent).toBe("No visualization is available.");
  });

  it("handles empty data without crashing", () => {
    const empty = spec("bar");
    empty.data = [];
    render(<VisualizationRenderer response={response(empty)} />);
    expect(screen.getByRole("alert").textContent).toContain("invalid");
  });

  it("rejects a runtime-invalid chart type", () => {
    const invalid = spec("bar");
    (invalid as unknown as { chart_type: string }).chart_type = "area";
    render(<VisualizationRenderer response={response(invalid)} />);
    expect(screen.getByRole("alert").textContent).toContain("Invalid visualization");
  });
});
