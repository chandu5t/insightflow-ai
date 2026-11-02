import type { ReactNode } from "react";
import type { ChartType, VisualizationResponse, VisualizationSpec } from "../types/visualization";

type Point = { label: string; value: number; row: Record<string, unknown> };

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function pointsFor(spec: VisualizationSpec, xField = spec.x_axis.field, yField = spec.y_axis?.field): Point[] | null {
  if (!yField || !Array.isArray(spec.data)) return null;
  const points: Point[] = [];
  for (const row of spec.data) {
    const value = asNumber(row[yField]);
    if (value === null || row[xField] === null || row[xField] === undefined) return null;
    points.push({ label: String(row[xField]), value, row });
  }
  return points;
}

function svgLabel(value: unknown): string {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean"
    ? String(value)
    : "";
}

function ChartFrame({ title, children }: { title: string; children: ReactNode }) {
  return <svg className="visualization-svg" viewBox="0 0 640 360" role="img" aria-label={title}>{children}</svg>;
}

function isVisualizationSpec(value: unknown): value is VisualizationSpec {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<VisualizationSpec>;
  return ["bar", "line", "pie", "scatter"].includes(candidate.chart_type ?? "")
    && typeof candidate.visualization_id === "string"
    && typeof candidate.title === "string"
    && typeof candidate.source_step_id === "string"
    && typeof candidate.x_axis?.field === "string"
    && Array.isArray(candidate.data)
    && typeof candidate.traceability?.source_step_id === "string";
}

export function BarRenderer({ spec }: { spec: VisualizationSpec }) {
  const points = pointsFor(spec);
  if (!points?.length) return <p role="alert">This visualization data is invalid.</p>;
  const minValue = Math.min(0, ...points.map((point) => point.value));
  const maxValue = Math.max(0, ...points.map((point) => point.value));
  const span = maxValue - minValue || 1;
  const valueY = (value: number) => 300 - ((value - minValue) / span) * 220;
  const baselineY = valueY(0);
  const barWidth = Math.min(54, 500 / points.length - 8);
  return (
    <ChartFrame title={spec.title}>
      <line x1="70" y1={baselineY} x2="600" y2={baselineY} className="chart-axis" />
      {points.map((point, index) => {
        const endY = valueY(point.value);
        const height = Math.abs(baselineY - endY);
        const x = 84 + index * (500 / points.length);
        const labelY = point.value < 0 ? endY + 16 : endY - 6;
        return (
          <g key={`${point.label}-${index}`} data-category={point.label} data-value={String(point.row[spec.y_axis!.field])}>
            <title>{`${point.label}: ${svgLabel(point.row[spec.y_axis!.field])}`}</title>
            <rect x={x} y={Math.min(baselineY, endY)} width={barWidth} height={height} className="chart-bar" />
            <text x={x + barWidth / 2} y="326" textAnchor="middle">{point.label}</text>
            <text x={x + barWidth / 2} y={labelY} textAnchor="middle">{svgLabel(point.row[spec.y_axis!.field])}</text>
          </g>
        );
      })}
    </ChartFrame>
  );
}

export function LineRenderer({ spec }: { spec: VisualizationSpec }) {
  const points = pointsFor(spec);
  if (!points || points.length < 2) return <p role="alert">This visualization data is invalid.</p>;
  const min = Math.min(...points.map((point) => point.value));
  const max = Math.max(...points.map((point) => point.value));
  const span = max - min || 1;
  const coords = points.map((point, index) => ({
    ...point,
    x: 70 + (index * 530) / (points.length - 1),
    y: 295 - ((point.value - min) / span) * 220,
  }));
  return (
    <ChartFrame title={spec.title}>
      <polyline points={coords.map((point) => `${point.x},${point.y}`).join(" ")} className="chart-line" />
      {coords.map((point, index) => (
        <g key={`${point.label}-${index}`} data-x={String(point.row[spec.x_axis.field])} data-y={String(point.row[spec.y_axis!.field])}>
          <title>{`${point.label}: ${svgLabel(point.row[spec.y_axis!.field])}`}</title>
          <circle cx={point.x} cy={point.y} r="4" className="chart-point" />
          <text x={point.x} y="326" textAnchor="middle">{point.label}</text>
        </g>
      ))}
    </ChartFrame>
  );
}

function polar(cx: number, cy: number, radius: number, angle: number) {
  return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
}

function piePath(start: number, end: number) {
  const center = 180;
  const radius = 140;
  const first = polar(center, center, radius, start);
  const last = polar(center, center, radius, end);
  const large = end - start > Math.PI ? 1 : 0;
  return `M ${center} ${center} L ${first.x} ${first.y} A ${radius} ${radius} 0 ${large} 1 ${last.x} ${last.y} Z`;
}

export function PieRenderer({ spec }: { spec: VisualizationSpec }) {
  const points = pointsFor(spec);
  if (!points?.length || points.some((point) => point.value < 0)) {
    return <p role="alert">This visualization data is invalid.</p>;
  }
  const whole = spec.metadata.part_to_whole_whole_value;
  if (typeof whole !== "number" || !Number.isFinite(whole) || whole <= 0) {
    return <p role="alert">This visualization data is invalid.</p>;
  }
  if (points.length === 1 && points[0].value === whole) {
    return (
      <ChartFrame title={spec.title}>
        <circle cx="180" cy="180" r="140" className="chart-slice chart-slice-0"
          data-category={points[0].label} data-value={String(points[0].row[spec.y_axis!.field])}
          aria-label={`${points[0].label}: ${svgLabel(points[0].row[spec.y_axis!.field])}`} />
        <text x="370" y="45">{`${points[0].label}: ${svgLabel(points[0].row[spec.y_axis!.field])}`}</text>
      </ChartFrame>
    );
  }
  const slices = points.map((point, index) => {
    const start = -Math.PI / 2 + points.slice(0, index)
      .reduce((total, previous) => total + (previous.value / whole) * Math.PI * 2, 0);
    return { point, start, end: start + (point.value / whole) * Math.PI * 2 };
  });
  return (
    <ChartFrame title={spec.title}>
      {slices.map(({ point, start, end }, index) => {
        return (
          <path
            key={`${point.label}-${index}`}
            d={piePath(start, end)}
            className={`chart-slice chart-slice-${index % 8}`}
            data-category={point.label}
            data-value={String(point.row[spec.y_axis!.field])}
            aria-label={`${point.label}: ${svgLabel(point.row[spec.y_axis!.field])}`}
          />
        );
      })}
      <g className="chart-legend">
        {points.map((point, index) => (
          <text key={`${point.label}-legend`} x="370" y={45 + index * 22}>{`${point.label}: ${svgLabel(point.row[spec.y_axis!.field])}`}</text>
        ))}
      </g>
    </ChartFrame>
  );
}

export function ScatterRenderer({ spec }: { spec: VisualizationSpec }) {
  const xField = spec.x_axis.field;
  const yField = spec.y_axis?.field;
  const rows = spec.data;
  const numericRows = rows.map((row) => ({ row, x: asNumber(row[xField]), y: yField ? asNumber(row[yField]) : null }));
  if (numericRows.length < 2 || numericRows.some((point) => point.x === null || point.y === null)) {
    return <p role="alert">This visualization data is invalid.</p>;
  }
  const xs = numericRows.map((point) => point.x!);
  const ys = numericRows.map((point) => point.y!);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scaleX = (value: number) => 70 + ((value - minX) / (maxX - minX || 1)) * 530;
  const scaleY = (value: number) => 300 - ((value - minY) / (maxY - minY || 1)) * 250;
  return (
    <ChartFrame title={spec.title}>
      <line x1="70" y1="300" x2="600" y2="300" className="chart-axis" />
      <line x1="70" y1="40" x2="70" y2="300" className="chart-axis" />
      {numericRows.map(({ row, x, y }, index) => (
        <circle key={index} cx={scaleX(x!)} cy={scaleY(y!)} r="5" className="chart-point"
          data-x={String(row[xField])} data-y={String(row[yField!])}>
          <title>{`${svgLabel(row[xField])}, ${svgLabel(row[yField!])}`}</title>
        </circle>
      ))}
    </ChartFrame>
  );
}

const RENDERERS: Record<ChartType, (props: { spec: VisualizationSpec }) => ReactNode> = {
  bar: BarRenderer,
  line: LineRenderer,
  pie: PieRenderer,
  scatter: ScatterRenderer,
};

export function VisualizationRenderer({ response }: { response: VisualizationResponse }) {
  if (response.status !== "generated" || response.visualization === null) {
    const message = Array.isArray(response.errors) && typeof response.errors[0]?.message === "string"
      ? response.errors[0].message
      : "No visualization is available.";
    return <div role="status" className={`visualization-status visualization-${response.status}`}>{message}</div>;
  }
  const spec: unknown = response.visualization;
  if (!isVisualizationSpec(spec)) {
    return <div role="alert" className="visualization-status visualization-failed">Invalid visualization specification.</div>;
  }
  const Renderer = RENDERERS[spec.chart_type];
  return (
    <section className="visualization-result" aria-labelledby={`viz-${spec.visualization_id}`}>
      <h3 id={`viz-${spec.visualization_id}`}>{spec.title}</h3>
      <p>{spec.description}</p>
      <Renderer spec={spec} />
      <p className="visualization-trace">Source step: {spec.traceability.source_step_id} · Verification: {spec.traceability.verification_status}</p>
    </section>
  );
}
