import type { Formula, Page } from "./console";

export interface DataField {
  field_id: string; dataset_id: string | null; dataset_name: string | null; category: string | null;
  subcategory: string | null; field_type: string | null; coverage: number | null; description: string | null;
}
export interface Operator {
  operator_name: string; category: string | null; definition: string | null; description: string | null;
  documentation: string | null; level: string | null; scope: string[]; parameters: Record<string, unknown>[]; roles: string[];
}
export interface Catalog<T> extends Page<T> {
  available: boolean; context: { instrument_type: string; region: string; universe: string; delay: number } | null;
  synced_at: string | null; categories: string[]; datasets: { dataset_id: string; dataset_name: string | null }[];
}
export interface Seed extends Formula { root_task_id: string; root_alpha_id: string | null; promoted_at: string }
export interface Seeds extends Page<Seed> { root_count: number }
export interface Check { name: string; status: string; actual: number | null; threshold: number | null }
export interface ResearchMetrics {
  sharpe: number | null; fitness: number | null; turnover: number | null;
  returns: number | null; drawdown: number | null; margin: number | null; pnl: number | null;
}
export interface FormulaDetail {
  summary: Formula; expression: string; settings: Record<string, unknown>; seed_promoted_at: string | null;
  mutation: { action: string; location: string; before: string; after: string } | null;
  result: (ResearchMetrics & { checks: Check[]; check_details_captured: boolean }) | null;
  checks: { source: "backtest" | "full"; observed_at: string | null; error: string | null; items: Check[] };
  yearly: (ResearchMetrics & { year: number; stage: string })[] | null;
  pnl: { observed_at: string; points: [string, number][] | null } | null;
  references: { names: string[]; operators: string[] };
}
export type SeriesMetric = "pnl" | "sharpe" | "turnover";
export interface PlatformSeries {
  alpha_id: string; metric: SeriesMetric; refresh_after_seconds: number;
  state: "ready" | "pending" | "error"; unit: string; observed_at: string;
  points: [string, number | null][] | null; retry_after_seconds: number | null;
  error: string | null; http_status: number | null;
}
