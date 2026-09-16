export interface DiagnosticRecord {
  task_id: string; alpha_id: string | null; grade: string | null; source: string;
  sharpe: number | null; fitness: number | null; turnover: number | null; finished_at: string;
  metric_group: string; platform_state: string; failed_checks: string[];
}
export interface QualityDiagnosis {
  configuration: string; configurations: { id: string; count: number; settings: Record<string, unknown> }[];
  runs: string[]; since: string | null; until: string;
  records: DiagnosticRecord[]; total: number; completed: number; inflight: number; errors: number; analyzable: number;
  thresholds: [number, number] | null;
  reasons: { name: string; passed: number; failed: number; unresolved: number }[];
  platform: Record<string, number>;
}
export interface CorrelationRecord {
  task_id: string; state: string; maximum: number | null; reference_id: string | null;
  compared: number; required: number; high_pairs: number; improved_pairs: number;
}
export interface DiagnosticCorrelation {
  configuration: string; records: CorrelationRecord[]; counts: Record<string, number>;
  reference_count: number; cutoff: number; minimum_intervals: number; improvement_passed: number;
}
