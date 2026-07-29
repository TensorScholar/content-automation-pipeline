export type Role = "manager" | "admin" | "user" | string;

export interface User {
  id: string;
  email: string;
  full_name?: string;
  role?: Role;
  is_superuser?: boolean;
  is_active?: boolean;
  created_at?: string;
}

export interface Project {
  id: string;
  name: string;
  domain?: string;
  vertical?: string;
  description?: string;
  wordpress_url?: string;
  wordpress_username?: string;
}

export type ProjectReadinessStatus = "ready" | "warning" | "blocked" | string;
export type ProjectReadinessCheckStatus = "pass" | "warn" | "fail" | string;
export type ProjectReadinessSeverity = "info" | "warning" | "blocking" | string;

export interface ProjectReadinessCheck {
  id: string;
  label: string;
  status: ProjectReadinessCheckStatus;
  severity: ProjectReadinessSeverity;
  message: string;
  remediation?: string | null;
}

export interface ProjectReadinessAction {
  id: string;
  label: string;
  method: string;
  endpoint?: string | null;
  destructive: boolean;
}

export interface ProjectReadiness {
  project_id: string;
  status: ProjectReadinessStatus;
  can_generate: boolean;
  can_publish: boolean;
  blocking_items: ProjectReadinessCheck[];
  warnings: ProjectReadinessCheck[];
  checks: ProjectReadinessCheck[];
  manager_actions: ProjectReadinessAction[];
  last_checked_at: string;
}

export type PerformanceOpportunityType =
  | "low_ctr_high_impressions"
  | "striking_distance_position"
  | "declining_clicks"
  | "missing_performance_data"
  | "unmapped_url"
  | string;

export type PerformanceOpportunitySeverity = "low" | "medium" | "high" | string;
export type PerformanceOpportunityStatus = "open" | "dismissed" | "resolved" | string;

export interface PerformanceSummary {
  snapshot_count: number;
  opportunity_count: number;
  high_priority_count: number;
  latest_imported_at?: string | null;
}

export interface PerformanceSnapshot {
  id: string;
  project_id: string;
  article_id?: string | null;
  url: string;
  date_from: string;
  date_to: string;
  clicks: number;
  impressions: number;
  ctr: number;
  average_position: number;
  source: string;
  imported_at?: string | null;
}

export interface PerformanceOpportunity {
  id: string;
  project_id: string;
  article_id?: string | null;
  snapshot_id?: string | null;
  article_title?: string | null;
  url: string;
  type: PerformanceOpportunityType;
  severity: PerformanceOpportunitySeverity;
  reason: string;
  suggested_action: string;
  supporting_metrics: Record<string, number | string | null | undefined>;
  status: PerformanceOpportunityStatus;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface ProjectPerformanceFeedback {
  project_id: string;
  summary: PerformanceSummary;
  snapshots: PerformanceSnapshot[];
  opportunities: PerformanceOpportunity[];
}

export interface PerformanceImportResponse {
  project_id: string;
  imported_count: number;
  snapshot_count: number;
  opportunity_count: number;
  opportunities_created_or_updated: number;
}

export interface LlmModelOption {
  provider: string;
  model: string;
  label: string;
  enabled: boolean;
  recommended?: boolean;
  reason?: string | null;
}

export interface LlmProviderOption {
  provider: string;
  label: string;
  configured: boolean;
  active: boolean;
  models: LlmModelOption[];
}

export interface LlmOptionsResponse {
  active_model: string;
  active_provider: string;
  fallback_model?: string | null;
  selectable_models: LlmModelOption[];
  providers: LlmProviderOption[];
  warnings: string[];
  user_message: string;
  manager_detail?: string | null;
  generated_at: string;
}

export interface TaskHistoryItem {
  id?: string;
  task_id: string;
  status: string;
  task_name?: string;
  topic?: string;
  result?: unknown;
  error?: string;
  error_code?: string;
  manager_error_detail?: string;
  retry_count?: number;
  created_at?: string;
  updated_at?: string;
  start_time?: string;
  end_time?: string;
}

export interface TaskStatusResponse {
  task_id: string;
  state: string;
  ready: boolean;
  state_source?: string;
  status?: string;
  progress?: number;
  retry_count?: number;
  error?: string;
  error_code?: string;
  manager_error_detail?: string;
  last_error?: string;
  quality_diagnostics?: {
    actual_word_count?: number;
    min_word_count?: number;
    max_word_count?: number;
    headings_count?: number;
    paragraphs_count?: number;
    language?: string;
    regeneration_attempted?: boolean;
    findings?: Array<{
      code?: string;
      message?: string;
      expected?: string;
      actual?: string;
    }>;
  } | null;
  result?: {
    article_id?: string;
    social_task_id?: string;
    project_id?: string;
    posts?: unknown;
    [key: string]: unknown;
  };
}

export interface Article {
  id?: string;
  project_id?: string;
  title: string;
  content: string;
  word_count?: number;
  quality_score?: number;
  generated_at?: string;
}

export interface ArticleDetail {
  id: string;
  project_id?: string;
  title: string;
  content: string;
  html_content?: string;
  word_count?: number;
  quality_score?: number;
  generated_at?: string;
  cost_usd?: number;
  language?: string;
  primary_keyword?: string;
  seo_analysis?: SeoAnalysis;
}

export type ArticleReviewStatus = "pending_review" | "approved" | "changes_requested" | "rejected" | string;
export type ArticleReviewAction = "approve" | "reject" | "request_changes";

export interface ArticleReviewChecklistItem {
  id: string;
  label: string;
  passed: boolean;
  blocking: boolean;
}

export interface ArticleReviewState {
  article_id: string;
  status: ArticleReviewStatus;
  note?: string | null;
  reviewed_by?: string | null;
  reviewer_name?: string | null;
  reviewed_at?: string | null;
  updated_at?: string | null;
  can_approve: boolean;
  blocking_reasons: string[];
  checklist: ArticleReviewChecklistItem[];
  risk_level?: string;
}

export interface DraftRiskIssue {
  id: string;
  severity: "blocking" | "warning" | "info" | string;
  category: string;
  message: string;
  suggested_fix: string;
}

export interface DraftRiskAssessment {
  article_id: string;
  overall_score: number;
  risk_level: "low" | "medium" | "high" | "blocked" | string;
  blocking_issues: DraftRiskIssue[];
  warnings: DraftRiskIssue[];
  issues: DraftRiskIssue[];
  suggested_fixes: string[];
}

export interface SeoAnalysis {
  score?: number;
  checklist?: SeoChecklistItem[];
  recommendations?: string[];
}

export interface SeoChecklistItem {
  label: string;
  passed: boolean;
  detail?: string;
}
