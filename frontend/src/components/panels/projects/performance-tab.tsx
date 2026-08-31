"use client";

import clsx from "clsx";
import { useToast } from "@/components/ui/toast";
import { Button } from "@/components/ui/button";
import { SelectDropdown } from "@/components/ui/select-dropdown";
import { useI18n } from "@/i18n/provider";
import {
  VERTICAL_OPTIONS,
  READINESS_COPY,
  READINESS_ITEM_COPY,
  PERFORMANCE_COPY,
  SEARCH_CONSOLE_COPY,
  SEO_INTELLIGENCE_COPY,
  SEO_NEXT_ACTION_COPY,
  SEO_WARNING_COPY,
  PROJECT_ERROR_COPY,
  readinessItemKind,
  localizeReadinessLabel,
  localizeReadinessText,
  formatReadinessDate,
  extractError,
  localizeProjectError,
} from "./project-constants";
import type { ReadinessLocale } from "./project-constants";
import { performanceSeverityClasses, performanceTypeLabel, formatCompactNumber, formatFixedNumber, formatCtr, formatShortDate, metricFromOpportunity } from "./project-performance-helpers";
import { Project, ProjectPerformanceFeedback, ProjectReadiness, SearchConsoleStatus, SeoIntelligenceResponse, PerformanceSnapshot, PerformanceOpportunity } from "@/types/models";
import { SearchConsoleCard } from "./search-console-card";
import { SeoIntelligenceCard } from "./seo-intelligence-card";

export function readinessStatusClasses(status: string) {
  if (status === "ready" || status === "pass") {
    return "border-success/20 bg-success/10 text-success";
  }
  if (status === "blocked" || status === "fail") {
    return "border-danger/20 bg-danger/10 text-danger";
  }
  return "border-warning/20 bg-warning/10 text-warning";
}

export function readinessDotClasses(status: string) {
  if (status === "ready" || status === "pass") return "bg-success";
  if (status === "blocked" || status === "fail") return "bg-danger";
  return "bg-warning";
}

export function PerformanceTab({
  copy,
  locale,
  canManageProjects,
  feedback,
  loading,
  error,
  dismissingOpportunityId,
  onRefresh,
  seoIntelligence,
  seoIntelligenceCopy,
  seoIntelligenceLoading,
  seoIntelligenceError,
  onOpenImport,
  onDismiss,
  searchConsole,
  searchConsoleCopy,
  searchConsoleLoading,
  searchConsoleAction,
  searchConsoleError,
  onConnectSearchConsole,
  onRefreshSearchConsole,
  onRefreshSearchConsoleProperties,
  onSelectSearchConsoleProperty,
  onSyncSearchConsole,
  onDisconnectSearchConsole,
}: {
  copy: typeof PERFORMANCE_COPY.en;
  locale: ReadinessLocale;
  canManageProjects: boolean;
  feedback: ProjectPerformanceFeedback | null;
  loading: boolean;
  error: string | null;
  dismissingOpportunityId: string | null;
  onRefresh: () => void;
  seoIntelligence: SeoIntelligenceResponse | null;
  seoIntelligenceCopy: typeof SEO_INTELLIGENCE_COPY.en;
  seoIntelligenceLoading: boolean;
  seoIntelligenceError: string | null;
  onOpenImport: () => void;
  onDismiss: (opportunityId: string) => void;
  searchConsole: SearchConsoleStatus | null;
  searchConsoleCopy: typeof SEARCH_CONSOLE_COPY.en;
  searchConsoleLoading: boolean;
  searchConsoleAction: string | null;
  searchConsoleError: string | null;
  onConnectSearchConsole: () => void;
  onRefreshSearchConsole: () => void;
  onRefreshSearchConsoleProperties: () => void;
  onSelectSearchConsoleProperty: (siteUrl: string) => void;
  onSyncSearchConsole: () => void;
  onDisconnectSearchConsole: () => void;
}) {
  const hasData = Boolean(
    feedback && (feedback.snapshots.length > 0 || feedback.opportunities.length > 0)
  );

  return (
    <div className="max-w-5xl space-y-4 animate-fade-in">
      <SearchConsoleCard
        copy={searchConsoleCopy}
        locale={locale}
        canManageProjects={canManageProjects}
        status={searchConsole}
        loading={searchConsoleLoading}
        action={searchConsoleAction}
        error={searchConsoleError}
        onConnect={onConnectSearchConsole}
        onRefresh={onRefreshSearchConsole}
        onRefreshProperties={onRefreshSearchConsoleProperties}
        onSelectProperty={onSelectSearchConsoleProperty}
        onSync={onSyncSearchConsole}
        onDisconnect={onDisconnectSearchConsole}
      />
      <SeoIntelligenceCard
        payload={seoIntelligence}
        copy={seoIntelligenceCopy}
        locale={locale}
        loading={seoIntelligenceLoading}
        error={seoIntelligenceError}
      />
      <section className="smx-panel-subtle p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-xs font-medium text-ink-muted">{copy.subtitle}</p>
            <h3 className="mt-1 text-xl font-semibold tracking-tight text-ink">
              {copy.title}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outlined" size="sm" loading={loading} onClick={onRefresh}>
              {copy.refresh}
            </Button>
            {canManageProjects && (
              <Button variant="primary" size="sm" onClick={onOpenImport}>
                {copy.import}
              </Button>
            )}
          </div>
        </div>

        {loading && !feedback && (
          <div className="mt-5 rounded-lg border border-line bg-surface-alt px-4 py-3 text-sm font-medium text-ink-secondary">
            {copy.loading}
          </div>
        )}

        {error && (
          <div className="mt-5 rounded-lg border border-danger/20 bg-danger/10 px-4 py-3 text-sm font-medium text-danger" role="alert">
            {copy.failed} {localizeProjectError(error, locale)}
          </div>
        )}

        {feedback && (
          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <PerformanceSummaryCard label={copy.snapshots} value={feedback.summary.snapshot_count} />
            <PerformanceSummaryCard label={copy.opportunities} value={feedback.summary.opportunity_count} />
            <PerformanceSummaryCard label={copy.highPriority} value={feedback.summary.high_priority_count} tone={feedback.summary.high_priority_count > 0 ? "warning" : "default"} />
            <PerformanceSummaryCard
              label={copy.latestImport}
              value={feedback.summary.latest_imported_at ? formatShortDate(feedback.summary.latest_imported_at, locale) : copy.noImport}
              valueClassName="text-base"
            />
          </div>
        )}
      </section>

      {feedback && !hasData && !loading && (
        <section className="rounded-xl border border-dashed border-line bg-surface p-6 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-brand/15 bg-brand/10 text-brand">
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 19V5m0 14h16M8 16v-5m4 5V8m4 8v-7" />
            </svg>
          </div>
          <h4 className="text-body-lg font-semibold text-ink">{copy.emptyTitle}</h4>
          <p className="mx-auto mt-2 max-w-xl text-sm leading-5 text-ink-muted">
            {copy.emptyBody}
          </p>
          {canManageProjects && (
            <Button variant="outlined" size="sm" className="mt-5" onClick={onOpenImport}>
              {copy.import}
            </Button>
          )}
        </section>
      )}

      {feedback && hasData && (
        <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
          <div className="rounded-xl border border-line bg-surface">
            <div className="border-b border-line px-4 py-3">
              <h4 className="text-base font-semibold text-ink">{copy.opportunitiesTitle}</h4>
            </div>
            {feedback.opportunities.length === 0 ? (
              <p className="px-4 py-6 text-sm leading-5 text-ink-muted">{copy.noOpportunities}</p>
            ) : (
              <div className="divide-y divide-line">
                {feedback.opportunities.map((opportunity) => (
                  <PerformanceOpportunityCard
                    key={opportunity.id}
                    copy={copy}
                    opportunity={opportunity}
                    canManageProjects={canManageProjects}
                    dismissing={dismissingOpportunityId === opportunity.id}
                    onDismiss={() => onDismiss(opportunity.id)}
                  />
                ))}
              </div>
            )}
          </div>

          <aside className="rounded-xl border border-line bg-surface">
            <div className="border-b border-line px-4 py-3">
              <h4 className="text-base font-semibold text-ink">{copy.recentSnapshots}</h4>
            </div>
            {feedback.snapshots.length === 0 ? (
              <p className="px-4 py-5 text-sm leading-5 text-ink-muted">{copy.noSnapshots}</p>
            ) : (
              <div className="divide-y divide-line">
                {feedback.snapshots.slice(0, 8).map((snapshot) => (
                  <PerformanceSnapshotRow key={snapshot.id} copy={copy} locale={locale} snapshot={snapshot} />
                ))}
              </div>
            )}
          </aside>
        </section>
      )}
    </div>
  );
}

export function PerformanceSummaryCard({
  label,
  value,
  tone = "default",
  valueClassName,
}: {
  label: string;
  value: number | string;
  tone?: "default" | "warning";
  valueClassName?: string;
}) {
  return (
    <div className="rounded-xl border border-line bg-surface-alt p-4">
      <p className="text-xs font-medium text-ink-muted">{label}</p>
      <p className={clsx(
        "mt-2 truncate font-semibold tabular-nums text-ink",
        typeof value === "number" ? "text-2xl" : "text-body-lg",
        tone === "warning" && "text-warning",
        valueClassName,
      )}>
        {typeof value === "number" ? formatCompactNumber(value) : value}
      </p>
    </div>
  );
}

export function PerformanceOpportunityCard({
  copy,
  opportunity,
  canManageProjects,
  dismissing,
  onDismiss,
}: {
  copy: typeof PERFORMANCE_COPY.en;
  opportunity: PerformanceOpportunity;
  canManageProjects: boolean;
  dismissing: boolean;
  onDismiss: () => void;
}) {
  const clicks = metricFromOpportunity(opportunity, "clicks", metricFromOpportunity(opportunity, "current_clicks"));
  const impressions = metricFromOpportunity(opportunity, "impressions");
  const ctr = metricFromOpportunity(opportunity, "ctr");
  const position = metricFromOpportunity(opportunity, "average_position");
  const previousClicks = metricFromOpportunity(opportunity, "previous_clicks");

  return (
    <article className="px-4 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span className={clsx("inline-flex items-center rounded-lg border px-2.5 py-1 text-xs font-semibold", performanceSeverityClasses(opportunity.severity))}>
              {copy.severity[opportunity.severity as keyof typeof copy.severity] ?? opportunity.severity}
            </span>
            <span className="text-sm font-semibold text-ink">
              {performanceTypeLabel(copy, opportunity.type)}
            </span>
          </div>
          <p className="text-sm leading-5 text-ink-secondary">{opportunity.reason}</p>
          <p className="mt-1 text-sm leading-5 text-ink-muted">{opportunity.suggested_action}</p>
        </div>
        {canManageProjects && (
          <Button variant="ghost" size="sm" loading={dismissing} onClick={onDismiss}>
            {copy.dismiss}
          </Button>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {opportunity.article_title ? (
          <PerformancePill label={copy.article} value={opportunity.article_title} />
        ) : (
          <PerformancePill label={copy.article} value={copy.unmapped} />
        )}
        {clicks !== null && <PerformancePill label={copy.clicks} value={formatCompactNumber(clicks)} />}
        {previousClicks !== null && <PerformancePill label={copy.previousClicks} value={formatCompactNumber(previousClicks)} />}
        {impressions !== null && <PerformancePill label={copy.impressions} value={formatCompactNumber(impressions)} />}
        {ctr !== null && <PerformancePill label={copy.ctr} value={formatCtr(ctr)} />}
        {position !== null && <PerformancePill label={copy.position} value={formatFixedNumber(position)} />}
      </div>

      <p className="mt-3 truncate text-xs text-ink-muted" dir="ltr">
        {opportunity.url}
      </p>
    </article>
  );
}

export function PerformancePill({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1 rounded-lg border border-line bg-surface-alt px-2.5 py-1 text-xs">
      <span className="shrink-0 text-ink-muted">{label}</span>
      <span className="min-w-0 truncate font-semibold text-ink-secondary">{value}</span>
    </span>
  );
}

export function PerformanceSnapshotRow({
  copy,
  locale,
  snapshot,
}: {
  copy: typeof PERFORMANCE_COPY.en;
  locale: ReadinessLocale;
  snapshot: PerformanceSnapshot;
}) {
  return (
    <div className="px-4 py-3">
      <p className="truncate text-sm font-semibold text-ink" dir="ltr">
        {snapshot.url}
      </p>
      <p className="mt-1 text-xs text-ink-muted">
        {copy.period}: {formatShortDate(snapshot.date_from, locale)} - {formatShortDate(snapshot.date_to, locale)}
      </p>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <PerformanceMiniMetric label={copy.clicks} value={formatCompactNumber(snapshot.clicks)} />
        <PerformanceMiniMetric label={copy.impressions} value={formatCompactNumber(snapshot.impressions)} />
        <PerformanceMiniMetric label={copy.ctr} value={formatCtr(snapshot.ctr)} />
        <PerformanceMiniMetric label={copy.position} value={formatFixedNumber(snapshot.average_position)} />
      </div>
    </div>
  );
}

export function PerformanceMiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface-alt px-2.5 py-2">
      <p className="text-xs font-medium text-ink-muted">{label}</p>
      <p className="mt-1 text-sm font-semibold tabular-nums text-ink">{value}</p>
    </div>
  );
}

function projectDraft(project: Project) {
  const preset = VERTICAL_OPTIONS.find(
    (option) => option.value === project.vertical || option.en === project.vertical
  );
  return {
    name: project.name,
    domain: project.domain ?? "",
    description: project.description ?? "",
    vertical: preset?.value ?? (project.vertical ? "__custom__" : VERTICAL_OPTIONS[0].value),
    customVertical: preset ? "" : project.vertical ?? "",
  };
}

