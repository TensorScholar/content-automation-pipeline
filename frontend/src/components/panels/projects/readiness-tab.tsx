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
import { Project, ProjectPerformanceFeedback, ProjectReadiness, SearchConsoleStatus, SeoIntelligenceResponse, PerformanceSnapshot, PerformanceOpportunity } from "@/types/models";

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

export function ReadinessTab({
  copy,
  locale,
  readiness,
  loading,
  error,
  onRefresh,
  onOpenRulebook,
  onOpenWordPress,
}: {
  copy: typeof READINESS_COPY.en;
  locale: ReadinessLocale;
  readiness: ProjectReadiness | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
  onOpenRulebook: () => void;
  onOpenWordPress: () => void;
}) {
  const wordpressOnlyBlocking = !!readiness
    && readiness.blocking_items.length > 0
    && readiness.blocking_items.every((item) => readinessItemKind(item.id, item.label) === "wordpressPublishing");
  const displayStatus = wordpressOnlyBlocking ? "warning" : readiness?.status;
  const canGenerateForDisplay = !!readiness && (readiness.can_generate || wordpressOnlyBlocking);
  const generationBlocker = readiness?.blocking_items.find(
    (item) => readinessItemKind(item.id, item.label) !== "wordpressPublishing"
  );
  const publishingBlocker = readiness?.blocking_items.find(
    (item) => readinessItemKind(item.id, item.label) === "wordpressPublishing"
  );
  const statusLabel =
    displayStatus === "ready"
      ? copy.ready
      : displayStatus === "blocked"
        ? copy.blocked
        : copy.warning;

  return (
    <div className="max-w-4xl space-y-4 animate-fade-in">
      <section className="smx-panel-subtle p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-xs font-medium text-ink-muted">{copy.subtitle}</p>
            <h3 className="mt-1 text-xl font-semibold tracking-tight text-ink">
              {copy.title}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            {readiness && (
              <span className={clsx("inline-flex h-8 items-center gap-2 rounded-lg border px-3 text-xs font-semibold", readinessStatusClasses(displayStatus ?? "warning"))}>
                <span className={clsx("h-2 w-2 rounded-full", readinessDotClasses(displayStatus ?? "warning"))} aria-hidden />
                {statusLabel}
              </span>
            )}
            <Button variant="outlined" size="sm" loading={loading} onClick={onRefresh}>
              {copy.refresh}
            </Button>
          </div>
        </div>

        {loading && !readiness && (
          <div className="mt-5 rounded-lg border border-line bg-surface-alt px-4 py-3 text-sm font-medium text-ink-secondary">
            {copy.loading}
          </div>
        )}

        {error && (
          <div className="mt-5 rounded-lg border border-danger/20 bg-danger/10 px-4 py-3 text-sm font-medium text-danger" role="alert">
            {copy.failed} {localizeProjectError(error, locale)}
          </div>
        )}

        {readiness && (
          <>
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <div
                className={clsx(
                  "rounded-xl border p-4",
                  canGenerateForDisplay
                    ? "border-success/20 bg-success/[0.06]"
                    : "border-danger/20 bg-danger/[0.06]"
                )}
              >
                <p className="text-xs font-medium text-ink-muted">{copy.canGenerate}</p>
                <p className="mt-2 text-lg font-semibold text-ink">
                  {canGenerateForDisplay ? copy.available : copy.unavailable}
                </p>
                {!canGenerateForDisplay && generationBlocker && (
                  <p className="mt-2 text-xs leading-5 text-ink-muted">
                    {localizeReadinessText(generationBlocker.message, locale)}
                  </p>
                )}
              </div>
              <div
                className={clsx(
                  "rounded-xl border p-4",
                  readiness.can_publish
                    ? "border-success/20 bg-success/[0.06]"
                    : "border-warning/20 bg-warning/[0.06]"
                )}
              >
                <p className="text-xs font-medium text-ink-muted">{copy.canPublish}</p>
                <p className="mt-2 text-lg font-semibold text-ink">
                  {readiness.can_publish ? copy.available : copy.unavailable}
                </p>
                {!readiness.can_publish && publishingBlocker && (
                  <p className="mt-2 text-xs leading-5 text-ink-muted">
                    {localizeReadinessText(publishingBlocker.message, locale)}
                  </p>
                )}
              </div>
            </div>
            <p className="mt-4 text-xs text-ink-muted">
              {copy.lastChecked}: {formatReadinessDate(readiness.last_checked_at, locale)}
            </p>
          </>
        )}
      </section>

      {readiness && (
        <section className="grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
          <div className="min-w-0 rounded-xl border border-line bg-surface">
            <div className="border-b border-line px-4 py-3">
              <h4 className="text-base font-semibold text-ink">{copy.allChecks}</h4>
            </div>
            <div className="divide-y divide-line">
              {readiness.checks.map((check) => (
                <div key={check.id} className="grid gap-3 px-4 py-3 sm:grid-cols-[160px_minmax(0,1fr)]">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className={clsx("h-2 w-2 shrink-0 rounded-full", readinessDotClasses(check.status))} aria-hidden />
                    <span className="truncate text-sm font-semibold text-ink">
                      {localizeReadinessLabel(check.id, check.label, locale)}
                    </span>
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm leading-5 text-ink-secondary">
                      {localizeReadinessText(check.message, locale)}
                    </p>
                    {check.remediation && (
                      <p className="mt-1 text-xs leading-5 text-ink-muted">
                        {localizeReadinessText(check.remediation, locale)}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <aside className="min-w-0 space-y-4">
            <div className="rounded-xl border border-line bg-surface p-4">
              <h4 className="text-sm font-semibold text-ink">{copy.blockers}</h4>
              <p className="mt-2 text-metric font-semibold tabular-nums text-ink">
                {readiness.blocking_items.length}
              </p>
            </div>
            <div className="rounded-xl border border-line bg-surface p-4">
              <h4 className="text-sm font-semibold text-ink">{copy.warnings}</h4>
              <p className="mt-2 text-metric font-semibold tabular-nums text-ink">
                {readiness.warnings.length}
              </p>
            </div>
            <div className="rounded-xl border border-line bg-surface p-4">
              <h4 className="text-sm font-semibold text-ink">{copy.actions}</h4>
              <div className="mt-3 space-y-2">
                {readiness.manager_actions.length === 0 ? (
                  <p className="text-xs text-ink-muted">{copy.noActions}</p>
                ) : (
                  readiness.manager_actions.map((action) => (
                    <Button
                      key={action.id}
                      variant="outlined"
                      size="sm"
                      fullWidth
                      onClick={() => {
                        if (action.id === "open_rulebook") onOpenRulebook();
                        if (action.id === "test_wordpress_connection") onOpenWordPress();
                      }}
                    >
                      {action.id === "open_rulebook"
                        ? copy.openRulebook
                        : action.id === "test_wordpress_connection"
                          ? copy.openWordPress
                          : action.label}
                    </Button>
                  ))
                )}
              </div>
            </div>
          </aside>
        </section>
      )}
    </div>
  );
}

export function performanceSeverityClasses(severity: string) {
  if (severity === "high") {
    return "border-danger/20 bg-danger/10 text-danger";
  }
  if (severity === "medium") {
    return "border-warning/20 bg-warning/10 text-warning";
  }
  return "border-info/20 bg-info-subtle text-info";
}

export function performanceTypeLabel(copy: typeof PERFORMANCE_COPY.en, type: string) {
  const key = type as keyof typeof PERFORMANCE_COPY.en.types;
  return copy.types[key] ?? type.replaceAll("_", " ");
}

export function formatCompactNumber(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function formatFixedNumber(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

export function formatCtr(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${formatFixedNumber(value * 100, 2)}%`;
}

export function formatShortDate(value: string | null | undefined, locale: ReadinessLocale) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  return date.toLocaleDateString(localeName, { month: "short", day: "numeric", year: "numeric" });
}

export function metricFromOpportunity(
  opportunity: PerformanceOpportunity,
  key: string,
  fallback?: number | string | null,
) {
  const value = opportunity.supporting_metrics?.[key] ?? fallback;
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
  }
  return null;
}

