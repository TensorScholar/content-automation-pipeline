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

function formatCompactNumber(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
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

export function SeoIntelligenceCard({
  payload,
  copy,
  locale,
  loading,
  error,
}: {
  payload: SeoIntelligenceResponse | null;
  copy: typeof SEO_INTELLIGENCE_COPY.en;
  locale: ReadinessLocale;
  loading: boolean;
  error: string | null;
}) {
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  if (loading && !payload) {
    return <section className="smx-panel-subtle animate-pulse px-5 py-6 text-sm text-ink-muted">{copy.loading}</section>;
  }
  if (error && !payload) {
    return <section className="rounded-xl border border-warning/20 bg-warning/10 px-5 py-4 text-sm text-warning" role="alert">{copy.failed} {localizeProjectError(error, locale)}</section>;
  }
  if (!payload) return null;

  const coverage = Math.round(payload.portfolio.coverage_ratio * 100);
  const dataStatusLabel = payload.data_quality.status === "good"
    ? copy.dataGood
    : payload.data_quality.status === "insufficient"
      ? copy.dataInsufficient
      : copy.dataLimited;
  const nextActionCopy = SEO_NEXT_ACTION_COPY[locale] ?? SEO_NEXT_ACTION_COPY.en;
  return (
    <section className="smx-panel-subtle overflow-hidden" aria-live="polite">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-line px-5 py-4">
        <div>
          <p className="text-xs font-medium text-ink-muted">{copy.subtitle}</p>
          <h3 className="mt-1 text-xl font-semibold tracking-tight text-ink">{copy.title}</h3>
          <p className="mt-1 text-xs text-ink-muted">{copy.safe}</p>
        </div>
        <span className={clsx("rounded-full border px-2.5 py-1 text-xs font-semibold", payload.data_quality.status === "good" ? "border-success/20 bg-success/10 text-success" : payload.data_quality.status === "insufficient" ? "border-danger/20 bg-danger-subtle text-danger" : "border-warning/20 bg-warning/10 text-warning")}>
          {copy.dataQuality}: {dataStatusLabel}
        </span>
      </div>
      <div className="grid gap-3 p-5 sm:grid-cols-3">
        <PerformanceSummaryCard label={copy.health} value={`${payload.portfolio.health_score}/100`} tone={payload.portfolio.health_score < 55 ? "warning" : "default"} />
        <PerformanceSummaryCard label={copy.coverage} value={`${new Intl.NumberFormat(localeName).format(coverage)}%`} />
        <PerformanceSummaryCard label={copy.highPriority} value={payload.portfolio.high_priority_count} tone={payload.portfolio.high_priority_count > 0 ? "warning" : "default"} />
      </div>
      <div className="border-t border-line px-5 py-4">
        <h4 className="text-sm font-semibold text-ink">{copy.queue}</h4>
        {payload.recommended_queue.length === 0 ? (
          <p className="mt-3 text-sm text-ink-muted">{copy.noQueue}</p>
        ) : (
          <div className="mt-3 space-y-2">
            {payload.recommended_queue.slice(0, 5).map((item) => (
              <article key={item.opportunity_id} className="flex items-start gap-3 rounded-xl border border-line bg-surface px-3 py-3">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-brand/10 text-xs font-bold text-brand">{item.rank}</span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="truncate text-sm font-semibold text-ink">{item.article_title || item.url}</p>
                    <span className="rounded-md bg-ink/[0.04] px-2 py-0.5 text-xs font-semibold text-ink-secondary">{item.priority_score}/100</span>
                  </div>
                  <p className="mt-1 text-xs leading-5 text-ink-muted">
                    {nextActionCopy[item.type as keyof typeof SEO_NEXT_ACTION_COPY.en] ?? item.next_action?.title ?? item.type}
                  </p>
                  <p className="mt-1 text-xs text-ink-muted">{Math.round(item.confidence * 100)}% {copy.confidence}</p>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
      {payload.data_quality.warnings.length > 0 ? (
        <details className="border-t border-line px-5 py-3 text-xs text-ink-secondary">
          <summary className="cursor-pointer font-medium">{copy.dataQuality} ({payload.data_quality.warnings.length})</summary>
          <ul className="mt-2 space-y-1.5">
            {payload.data_quality.warnings.map((warning) => (
              <li key={warning.code}>
                • {SEO_WARNING_COPY[locale]?.[warning.code as keyof typeof SEO_WARNING_COPY.en] ?? warning.message}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}

