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
import { formatCompactNumber } from "./project-performance-helpers";
import { Project, ProjectPerformanceFeedback, ProjectReadiness, SearchConsoleStatus, SeoIntelligenceResponse, PerformanceSnapshot, PerformanceOpportunity } from "@/types/models";

export function SearchConsoleCard({
  copy,
  locale,
  canManageProjects,
  status,
  loading,
  action,
  error,
  onConnect,
  onRefresh,
  onRefreshProperties,
  onSelectProperty,
  onSync,
  onDisconnect,
}: {
  copy: typeof SEARCH_CONSOLE_COPY.en;
  locale: ReadinessLocale;
  canManageProjects: boolean;
  status: SearchConsoleStatus | null;
  loading: boolean;
  action: string | null;
  error: string | null;
  onConnect: () => void;
  onRefresh: () => void;
  onRefreshProperties: () => void;
  onSelectProperty: (siteUrl: string) => void;
  onSync: () => void;
  onDisconnect: () => void;
}) {
  const latestRun = status?.recent_sync_runs?.[0];
  const propertyOptions = (status?.properties ?? []).map((item) => ({
    value: item.site_url,
    label: item.site_url,
  }));
  const statusTone = latestRun?.status === "failed"
    ? "border-danger/20 bg-danger/10 text-danger"
    : latestRun?.status === "succeeded"
      ? "border-success/20 bg-success/10 text-success"
      : "border-info/20 bg-info/10 text-info";
  return (
    <section className="smx-panel-subtle p-5" aria-labelledby="search-console-title">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 id="search-console-title" className="text-lg font-semibold text-ink">{copy.title}</h3>
            <span className="rounded-md border border-brand/20 bg-brand/10 px-2 py-0.5 text-xs font-semibold text-brand">{copy.readOnly}</span>
            <span className={clsx(
              "rounded-md border px-2 py-0.5 text-xs font-semibold",
              status?.connected
                ? "border-success/20 bg-success/10 text-success"
                : "border-line bg-ink/[0.03] text-ink-muted",
            )}>
              {status?.connected ? copy.connected : copy.disconnected}
            </span>
          </div>
          <p className="mt-2 max-w-2xl text-xs leading-5 text-ink-muted">{copy.subtitle}</p>
        </div>
        <Button variant="outlined" size="sm" loading={loading} onClick={onRefresh}>{PERFORMANCE_COPY[locale].refresh}</Button>
      </div>

      {loading && !status && <p className="mt-4 text-sm text-ink-muted">{copy.loading}</p>}
      {error && <div className="mt-4 rounded-lg border border-danger/20 bg-danger/10 px-3 py-2 text-xs text-danger" role="alert">{error}</div>}
      {status && !status.configured && <div className="mt-4 rounded-lg border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">{copy.notConfigured}</div>}

      {status?.configured && !status.connected && canManageProjects && (
        <Button className="mt-4" variant="primary" size="sm" loading={action === "connect"} onClick={onConnect}>{copy.connect}</Button>
      )}

      {status?.connected && (
        <div className="mt-4 space-y-4">
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
            <SelectDropdown
              label={copy.property}
              options={propertyOptions}
              value={status.selected_site_url ?? undefined}
              placeholder={propertyOptions.length ? copy.selectProperty : copy.noProperties}
              disabled={!canManageProjects || Boolean(action) || propertyOptions.length === 0}
              onChange={onSelectProperty}
            />
            <div className="flex flex-wrap gap-2">
              {canManageProjects && (
                <>
                  <Button variant="outlined" size="sm" loading={action === "properties"} disabled={Boolean(action) && action !== "properties"} onClick={onRefreshProperties}>{copy.refreshProperties}</Button>
                  <Button variant="primary" size="sm" loading={action === "sync"} disabled={!status.selected_site_url || (Boolean(action) && action !== "sync")} onClick={onSync}>{copy.syncNow}</Button>
                </>
              )}
            </div>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-line bg-surface-alt px-3 py-3">
            <div className="min-w-0 text-xs text-ink-muted">
              <span className="font-semibold text-ink-secondary">{copy.lastSync}: </span>
              {status.last_sync_at ? formatReadinessDate(status.last_sync_at, locale) : copy.never}
              {latestRun && (
                <span className={clsx("ms-2 inline-flex rounded-md border px-2 py-0.5 font-semibold", statusTone)}>
                  {latestRun.status}
                  {latestRun.status === "succeeded" ? ` · ${formatCompactNumber(latestRun.row_count)}` : ""}
                </span>
              )}
            </div>
            {canManageProjects && (
              <div className="flex gap-2">
                <Button variant="ghost" size="sm" loading={action === "connect"} onClick={onConnect}>{copy.reconnect}</Button>
                <Button variant="ghost" size="sm" loading={action === "disconnect"} disabled={Boolean(action) && action !== "disconnect"} onClick={onDisconnect}>{copy.disconnect}</Button>
              </div>
            )}
          </div>
          {(status.last_error_message || latestRun?.error_message) && (
            <div className="rounded-lg border border-danger/20 bg-danger/10 px-3 py-2 text-xs text-danger" role="alert">
              {status.last_error_message || latestRun?.error_message}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

