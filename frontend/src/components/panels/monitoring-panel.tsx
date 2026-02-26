"use client";

import { useEffect, useState } from "react";
import { apiRequest } from "@/lib/api";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { MetricCard } from "@/components/ui/metric-card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { StatusBadge } from "@/components/ui/status-badge";
import { SkeletonLoader } from "@/components/ui/skeleton-loader";

/* ═══════════════════════════════════════════════════════════════
   Spec: Screen 8 — Monitoring (Manager-Only)
   Health dependency grid, performance metrics, Grafana iframe
   ═══════════════════════════════════════════════════════════════ */

interface MonitoringPanelProps { token: string; }

interface HealthPayload {
  status?: string;
  version?: string;
  dependencies?: Record<string, { status?: string }>;
}

interface PerformancePayload {
  metrics?: {
    daily_costs?: { total_cost_usd?: number; article_count?: number; threshold_usd?: number };
    db_pool?: { pool_size?: number; checked_out?: number };
  };
}

const GRAFANA_URL = process.env.NEXT_PUBLIC_GRAFANA_URL ?? "";

export function MonitoringPanel({ token }: MonitoringPanelProps) {
  const { t } = useI18n();
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [performance, setPerformance] = useState<PerformancePayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastCheckTime, setLastCheckTime] = useState<string | null>(null);

  const load = async () => {
    try {
      const [h, p] = await Promise.all([
        apiRequest<HealthPayload>("/system/health", { token }),
        apiRequest<PerformancePayload>("/system/performance", { token }),
      ]);
      setHealth(h);
      setPerformance(p);
      setLastCheckTime(new Date().toLocaleTimeString());
    } catch { setHealth(null); setPerformance(null); }
    finally { setLoading(false); }
  };

  useEffect(() => { void load(); }, [token]); // eslint-disable-line react-hooks/exhaustive-deps

  const deps = health?.dependencies ?? {};
  const depKeys: Array<{ key: string; label: string }> = [
    { key: "api", label: t("monitoring.healthApi") },
    { key: "database", label: t("monitoring.healthDb") },
    { key: "redis", label: t("monitoring.healthRedis") },
    { key: "celery", label: t("monitoring.healthCelery") },
  ];

  const daily = performance?.metrics?.daily_costs;
  const todayCost = daily?.total_cost_usd ?? 0;
  const todayArticles = daily?.article_count ?? 0;
  const threshold = daily?.threshold_usd ?? 10;
  const costPercent = threshold > 0 ? Math.min(100, (todayCost / threshold) * 100) : 0;
  const avgCostPerArticle = todayArticles > 0 ? todayCost / todayArticles : 0;

  const pool = performance?.metrics?.db_pool;
  const poolSize = pool?.pool_size ?? 0;
  const poolUsed = pool?.checked_out ?? 0;
  const poolPercent = poolSize > 0 ? Math.min(100, (poolUsed / poolSize) * 100) : 0;

  return (
    <section className="animate-fade-in space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-display-lg text-ink">{t("monitoring.title")}</h2>
        <div className="flex items-center gap-3">
          {lastCheckTime && (
            <span className="text-body-sm text-ink-secondary">
              {t("monitoring.lastCheck")}: {lastCheckTime}
            </span>
          )}
          <Button variant="outlined" onClick={() => void load()}>
            {t("common.refresh")}
          </Button>
        </div>
      </div>

      {/* ── Health Dependency Grid ── */}
      <div className="grid gap-3 grid-cols-2 md:grid-cols-4">
        {depKeys.map(({ key, label }) => {
          const dep = deps[key];
          const statusStr = key === "api" ? (health?.status ?? "unknown") : (dep?.status ?? "unknown");
          const isHealthy = statusStr.toLowerCase().includes("healthy") || statusStr.toLowerCase() === "ok" || statusStr.toLowerCase() === "connected";
          return (
            <div key={key} className={`elevated-card border-s-2 p-4 ${isHealthy ? "border-s-success" : "border-s-danger"}`}>
              {loading ? (
                <div className="space-y-2">
                  <SkeletonLoader height={12} width={80} />
                  <SkeletonLoader height={20} width={64} />
                </div>
              ) : (
                <>
                  <div className="mb-1 flex items-center gap-2">
                    <span className={`h-2 w-2 rounded-full ${isHealthy ? "bg-success" : "bg-danger animate-pulse-soft"}`} aria-hidden />
                    <span className="text-body-sm font-semibold uppercase tracking-wider text-ink-secondary">{label}</span>
                  </div>
                  <StatusBadge variant={isHealthy ? "success" : "error"} dot={false}>
                    {statusStr}
                  </StatusBadge>
                </>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Performance Metrics ── */}
      <div className="grid gap-4 md:grid-cols-3">
        {/* Daily Cost */}
        <MetricCard
          label={t("monitoring.dailyCost")}
          value={`$${todayCost.toFixed(2)}`}
          loading={loading}
        >
          <ProgressBar
            value={costPercent}
            className="mt-3"
            showLabel
            label={t("dashboard.ofCap", { percent: costPercent.toFixed(0), cap: threshold.toFixed(0) })}
          />
        </MetricCard>

        {/* Articles Today */}
        <MetricCard
          label={t("monitoring.articlesToday")}
          value={String(todayArticles)}
          loading={loading}
        >
          <p className="mt-3 text-body-sm text-ink-secondary">
            {t("monitoring.avgCost")}: ${avgCostPerArticle.toFixed(3)}
          </p>
        </MetricCard>

        {/* Connection Pool */}
        <MetricCard
          label={t("monitoring.connectionPool")}
          value={`${poolUsed}/${poolSize}`}
          loading={loading}
        >
          <ProgressBar
            value={poolPercent}
            className="mt-3"
            showLabel
            label={t("monitoring.utilized", { percent: poolPercent.toFixed(0) })}
          />
        </MetricCard>
      </div>

      {/* ── Grafana Dashboard ── */}
      <article className="elevated-card overflow-hidden">
        <div className="border-b border-border px-5 py-4">
          <h3 className="text-heading-sm text-ink">{t("monitoring.grafana")}</h3>
        </div>
        {GRAFANA_URL ? (
          <iframe
            src={GRAFANA_URL}
            title="Grafana Dashboard"
            className="w-full border-0"
            style={{ height: "560px" }}
            loading="lazy"
          />
        ) : (
          <div className="px-5 py-8 text-center">
            <p className="text-body-md text-ink-secondary">{t("monitoring.grafanaSetup")}</p>
          </div>
        )}
      </article>
    </section>
  );
}
