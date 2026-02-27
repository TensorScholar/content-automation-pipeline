"use client";

import { useEffect, useState } from "react";
import { apiRequest } from "@/lib/api";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { MetricCard } from "@/components/ui/metric-card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { StatusBadge } from "@/components/ui/status-badge";
import { SkeletonLoader } from "@/components/ui/skeleton-loader";
import { clsx } from "clsx";

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
    <section className="animate-fade-in relative flex flex-col space-y-6 bg-[#F5F5F7] min-h-[calc(100vh-80px)] p-4 md:p-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-2">
        <div>
          <h2 className="text-[28px] font-bold text-slate-900 tracking-tight">{t("monitoring.title") || "System Monitoring"}</h2>
          <p className="text-[14px] text-slate-500 mt-1">Review the live health status of backend microservices.</p>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          {lastCheckTime && (
            <span className="text-[13px] font-medium text-slate-500 bg-white px-3 py-1.5 rounded-full border border-slate-200">
              {t("monitoring.lastCheck") || "Last Check"}: {lastCheckTime}
            </span>
          )}
          <button
            type="button"
            onClick={() => void load()}
            className="w-10 h-10 flex items-center justify-center rounded-full bg-white text-slate-700 shadow-sm border border-slate-200 hover:bg-slate-50 transition-colors"
            title={t("common.refresh")}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
          </button>
        </div>
      </div>

      {/* ── Health Dependency Grid ── */}
      <div className="grid gap-4 grid-cols-2 md:grid-cols-4">
        {depKeys.map(({ key, label }) => {
          const dep = deps[key];
          const statusStr = key === "api" ? (health?.status ?? "unknown") : (dep?.status ?? "unknown");
          const isHealthy = statusStr.toLowerCase().includes("healthy") || statusStr.toLowerCase() === "ok" || statusStr.toLowerCase() === "connected";
          return (
            <div key={key} className={clsx(
              "relative overflow-hidden rounded-2xl border p-5 transition-all duration-300",
              "bg-white backdrop-blur-xl shadow-sm hover:shadow-md",
              isHealthy ? "border-emerald-100" : "border-red-100"
            )}>
              {loading ? (
                <div className="space-y-3">
                  <SkeletonLoader height={14} width={80} />
                  <SkeletonLoader height={24} width={64} />
                </div>
              ) : (
                <div className="flex flex-col h-full justify-between gap-3">
                  <div className="flex flex-col gap-1">
                    <span className="text-[11px] font-bold uppercase tracking-widest text-slate-400">
                      {label}
                    </span>
                    <span className={clsx(
                      "text-xl font-semibold tracking-tight",
                      isHealthy ? "text-emerald-700" : "text-red-600"
                    )}>
                      {statusStr.toLowerCase() === "unknown"
                        ? (t("monitoring.statusUnknown") || "Unknown")
                        : statusStr.toLowerCase().includes("healthy")
                          ? (t("monitoring.healthy") || "Healthy")
                          : statusStr}
                    </span>
                  </div>

                  <div className="absolute top-4 end-4">
                    <span className={clsx(
                      "flex h-2.5 w-2.5 rounded-full",
                      isHealthy ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]" : "bg-red-500 animate-pulse-soft shadow-[0_0_8px_rgba(239,68,68,0.5)]"
                    )} aria-hidden />
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Performance Metrics ── */}
      <div className="grid gap-4 md:grid-cols-3">
        {/* Daily Cost */}
        <div className="rounded-2xl border border-slate-200/60 bg-white p-5 shadow-sm">
          <span className="text-[11px] font-bold uppercase tracking-widest text-slate-400 block mb-1">
            {t("monitoring.dailyCost")}
          </span>
          {loading ? (
            <SkeletonLoader height={32} width={120} className="mb-4" />
          ) : (
            <div className="text-3xl font-semibold text-slate-900 mb-4 tracking-tight">
              ${todayCost.toFixed(2)}
            </div>
          )}
          <ProgressBar
            value={costPercent}
            className="mt-2"
            showLabel
            label={t("dashboard.ofCap", { percent: costPercent.toFixed(0), cap: threshold.toFixed(0) })}
          />
        </div>

        {/* Articles Today */}
        <div className="rounded-2xl border border-slate-200/60 bg-white p-5 shadow-sm">
          <span className="text-[11px] font-bold uppercase tracking-widest text-slate-400 block mb-1">
            {t("monitoring.articlesToday")}
          </span>
          {loading ? (
            <SkeletonLoader height={32} width={80} className="mb-4" />
          ) : (
            <div className="text-3xl font-semibold text-slate-900 mb-4 tracking-tight">
              {String(todayArticles)}
            </div>
          )}
          <p className="mt-2 text-[13px] font-medium text-slate-500">
            {t("monitoring.avgCost")}: <span className="text-slate-700">${avgCostPerArticle.toFixed(3)}</span>
          </p>
        </div>

        {/* Connection Pool */}
        <div className="rounded-2xl border border-slate-200/60 bg-white p-5 shadow-sm">
          <span className="text-[11px] font-bold uppercase tracking-widest text-slate-400 block mb-1">
            {t("monitoring.connectionPool")}
          </span>
          {loading ? (
            <SkeletonLoader height={32} width={100} className="mb-4" />
          ) : (
            <div className="text-3xl font-semibold text-slate-900 mb-4 tracking-tight">
              {`${poolUsed}/${poolSize}`}
            </div>
          )}
          <ProgressBar
            value={poolPercent}
            className="mt-2"
            showLabel
            label={t("monitoring.utilized", { percent: poolPercent.toFixed(0) })}
          />
        </div>
      </div>

      {/* ── Grafana Dashboard ── */}
      <article className="rounded-3xl border border-slate-200/60 bg-white shadow-sm overflow-hidden mt-6">
        <div className="border-b border-slate-100 px-6 py-5 bg-slate-50/50">
          <h3 className="text-[15px] font-bold text-slate-900">{t("monitoring.grafana")}</h3>
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
