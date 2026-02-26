"use client";

import { useEffect, useMemo, useState } from "react";
import { apiRequest } from "@/lib/api";
import { Project } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { MetricCard } from "@/components/ui/metric-card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { EmptyState, EmptyIllustration } from "@/components/ui/empty-state";
import { Button } from "@/components/ui/button";
import { SkeletonLoader } from "@/components/ui/skeleton-loader";

/* ═══════════════════════════════════════════════════════════════
   Dashboard v3 — Compact single-viewport, monochrome, professional
   Design directive: NO scrolling. NO childish colors. Premium SaaS.
   ═══════════════════════════════════════════════════════════════ */

interface PerformancePayload {
  metrics?: {
    daily_costs?: {
      total_cost_usd?: number;
      article_count?: number;
      threshold_usd?: number;
    };
  };
}

interface HealthPayload {
  status?: string;
  version?: string;
}

interface DashboardPanelProps {
  token: string;
  projects: Project[];
  onNavigate?: (page: string) => void;
}

export function DashboardPanel({ token, projects, onNavigate }: DashboardPanelProps) {
  const { t } = useI18n();
  const [performance, setPerformance] = useState<PerformancePayload | null>(null);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const [perf, healthRes] = await Promise.all([
          apiRequest<PerformancePayload>("/system/performance", { token }),
          apiRequest<HealthPayload>("/system/health", { token }),
        ]);
        if (!mounted) return;
        setPerformance(perf);
        setHealth(healthRes);
      } catch {
        if (!mounted) return;
        setPerformance(null);
        setHealth(null);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    void load();
    return () => { mounted = false; };
  }, [token]);

  const wordpressConnected = useMemo(
    () => projects.filter((p) => Boolean(p.wordpress_url)).length,
    [projects]
  );
  const pendingWordpress = projects.length - wordpressConnected;

  const daily = performance?.metrics?.daily_costs;
  const todayCost = daily?.total_cost_usd ?? 0;
  const todayArticles = daily?.article_count ?? 0;
  const threshold = daily?.threshold_usd ?? 10;
  const percent = threshold > 0 ? Math.min(100, (todayCost / threshold) * 100) : 0;

  const recentProjects = projects.slice(0, 3);
  const overflowCount = Math.max(0, projects.length - 3);

  const isHealthy = health?.status?.toLowerCase().includes("healthy") ?? true;
  const healthLabel = isHealthy ? t("dashboard.systemHealthy") : t("dashboard.systemUnhealthy");

  // Onboarding
  const hasProject = projects.length > 0;
  const hasWp = wordpressConnected > 0;
  const steps = [
    { key: "createProject" as const, done: hasProject, label: t("dashboard.onboarding.createProject") },
    { key: "connectWp" as const, done: hasWp, label: t("dashboard.onboarding.connectWp") },
    { key: "setupRulebook" as const, done: false, label: t("dashboard.onboarding.setupRulebook") },
    { key: "generateFirst" as const, done: todayArticles > 0, label: t("dashboard.onboarding.generateFirst") },
  ];
  const completedSteps = steps.filter((s) => s.done).length;
  const showOnboarding = completedSteps < 4;

  return (
    <section className="animate-fade-in space-y-4">

      {/* ── Header row ── */}
      <div className="flex items-center justify-between">
        <h2 className="text-display-lg text-ink">{t("dashboard.title")}</h2>
        <span className="rounded-full border border-border bg-surface px-3 py-1 text-body-sm text-ink-tertiary">
          {t("dashboard.apiVersion")} {health?.version ?? "v1"}
        </span>
      </div>

      {/* ── Cost warnings ── */}
      {percent >= 95 && (
        <div className="rounded-lg border-s-4 border-s-danger bg-danger/5 px-4 py-2.5 text-body-sm font-semibold text-danger" role="alert">
          ⚠ {t("dashboard.costWarning95")}
        </div>
      )}
      {percent >= 80 && percent < 95 && (
        <div className="rounded-lg border-s-4 border-s-warning bg-warning/5 px-4 py-2.5 text-body-sm font-semibold text-warning" role="alert">
          ⚠ {t("dashboard.costWarning80")}
        </div>
      )}

      {/* ── Empty state ── */}
      {projects.length === 0 && !loading && (
        <EmptyState
          illustration={<EmptyIllustration />}
          title={t("dashboard.noProjects")}
          subtitle={t("dashboard.createFirst")}
          action={
            onNavigate && (
              <Button variant="primary" size="lg" onClick={() => onNavigate("projects")}>
                {t("projects.createProject")}
              </Button>
            )
          }
        />
      )}

      {/* ── KPI Cards — clean, monochrome, compact ── */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label={t("dashboard.totalProjects")}
          value={loading ? "-" : String(projects.length)}
          loading={loading}
          onClick={() => onNavigate?.("projects")}
        />
        <MetricCard
          label={t("dashboard.articlesToday")}
          value={loading ? "-" : String(todayArticles)}
          loading={loading}
        />
        <MetricCard
          label={t("dashboard.wpConnected")}
          value={loading ? "-" : String(wordpressConnected)}
          statusDot={wordpressConnected > 0 ? "bg-success" : "bg-ink-tertiary"}
          loading={loading}
        />
        <MetricCard
          label={t("dashboard.wpPending")}
          value={loading ? "-" : String(pendingWordpress)}
          statusDot={pendingWordpress > 0 ? "bg-warning" : "bg-success"}
          loading={loading}
        />
      </div>

      {/* ── Cost + Onboarding side-by-side (fit viewport) ── */}
      <div className={showOnboarding ? "grid gap-3 lg:grid-cols-2" : ""}>

        {/* Cost card — compact */}
        <div className="elevated-card p-4">
          <div className="flex items-center justify-between">
            <p className="text-body-sm font-medium uppercase tracking-wider text-ink-secondary">
              {t("dashboard.llmCostToday")}
            </p>
            {health && (
              <span className={`inline-flex items-center gap-1.5 text-body-sm ${isHealthy ? "text-success" : "text-warning"}`}>
                <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden />
                {healthLabel}
              </span>
            )}
          </div>
          {loading ? (
            <SkeletonLoader height={28} width="100px" className="mt-1.5" />
          ) : (
            <>
              <div className="mt-1 flex items-baseline gap-2">
                <p className="text-[1.75rem] font-bold text-ink">${todayCost.toFixed(2)}</p>
                {todayCost === 0 && (
                  <span className="text-body-sm text-ink-tertiary">{t("dashboard.noUsageToday")}</span>
                )}
              </div>
              <ProgressBar
                value={Math.max(percent, 1.5)}
                className="mt-2"
                showLabel
                label={t("dashboard.ofCap", { percent: percent.toFixed(0), cap: threshold.toFixed(0) })}
              />
            </>
          )}
        </div>

        {/* Onboarding — compact inline */}
        {showOnboarding && (
          <div className="elevated-card p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-body-sm font-semibold text-ink">
                {t("dashboard.onboardingTitle")} · {completedSteps}/4
              </h3>
              <ProgressBar value={(completedSteps / 4) * 100} className="w-20" />
            </div>
            <ul className="space-y-1.5">
              {steps.map((step, index) => (
                <li key={step.key} className="flex items-center gap-2.5 text-body-sm">
                  <span className={`grid h-5 w-5 shrink-0 place-items-center rounded-full text-[10px] font-bold ${step.done
                    ? "bg-brand text-white"
                    : "border border-border text-ink-tertiary"
                    }`}>
                    {step.done ? "✓" : index + 1}
                  </span>
                  <span className={step.done ? "text-ink-tertiary line-through" : "text-ink-secondary"}>
                    {step.label}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* ── Recent Projects — compact cards ── */}
      {recentProjects.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-body-sm font-semibold text-ink">{t("dashboard.recentProjects")}</h3>
            {overflowCount > 0 && (
              <button
                type="button"
                onClick={() => onNavigate?.("projects")}
                className="text-body-sm font-medium text-brand transition-colors duration-fast hover:text-brand-hover"
              >
                {t("dashboard.moreProjects", { count: overflowCount })}
              </button>
            )}
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            {recentProjects.map((project) => (
              <article
                key={project.id}
                className="elevated-card cursor-pointer px-4 py-3 transition-all duration-base smx-card-hover"
                onClick={() => onNavigate?.("projects")}
              >
                <div className="flex items-center gap-2">
                  <p className="truncate text-body-sm font-semibold text-ink" title={project.name}>{project.name}</p>
                  {project.wordpress_url && (
                    <span className="shrink-0 rounded bg-surface-alt px-1.5 py-0.5 text-[10px] font-bold text-ink-tertiary uppercase">WP</span>
                  )}
                </div>
                <p className="mt-0.5 truncate text-body-sm text-ink-tertiary" title={project.domain || undefined}>
                  {project.domain || t("projects.noDomain")}
                </p>
              </article>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
