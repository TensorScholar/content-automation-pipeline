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
   Spec: Screen 2 — Dashboard
   4 MetricCards, cost progress bar, onboarding checklist,
   recent projects preview, cost warning banners
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

  // Onboarding checklist
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
    <section className="animate-fade-in space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-display-lg text-ink">{t("dashboard.title")}</h2>
        <span className="rounded-full border border-border bg-surface px-3 py-1 text-body-sm text-ink-tertiary">
          {t("dashboard.apiVersion")} {health?.version ?? "v1"}
        </span>
      </div>

      {/* ── Cost warning banners ── */}
      {percent >= 95 && (
        <div className="animate-slide-down rounded-sm border border-danger/30 bg-danger-subtle px-4 py-3 text-body-md font-semibold text-danger" role="alert">
          ⚠ {t("dashboard.costWarning95")}
        </div>
      )}
      {percent >= 80 && percent < 95 && (
        <div className="animate-slide-down rounded-sm border border-warning/30 bg-warning-subtle px-4 py-3 text-body-md font-semibold text-warning" role="alert">
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

      {/* ── KPI Metric Cards ── */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label={t("dashboard.totalProjects")}
          value={loading ? "-" : String(projects.length)}
          loading={loading}
          onClick={() => onNavigate?.("projects")}
          className="smx-card-hover"
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

      {/* ── Daily LLM Cost with ProgressBar ── */}
      <div className="elevated-card p-5">
        <div className="flex items-center justify-between">
          <p className="text-body-sm font-semibold uppercase tracking-wider text-ink-secondary">
            {t("dashboard.llmCostToday")}
          </p>
          {health && (
            <span className={`inline-flex items-center gap-1.5 text-body-sm ${health.status?.toLowerCase().includes("healthy") ? "text-success" : "text-warning"
              }`}>
              <span className="h-2 w-2 rounded-full bg-current" aria-hidden />
              {health.status ?? "unknown"}
            </span>
          )}
        </div>
        {loading ? (
          <SkeletonLoader height={36} width="120px" className="mt-2" />
        ) : (
          <>
            <p className="mt-2 text-[2.25rem] font-bold text-ink">${todayCost.toFixed(2)}</p>
            <ProgressBar
              value={percent}
              className="mt-3"
              showLabel
              label={t("dashboard.ofCap", { percent: percent.toFixed(0), cap: threshold.toFixed(0) })}
            />
          </>
        )}
      </div>

      {/* ── Onboarding Checklist ── */}
      {showOnboarding && (
        <div className="elevated-card p-5 animate-fade-in">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-heading-sm text-ink">
              🚀 {completedSteps}/4
            </h3>
            <ProgressBar value={(completedSteps / 4) * 100} className="w-24" />
          </div>
          <ul className="space-y-3">
            {steps.map((step) => (
              <li
                key={step.key}
                className="flex items-center gap-3 text-body-md"
              >
                <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-full text-body-sm font-bold ${step.done
                    ? "bg-success text-white"
                    : "border-2 border-border text-ink-tertiary"
                  }`}>
                  {step.done ? "✓" : ""}
                </span>
                <span className={step.done ? "text-ink-secondary line-through" : "text-ink"}>
                  {step.label}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Recent Projects ── */}
      {recentProjects.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-heading-sm text-ink">{t("dashboard.recentProjects")}</h3>
          <div className="grid gap-3 md:grid-cols-3">
            {recentProjects.map((project) => (
              <article
                key={project.id}
                className="elevated-card cursor-pointer p-4 transition-all duration-base smx-card-hover"
                onClick={() => onNavigate?.("projects")}
              >
                <div className="mb-1 flex items-center gap-2">
                  <p className="truncate text-body-md font-semibold text-ink">{project.name}</p>
                  {project.wordpress_url && (
                    <span className="shrink-0 rounded-full bg-success-subtle px-2 py-0.5 text-body-sm font-semibold text-success">
                      WP
                    </span>
                  )}
                </div>
                <p className="truncate text-body-sm text-ink-secondary">
                  {project.domain || t("projects.noDomain")}
                </p>
              </article>
            ))}
          </div>
          {overflowCount > 0 && (
            <button
              type="button"
              onClick={() => onNavigate?.("projects")}
              className="text-body-sm font-semibold text-brand transition-colors duration-fast hover:text-brand-hover"
            >
              {t("dashboard.moreProjects", { count: overflowCount })}
            </button>
          )}
        </div>
      )}
    </section>
  );
}
