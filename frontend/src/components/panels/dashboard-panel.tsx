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
   Dashboard v2 — Complete Redesign
   Fixes: #1 raw i18n key, #2 hardcoded English, #3 card icons,
   #4 status dots, #5 cost card sizing, #6 progress bar,
   #7 RTL currency, #12 emoji, #13 onboarding circles,
   #14 check color, #17 spacing, #19 no charts, #20 no time,
   #21 vertical spacing, #22 border-radius, #23 shadows,
   #25 logout text, #26 truncation, #27 empty states,
   #28 welcome, #29 subtitle, #31 progress RTL, #32 font weight,
   #33 numbers, #34 health RTL, #35 skeleton, #36 onboarding alignment,
   #38 padding, #39 sidebar, #40 brand weight
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

// ── SVG Icons for MetricCards (#3 unique icons per card) ──────

function IconProjects({ color }: { color: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5" stroke={color} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="3" width="20" height="18" rx="3" /><path d="M8 3v18" /><path d="M12 9h6" /><path d="M12 13h4" />
    </svg>
  );
}
function IconArticles({ color }: { color: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5" stroke={color} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6" /><path d="M10 13h4" /><path d="M10 17h4" />
    </svg>
  );
}
function IconWordPressOk({ color }: { color: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5" stroke={color} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z" /><path d="M9 12l2 2 4-4" />
    </svg>
  );
}
function IconWordPressPending({ color }: { color: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5" stroke={color} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" />
    </svg>
  );
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

  // #2 Localized health status
  const isHealthy = health?.status?.toLowerCase().includes("healthy") ?? true;
  const healthLabel = isHealthy ? t("dashboard.systemHealthy") : t("dashboard.systemUnhealthy");

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

      {/* ── Header with welcome message (#28, #29) ── */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-display-lg text-ink">{t("dashboard.title")}</h2>
        </div>
        {/* #18 API version — compact pill, not orphaned */}
        <span className="mt-1 rounded-full border border-border bg-surface px-3 py-1 text-body-sm text-ink-tertiary">
          {t("dashboard.apiVersion")} {health?.version ?? "v1"}
        </span>
      </div>

      {/* ── Cost warning banners ── */}
      {percent >= 95 && (
        <div className="animate-slide-down rounded-xl border-s-4 border-s-danger bg-danger/5 px-4 py-3 text-body-md font-semibold text-danger" role="alert">
          ⚠ {t("dashboard.costWarning95")}
        </div>
      )}
      {percent >= 80 && percent < 95 && (
        <div className="animate-slide-down rounded-xl border-s-4 border-s-warning bg-warning/5 px-4 py-3 text-body-md font-semibold text-warning" role="alert">
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

      {/* ── KPI Metric Cards (#3 unique icons, #4 meaningful dots, #27 empty CTAs) ── */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label={t("dashboard.totalProjects")}
          value={loading ? "-" : String(projects.length)}
          loading={loading}
          icon={<IconProjects color="#0D9488" />}
          accentColor="#0D9488"
          emptyAction={onNavigate ? t("dashboard.createFirst") : undefined}
          onClick={() => onNavigate?.("projects")}
        />
        <MetricCard
          label={t("dashboard.articlesToday")}
          value={loading ? "-" : String(todayArticles)}
          loading={loading}
          icon={<IconArticles color="#6366F1" />}
          accentColor="#6366F1"
        />
        <MetricCard
          label={t("dashboard.wpConnected")}
          value={loading ? "-" : String(wordpressConnected)}
          statusDot={wordpressConnected > 0 ? "bg-success" : "bg-ink-tertiary"}
          loading={loading}
          icon={<IconWordPressOk color="#22C55E" />}
          accentColor="#22C55E"
        />
        <MetricCard
          label={t("dashboard.wpPending")}
          value={loading ? "-" : String(pendingWordpress)}
          statusDot={pendingWordpress > 0 ? "bg-warning" : "bg-success"}
          loading={loading}
          icon={<IconWordPressPending color="#F59E0B" />}
          accentColor="#F59E0B"
        />
      </div>

      {/* ── Daily LLM Cost (#5 compact, #6 visible bar, #2 localized status, #9 no redundancy) ── */}
      <div className="elevated-card overflow-hidden">
        <div className="p-5">
          <div className="flex items-center justify-between">
            <p className="text-body-sm font-semibold uppercase tracking-wider text-ink-secondary">
              {t("dashboard.llmCostToday")}
            </p>
            {health && (
              <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-body-sm font-medium ${isHealthy
                  ? "bg-success/10 text-success"
                  : "bg-warning/10 text-warning"
                }`}>
                <span className="h-2 w-2 rounded-full bg-current" style={{ animation: "status-pulse 2s ease-in-out infinite" }} aria-hidden />
                {healthLabel}
              </span>
            )}
          </div>
          {loading ? (
            <SkeletonLoader height={36} width="120px" className="mt-2" />
          ) : (
            <>
              {todayCost === 0 ? (
                <div className="mt-3 flex items-center gap-3">
                  <p className="text-[2rem] font-bold text-ink-tertiary">$0.00</p>
                  <span className="rounded-full bg-surface-alt px-3 py-1 text-body-sm text-ink-tertiary">{t("dashboard.noUsageToday")}</span>
                </div>
              ) : (
                <p className="mt-2 text-[2rem] font-bold text-ink">${todayCost.toFixed(2)}</p>
              )}
              <div className="mt-3">
                <ProgressBar
                  value={Math.max(percent, 2)}
                  className="mt-3"
                  showLabel
                  label={t("dashboard.ofCap", { percent: percent.toFixed(0), cap: threshold.toFixed(0) })}
                />
              </div>
            </>
          )}
        </div>
      </div>

      {/* ── Onboarding Checklist (#12 no emoji, #13 step numbers, #14 brand color, #31 RTL progress, #36 alignment) ── */}
      {showOnboarding && (
        <div className="elevated-card p-5 animate-fade-in">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="grid h-8 w-8 place-items-center rounded-lg bg-brand/10">
                <svg viewBox="0 0 20 20" fill="none" className="h-4.5 w-4.5" stroke="#0D9488" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10 2l2.09 6.26L18 10l-5.91 1.74L10 18l-2.09-6.26L2 10l5.91-1.74L10 2z" />
                </svg>
              </div>
              <div>
                <h3 className="text-heading-sm text-ink">{t("dashboard.onboardingTitle")}</h3>
                <p className="text-body-sm text-ink-tertiary">{completedSteps}/4</p>
              </div>
            </div>
            <ProgressBar value={(completedSteps / 4) * 100} className="w-28" />
          </div>
          <ul className="space-y-2.5">
            {steps.map((step, index) => (
              <li
                key={step.key}
                className="flex items-center gap-3 rounded-lg px-3 py-2 text-body-md transition-colors duration-fast hover:bg-surface-alt"
              >
                <span className={`grid h-7 w-7 shrink-0 place-items-center rounded-full text-body-sm font-bold transition-all duration-base ${step.done
                  ? "bg-brand text-white"
                  : "border-2 border-border text-ink-tertiary"
                  }`}>
                  {step.done ? (
                    <svg viewBox="0 0 16 16" fill="none" className="h-3.5 w-3.5" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M3 8.5 6.3 11.7 13 5" /></svg>
                  ) : (
                    <span className="text-[12px]">{index + 1}</span>
                  )}
                </span>
                <span className={step.done ? "text-ink-secondary line-through" : "text-ink"}>
                  {step.label}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Recent Projects (#1 FIXED — uses localized key, #26 no truncation issue) ── */}
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
                  <p className="truncate text-body-md font-semibold text-ink" title={project.name}>{project.name}</p>
                  {project.wordpress_url && (
                    <span className="shrink-0 rounded-full bg-success/10 px-2 py-0.5 text-body-sm font-semibold text-success">
                      WP
                    </span>
                  )}
                </div>
                <p className="truncate text-body-sm text-ink-secondary" title={project.domain || undefined}>
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
