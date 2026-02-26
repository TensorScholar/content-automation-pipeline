"use client";

import { useEffect, useMemo, useState } from "react";
import { apiRequest } from "@/lib/api";
import { Project } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { MetricCard } from "@/components/ui/metric-card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Button } from "@/components/ui/button";
import { SkeletonLoader } from "@/components/ui/skeleton-loader";

/* ═══════════════════════════════════════════════════════════════
   Dashboard v5 — Smart Empty State Architecture
   Rule: 0 projects → hide metrics, show centered onboarding
         >0 projects → show metric grid + cost + recent
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

/* ── Workspace illustration for empty state ── */
function WorkspaceIllustration() {
  return (
    <div className="mx-auto mb-6 flex h-24 w-24 items-center justify-center rounded-3xl bg-gradient-to-br from-teal-50 to-teal-100/60">
      <svg viewBox="0 0 48 48" fill="none" className="h-12 w-12 text-teal-600">
        <rect x="6" y="8" width="36" height="32" rx="4" stroke="currentColor" strokeWidth="2" />
        <path d="M6 16h36" stroke="currentColor" strokeWidth="2" />
        <circle cx="12" cy="12" r="1.5" fill="currentColor" opacity="0.4" />
        <circle cx="17" cy="12" r="1.5" fill="currentColor" opacity="0.4" />
        <circle cx="22" cy="12" r="1.5" fill="currentColor" opacity="0.4" />
        <path d="M16 26h16M16 32h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.5" />
        <path d="M20 22l4 4 8-8" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

export function DashboardPanel({ token, projects, onNavigate }: DashboardPanelProps) {
  const { t, locale } = useI18n();

  // Locale-aware number formatting
  const ln = (n: number | string): string => {
    const s = String(n);
    if (locale === "fa") return s.replace(/\d/g, d => "۰۱۲۳۴۵۶۷۸۹"[+d]);
    if (locale === "ar") return s.replace(/\d/g, d => "٠١٢٣٤٥٦٧٨٩"[+d]);
    return s;
  };

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

  // Onboarding steps
  const hasProject = projects.length > 0;
  const hasWp = wordpressConnected > 0;
  const onboardingSteps = [
    { key: "createProject", done: hasProject, label: t("dashboard.onboarding.createProject"), primary: true },
    { key: "connectWp", done: hasWp, label: t("dashboard.onboarding.connectWp"), primary: false },
    { key: "setupRulebook", done: false, label: t("dashboard.onboarding.setupRulebook"), primary: false },
    { key: "generateFirst", done: todayArticles > 0, label: t("dashboard.onboarding.generateFirst"), primary: false },
  ];
  const completedSteps = onboardingSteps.filter((s) => s.done).length;

  const hasData = projects.length > 0;

  return (
    <section className="animate-fade-in">

      {/* ── Page title — aligned with content grid ── */}
      <h2 className="text-[24px] font-bold text-gray-900 mb-5">{t("dashboard.title")}</h2>

      {/* ── Cost warnings ── */}
      {percent >= 95 && (
        <div className="mb-4 rounded-xl border-s-4 border-s-red-500 bg-red-50 px-4 py-3 text-[13px] font-semibold text-red-700" role="alert">
          ⚠ {t("dashboard.costWarning95")}
        </div>
      )}
      {percent >= 80 && percent < 95 && (
        <div className="mb-4 rounded-xl border-s-4 border-s-amber-500 bg-amber-50 px-4 py-3 text-[13px] font-semibold text-amber-700" role="alert">
          ⚠ {t("dashboard.costWarning80")}
        </div>
      )}

      {/* ═══════════════════════════════════════════════
          SMART EMPTY STATE — 0 projects: No metric cards.
          Central onboarding with integrated checklist.
          ═══════════════════════════════════════════════ */}
      {!hasData && !loading && (
        <div className="flex items-center justify-center" style={{ minHeight: "calc(100vh - 200px)" }}>
          <div className="w-full max-w-md text-center">
            <WorkspaceIllustration />

            <h3 className="text-[20px] font-bold text-gray-900 mb-2">
              {t("dashboard.noProjects")}
            </h3>
            <p className="text-[14px] text-gray-500 mb-6 leading-relaxed">
              {t("dashboard.createFirst")}
            </p>

            {/* Primary CTA */}
            {onNavigate && (
              <Button
                variant="primary"
                size="lg"
                onClick={() => onNavigate("projects")}
                className="mb-8 px-8"
              >
                + {t("projects.createProject")}
              </Button>
            )}

            {/* Remaining onboarding steps — disabled/locked */}
            <div className="rounded-2xl border border-gray-100 bg-white p-5 text-start shadow-sm">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-gray-400 mb-3">
                {t("dashboard.onboardingTitle")} · {ln(completedSteps)}/{ln(4)}
              </p>
              <ul className="space-y-3">
                {onboardingSteps.map((step, i) => (
                  <li key={step.key} className="flex items-center gap-3">
                    <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-full text-[11px] font-bold ${step.done
                        ? "bg-teal-600 text-white"
                        : i === 0
                          ? "border-2 border-teal-600 text-teal-600"
                          : "border border-gray-200 text-gray-300"
                      }`}>
                      {step.done ? "✓" : ln(i + 1)}
                    </span>
                    <span className={`text-[14px] ${step.done
                        ? "text-gray-400 line-through"
                        : i === 0
                          ? "text-gray-700 font-medium"
                          : "text-gray-300"
                      }`}>
                      {step.label}
                    </span>
                    {!step.done && i > 0 && (
                      <svg viewBox="0 0 16 16" className="ms-auto h-3.5 w-3.5 text-gray-200" fill="currentColor">
                        <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 12.5A5.5 5.5 0 1 1 8 2.5a5.5 5.5 0 0 1 0 11zM7.25 5v3.25L9.5 9.5l.75-1.25L8.75 7.25V5h-1.5z" />
                      </svg>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="rounded-2xl border border-gray-100 bg-white p-5 shadow-sm">
              <div className="h-4 w-20 rounded bg-gray-100 animate-pulse mb-3" />
              <div className="h-9 w-16 rounded-lg bg-gray-100 animate-pulse" />
            </div>
          ))}
        </div>
      )}

      {/* ═══════════════════════════════════════════════
          ACTIVE STATE — has data: show metrics + cost + projects
          ═══════════════════════════════════════════════ */}
      {hasData && !loading && (
        <div className="space-y-5">

          {/* ── Metric Cards Grid ── */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label={t("dashboard.totalProjects")}
              value={ln(projects.length)}
              onClick={() => onNavigate?.("projects")}
            />
            <MetricCard
              label={t("dashboard.articlesToday")}
              value={ln(todayArticles)}
            />
            <MetricCard
              label={t("dashboard.wpConnected")}
              value={ln(wordpressConnected)}
              statusDot={wordpressConnected > 0 ? "bg-emerald-500" : "bg-gray-300"}
            />
            <MetricCard
              label={t("dashboard.wpPending")}
              value={ln(pendingWordpress)}
              statusDot={pendingWordpress > 0 ? "bg-amber-400" : "bg-emerald-500"}
            />
          </div>

          {/* ── LLM Cost Card + Onboarding row ── */}
          <div className={completedSteps < 4 ? "grid gap-4 lg:grid-cols-2" : ""}>

            {/* LLM Daily Cost */}
            <div className="rounded-2xl border border-gray-100 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <p className="text-[13px] font-medium uppercase tracking-wider text-gray-500">
                  {t("dashboard.llmCostToday")}
                </p>
                {health && (
                  <span className={`inline-flex items-center gap-1.5 text-[12px] font-medium ${isHealthy ? "text-emerald-600" : "text-amber-600"
                    }`}>
                    <span className={`h-2 w-2 rounded-full ${isHealthy ? "bg-emerald-500" : "bg-amber-400"}`} aria-hidden />
                    {healthLabel}
                  </span>
                )}
              </div>
              <p className="text-[32px] font-bold text-gray-900 leading-none mb-3">
                ${ln(todayCost.toFixed(2))}
              </p>
              <ProgressBar
                value={Math.max(percent, 1.5)}
                showLabel
                label={t("dashboard.ofCap", { percent: ln(percent.toFixed(0)), cap: ln(threshold.toFixed(0)) })}
              />
            </div>

            {/* Inline onboarding (when user has some progress but not complete) */}
            {completedSteps < 4 && (
              <div className="rounded-2xl border border-gray-100 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-[13px] font-semibold text-gray-700">
                    {t("dashboard.onboardingTitle")} · {ln(completedSteps)}/{ln(4)}
                  </p>
                  <ProgressBar value={(completedSteps / 4) * 100} className="w-20" />
                </div>
                <ul className="space-y-2.5">
                  {onboardingSteps.map((step, i) => (
                    <li key={step.key} className="flex items-center gap-2.5">
                      <span className={`grid h-5 w-5 shrink-0 place-items-center rounded-full text-[10px] font-bold ${step.done
                          ? "bg-teal-600 text-white"
                          : "border border-gray-200 text-gray-400"
                        }`}>
                        {step.done ? "✓" : ln(i + 1)}
                      </span>
                      <span className={`text-[13px] ${step.done ? "text-gray-400 line-through" : "text-gray-600"}`}>
                        {step.label}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* ── Recent Projects ── */}
          {recentProjects.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-[14px] font-semibold text-gray-700">{t("dashboard.recentProjects")}</h3>
                {overflowCount > 0 && (
                  <button
                    type="button"
                    onClick={() => onNavigate?.("projects")}
                    className="text-[13px] font-medium text-teal-600 hover:text-teal-700 transition-colors"
                  >
                    {t("dashboard.moreProjects", { count: overflowCount })}
                  </button>
                )}
              </div>
              <div className="grid gap-3 md:grid-cols-3">
                {recentProjects.map((project) => (
                  <article
                    key={project.id}
                    className="rounded-2xl border border-gray-100 bg-white px-4 py-3.5 shadow-sm cursor-pointer transition-all duration-200 hover:shadow-md hover:border-gray-200"
                    onClick={() => onNavigate?.("projects")}
                  >
                    <div className="flex items-center gap-2 mb-0.5">
                      <p className="truncate text-[14px] font-semibold text-gray-900" title={project.name}>{project.name}</p>
                      {project.wordpress_url && (
                        <span className="shrink-0 rounded-md bg-emerald-50 px-1.5 py-0.5 text-[10px] font-bold text-emerald-600 uppercase">WP</span>
                      )}
                    </div>
                    <p className="truncate text-[12px] text-gray-400" title={project.domain || undefined}>
                      {project.domain || t("projects.noDomain")}
                    </p>
                  </article>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
