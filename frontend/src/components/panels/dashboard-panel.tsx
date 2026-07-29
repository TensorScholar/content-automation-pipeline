"use client";

import { useEffect, useMemo, useState } from "react";
import { apiRequest } from "@/lib/api";
import { Project } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Button } from "@/components/ui/button";

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
  isAdmin?: boolean;
  onNavigate?: (page: string) => void;
}

type Tone = "neutral" | "good" | "warning";

function toneClasses(tone: Tone) {
  if (tone === "good") return "bg-emerald-500";
  if (tone === "warning") return "bg-amber-400";
  return "bg-gray-300 dark:bg-white/30";
}

function numberLocaleFor(locale: string) {
  if (locale === "fa") return "fa-IR";
  if (locale === "ar") return "ar";
  return "en-US";
}

function StatusTile({
  label,
  value,
  detail,
  kind = "number",
  tone = "neutral",
}: {
  label: string;
  value: string;
  detail: string;
  kind?: "number" | "status";
  tone?: Tone;
}) {
  return (
    <div className="smx-panel min-w-0 p-5">
      <div className="mb-2 flex items-center justify-between gap-3">
        <p className="truncate text-[12px] font-medium text-ink-tertiary">{label}</p>
        <span className={`h-2.5 w-2.5 shrink-0 rounded-full shadow-[0_0_0_3px_rgb(0_0_0/0.03)] dark:shadow-[0_0_0_3px_rgb(255_255_255/0.04)] ${toneClasses(tone)}`} aria-hidden />
      </div>
      <p
        className={`truncate font-semibold leading-tight tracking-normal text-ink tabular-nums ${
          kind === "number" ? "text-[20px]" : "text-[16px]"
        }`}
        dir="auto"
      >
        {value}
      </p>
      <p className="mt-1.5 truncate text-[12px] font-medium text-ink-tertiary">{detail}</p>
    </div>
  );
}

function StepRow({
  label,
  state,
  context,
  status,
}: {
  label: string;
  state: "done" | "pending" | "unverified";
  context: string;
  status: string;
}) {
  const done = state === "done";
  return (
    <li className="flex min-h-11 items-center gap-3 border-t border-black/5 px-4 first:border-t-0 dark:border-white/10">
      <span
        className={`grid h-5 w-5 shrink-0 place-items-center rounded-full text-[10px] font-semibold ${
          done
            ? "bg-brand text-white"
            : state === "unverified"
              ? "border border-amber-400/50 text-amber-700 dark:text-amber-300"
              : "border border-black/10 text-ink-tertiary dark:border-white/10"
        }`}
      >
        {done ? "✓" : state === "unverified" ? "?" : "–"}
      </span>
      <div className="min-w-0 flex-1">
        <span className={`block truncate text-[13px] ${done ? "text-ink-tertiary line-through" : "text-ink-secondary"}`}>{label}</span>
      </div>
      <span className="shrink-0 text-[11px] font-medium text-ink-tertiary">{context}</span>
      <span className="shrink-0 text-[11px] font-semibold text-ink-secondary">{status}</span>
    </li>
  );
}

export function DashboardPanel({ token, projects, isAdmin = false, onNavigate }: DashboardPanelProps) {
  const { t, locale } = useI18n();
  const [performance, setPerformance] = useState<PerformancePayload | null>(null);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    const load = async () => {
      try {
        const [perfResult, healthResult] = await Promise.allSettled([
          isAdmin
            ? apiRequest<PerformancePayload>("/system/performance", { token, signal: controller.signal })
            : Promise.resolve(null),
          apiRequest<HealthPayload>("/system/health", { token, signal: controller.signal }),
        ]);
        if (controller.signal.aborted) return;
        setPerformance(perfResult.status === "fulfilled" ? perfResult.value : null);
        setHealth(healthResult.status === "fulfilled" ? healthResult.value : null);
      } catch {
        if (controller.signal.aborted) return;
        setPerformance(null);
        setHealth(null);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };

    void load();
    return () => controller.abort();
  }, [isAdmin, token]);

  const numberLocale = numberLocaleFor(locale);
  const formatNumber = useMemo(
    () => new Intl.NumberFormat(numberLocale, { maximumFractionDigits: 0 }).format,
    [numberLocale]
  );
  const formatPercent = useMemo(
    () => new Intl.NumberFormat(numberLocale, { maximumFractionDigits: 0 }).format,
    [numberLocale]
  );
  const formatDecimal = useMemo(
    () => new Intl.NumberFormat(numberLocale, { maximumFractionDigits: 2, minimumFractionDigits: 2 }).format,
    [numberLocale]
  );
  const formatCurrency = useMemo(() => {
    if (locale === "en") {
      return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2,
        minimumFractionDigits: 2,
      }).format;
    }

    return (value: number) => `${formatDecimal(value)} USD`;
  }, [formatDecimal, locale]);

  const wordpressConnected = useMemo(() => projects.filter((p) => Boolean(p.wordpress_url)).length, [projects]);

  const daily = performance?.metrics?.daily_costs;
  const todayCost = daily?.total_cost_usd ?? 0;
  const todayArticles = daily?.article_count ?? 0;
  const threshold = daily?.threshold_usd ?? 10;
  const percent = threshold > 0 ? Math.min(100, (todayCost / threshold) * 100) : 0;
  const isHealthy = health?.status?.toLowerCase().includes("healthy") ?? true;
  const healthLabel = isHealthy ? t("dashboard.systemHealthy") : t("dashboard.systemUnhealthy");

  const hasProject = projects.length > 0;
  const hasWp = wordpressConnected > 0;
  const recentProjects = projects.slice(0, 4);
  const overflowCount = Math.max(0, projects.length - recentProjects.length);
  const pipelineCopy = locale === "fa"
    ? { subtitle: "وضعیت هر مورد مستقل و بر پایه داده‌های موجود نمایش داده می‌شود.", required: "الزامی", optional: "اختیاری", activity: "فعالیت", complete: "کامل", pending: "تکمیل نشده", unverified: "بررسی نشده" }
    : locale === "ar"
      ? { subtitle: "تُعرض حالة كل عنصر بشكل مستقل وفقاً للبيانات المتاحة.", required: "مطلوب", optional: "اختياري", activity: "نشاط", complete: "مكتمل", pending: "غير مكتمل", unverified: "غير متحقق" }
      : { subtitle: "Each item is shown independently from the available project data.", required: "Required", optional: "Optional", activity: "Activity", complete: "Complete", pending: "Not complete", unverified: "Not verified" };

  const nextAction = useMemo(() => {
    if (!hasProject) {
      return {
        title: t("dashboard.nextCreateProjectTitle"),
        description: t("dashboard.nextCreateProjectDesc"),
        cta: t("dashboard.actionCreateProject"),
        page: "projects",
      };
    }

    if (todayArticles === 0) {
      return {
        title: t("dashboard.nextGenerateTitle"),
        description: t("dashboard.nextGenerateDesc"),
        cta: t("dashboard.actionCreateContent"),
        page: "studio",
      };
    }

    if (!hasWp) {
      return {
        title: t("dashboard.nextConnectWpTitle"),
        description: t("dashboard.nextConnectWpDesc"),
        cta: t("dashboard.actionConnectWordpress"),
        page: "projects",
      };
    }

    return {
      title: t("dashboard.nextReviewTitle"),
      description: t("dashboard.nextReviewDesc"),
      cta: t("dashboard.actionReviewTasks"),
      page: "tasks",
    };
  }, [hasProject, hasWp, t, todayArticles]);

  const pipelineSteps: Array<{
    key: string;
    label: string;
    state: "done" | "pending" | "unverified";
    context: string;
    status: string;
  }> = [
    {
      key: "project",
      label: t("dashboard.pipelineProject"),
      state: hasProject ? "done" : "pending",
      context: pipelineCopy.required,
      status: hasProject ? pipelineCopy.complete : pipelineCopy.pending,
    },
    {
      key: "wordpress",
      label: t("dashboard.pipelineWordpress"),
      state: hasWp ? "done" : "pending",
      context: pipelineCopy.optional,
      status: hasWp ? pipelineCopy.complete : pipelineCopy.pending,
    },
    {
      key: "rules",
      label: t("dashboard.pipelineRules"),
      state: "unverified",
      context: pipelineCopy.required,
      status: pipelineCopy.unverified,
    },
    {
      key: "generate",
      label: t("dashboard.pipelineGenerate"),
      state: todayArticles > 0 ? "done" : "pending",
      context: pipelineCopy.activity,
      status: todayArticles > 0 ? pipelineCopy.complete : pipelineCopy.pending,
    },
  ];

  if (loading) {
    return (
      <section className="mx-auto flex min-h-full w-full max-w-[1120px] flex-col gap-4 py-1">
        <div className="smx-panel h-16" />
        <div className="grid gap-3 sm:grid-cols-4">
          {[1, 2, 3, 4].map((item) => (
            <div key={item} className="smx-panel h-24 px-4 py-4">
              <div className="mb-3 h-3 w-24 animate-pulse rounded bg-black/10 dark:bg-white/10" />
              <div className="h-7 w-14 animate-pulse rounded bg-black/10 dark:bg-white/10" />
              <div className="mt-3 h-3 w-20 animate-pulse rounded bg-black/10 dark:bg-white/10" />
            </div>
          ))}
        </div>
        <div className="smx-panel min-h-0 flex-1" />
      </section>
    );
  }

  return (
    <section className="mx-auto flex min-h-full w-full max-w-[1120px] flex-col gap-4 py-1">
      <header className="flex min-h-[72px] items-end justify-between gap-4 border-b border-black/5 pb-4 dark:border-white/10">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="truncate text-[24px] font-semibold leading-tight tracking-normal text-ink">
              {t("dashboard.commandTitle")}
            </h2>
          </div>
          <p className="mt-1 truncate text-[13px] font-medium text-ink-secondary" dir="auto">
            {t("dashboard.commandSubtitle")}
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <Button variant="outlined" size="md" onClick={() => onNavigate?.("projects")}>
            {t("dashboard.secondaryProject")}
          </Button>
          <Button variant="primary" size="md" onClick={() => onNavigate?.("studio")}>
            {t("dashboard.primaryCreate")}
          </Button>
        </div>
      </header>

      {percent >= 95 && (
        <div className="rounded-md border-s-2 border-s-red-500 bg-red-50/80 px-3 py-2 text-[12px] font-medium text-red-700 dark:bg-red-500/10 dark:text-red-300" role="alert">
          {t("dashboard.costWarning95")}
        </div>
      )}
      {percent >= 80 && percent < 95 && (
        <div className="rounded-md border-s-2 border-s-amber-500 bg-amber-50/80 px-3 py-2 text-[12px] font-medium text-amber-700 dark:bg-amber-500/10 dark:text-amber-300" role="alert">
          {t("dashboard.costWarning80")}
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatusTile
          label={t("dashboard.statusProject")}
          value={formatNumber(projects.length)}
          detail={projects.length > 1 ? t("dashboard.moreProjects", { count: formatNumber(projects.length - 1) }) : (hasProject ? t("dashboard.ready") : t("dashboard.noProjects"))}
          tone={hasProject ? "good" : "warning"}
        />
        <StatusTile
          label={t("dashboard.statusWordpress")}
          value={hasWp ? t("dashboard.connected") : t("dashboard.notConnected")}
          detail={hasWp ? t("dashboard.ready") : t("dashboard.needsSetup")}
          kind="status"
          tone={hasWp ? "good" : "warning"}
        />
        <StatusTile
          label={t("dashboard.statusToday")}
          value={formatNumber(todayArticles)}
          detail={t("dashboard.articlesToday")}
          tone={todayArticles > 0 ? "good" : "neutral"}
        />
        <StatusTile
          label={t("dashboard.statusSystem")}
          value={healthLabel}
          detail={health?.version ? `v${health.version}` : t("dashboard.lastUpdated")}
          kind="status"
          tone={isHealthy ? "good" : "warning"}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_320px]">
        <main className="space-y-4 pe-1">
          <section className="smx-panel p-5">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-[12px] font-medium text-ink-tertiary">{t("dashboard.nextStep")}</p>
                <h3 className="mt-1 text-[18px] font-semibold tracking-normal text-ink">{nextAction.title}</h3>
                <p className="mt-2 max-w-2xl text-[13px] leading-5 text-ink-secondary">{nextAction.description}</p>
              </div>
              <Button variant="primary" size="md" onClick={() => onNavigate?.(nextAction.page)} className="shrink-0">
                {nextAction.cta}
              </Button>
            </div>
          </section>

          <section className="smx-panel overflow-hidden">
            <div className="flex items-center justify-between gap-3 px-4 py-3">
              <div className="min-w-0">
                <h3 className="truncate text-[14px] font-semibold text-ink">{t("dashboard.pipelineTitle")}</h3>
                <p className="mt-1 truncate text-[12px] text-ink-tertiary">{pipelineCopy.subtitle}</p>
              </div>
            </div>
            <ul>
              {pipelineSteps.map((step) => (
                <StepRow
                  key={step.key}
                  label={step.label}
                  state={step.state}
                  context={step.context}
                  status={step.status}
                />
              ))}
            </ul>
          </section>

          <section className="smx-panel overflow-hidden">
            <div className="flex items-center justify-between gap-3 border-b border-black/5 px-4 py-3 dark:border-white/10">
              <h3 className="text-[14px] font-semibold text-ink">{t("dashboard.recentWork")}</h3>
              <button
                type="button"
                onClick={() => onNavigate?.("projects")}
                className="text-[12px] font-medium text-brand transition-colors hover:text-brand-hover"
              >
                {t("dashboard.openProjects")}
              </button>
            </div>
            {recentProjects.length === 0 ? (
              <p className="px-4 py-6 text-center text-[13px] text-ink-tertiary">{t("dashboard.noRecentWork")}</p>
            ) : (
              <div className="divide-y divide-black/5 dark:divide-white/10">
                {recentProjects.map((project) => (
                  <button
                    key={project.id}
                    type="button"
                    className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-3 bg-transparent px-4 py-3 text-start transition-colors hover:bg-black/5 dark:hover:bg-white/[0.05]"
                    onClick={() => onNavigate?.("projects")}
                  >
                    <span className="min-w-0">
                      <span className="block truncate text-[13px] font-medium text-ink" title={project.name}>
                        {project.name}
                      </span>
                      <span className="mt-0.5 block truncate text-[12px] text-ink-tertiary" title={project.domain || undefined} dir="ltr">
                        {project.domain || t("projects.noDomain")}
                      </span>
                    </span>
                    <span
                      className={`inline-flex h-6 items-center rounded-md px-2 text-[11px] font-medium ${
                        project.wordpress_url
                          ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                          : "bg-amber-500/10 text-amber-700 dark:text-amber-300"
                      }`}
                    >
                      {project.wordpress_url ? t("dashboard.connected") : t("dashboard.needsSetup")}
                    </span>
                  </button>
                ))}
                {overflowCount > 0 && (
                  <button
                    type="button"
                    onClick={() => onNavigate?.("projects")}
                    className="w-full bg-transparent px-4 py-2 text-start text-[12px] font-medium text-brand hover:bg-black/5 dark:hover:bg-white/[0.05]"
                  >
                    {t("dashboard.moreProjects", { count: formatNumber(overflowCount) })}
                  </button>
                )}
              </div>
            )}
          </section>
        </main>

        <aside className="space-y-4">
          <section className="smx-panel p-5">
            <div className="mb-3 flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-[12px] font-medium text-ink-secondary">{t("dashboard.costTitle")}</p>
                <p className="mt-1 text-[24px] font-semibold leading-none tracking-normal text-ink tabular-nums" dir="ltr">
                  {formatCurrency(todayCost)}
                </p>
              </div>
              <span
                className={`inline-flex h-6 items-center gap-1.5 rounded-md px-2 text-[11px] font-medium ${
                  isHealthy ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300" : "bg-amber-500/10 text-amber-700 dark:text-amber-300"
                }`}
              >
                <span className={`h-1.5 w-1.5 rounded-full ${isHealthy ? "bg-emerald-500" : "bg-amber-400"}`} aria-hidden />
                {healthLabel}
              </span>
            </div>
            <ProgressBar
              value={Math.max(percent, todayCost > 0 ? 1.5 : 0)}
              showLabel
              label={t("dashboard.ofCap", {
                percent: formatPercent(percent),
                cap: formatNumber(threshold),
              })}
            />
          </section>

          <section className="smx-panel p-5">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h3 className="text-[14px] font-semibold text-ink">{t("dashboard.budgetStatus")}</h3>
              <span className="text-[12px] text-ink-tertiary tabular-nums">
                {formatPercent(percent)}%
              </span>
            </div>
            <div className="smx-panel-subtle grid grid-cols-2 overflow-hidden">
              <div className="border-e border-black/5 p-3 dark:border-white/10">
                <p className="text-[11px] text-ink-tertiary">{t("dashboard.articlesToday")}</p>
                <p className="mt-1 text-[20px] font-semibold text-ink tabular-nums">{formatNumber(todayArticles)}</p>
              </div>
              <div className="p-3">
                <p className="text-[11px] text-ink-tertiary">{t("dashboard.totalProjects")}</p>
                <p className="mt-1 text-[20px] font-semibold text-ink tabular-nums">{formatNumber(projects.length)}</p>
              </div>
            </div>
          </section>
        </aside>
      </div>
    </section>
  );
}
