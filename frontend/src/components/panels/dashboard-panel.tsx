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

type Tone = "neutral" | "good" | "warning" | "danger";
type TelemetryAvailability = "loading" | "available" | "unavailable";
type HealthState = "loading" | "healthy" | "degraded" | "unhealthy" | "unavailable";

function authoritativeNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

function classifyHealth(
  payload: HealthPayload | null,
  availability: TelemetryAvailability
): HealthState {
  if (availability === "loading") return "loading";
  if (availability !== "available" || typeof payload?.status !== "string") return "unavailable";

  const status = payload.status.trim().toLowerCase();
  if (status === "healthy") return "healthy";
  if (status === "degraded") return "degraded";
  if (status === "unhealthy") return "unhealthy";
  return "unavailable";
}

function toneClasses(tone: Tone) {
  if (tone === "good") return "bg-emerald-500";
  if (tone === "warning") return "bg-amber-400";
  if (tone === "danger") return "bg-rose-500";
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
    <div className="min-w-0 bg-[rgb(var(--bg-elevated))] px-4 py-4 sm:px-5 sm:py-[18px] dark:bg-[rgb(var(--bg-elevated-dark))]">
      <div className="flex items-center gap-2">
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${toneClasses(tone)}`} aria-hidden />
        <p className="truncate text-[12px] font-medium text-ink-tertiary">{label}</p>
      </div>
      <p
        className={`mt-2 truncate font-semibold leading-none tracking-normal text-ink tabular-nums ${
          kind === "number" ? "text-[22px]" : "text-[17px]"
        }`}
        dir="auto"
      >
        {value}
      </p>
      <p className="mt-2 truncate text-[12px] font-medium text-ink-tertiary">{detail}</p>
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
  const statusClass = done
    ? "text-emerald-700 dark:text-emerald-300"
    : state === "unverified"
      ? "text-amber-700 dark:text-amber-300"
      : "text-ink-secondary";

  return (
    <li className="grid min-h-[54px] grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-3 border-t border-black/5 px-4 py-2.5 first:border-t-0 dark:border-white/10">
      <span
        className={`grid h-[18px] w-[18px] shrink-0 place-items-center rounded-full text-[9px] font-semibold ${
          done
            ? "bg-brand text-white"
            : state === "unverified"
              ? "border border-amber-400/50 text-amber-700 dark:text-amber-300"
              : "border border-black/10 text-ink-tertiary dark:border-white/10"
        }`}
        aria-hidden
      >
        {done ? "✓" : state === "unverified" ? "?" : "–"}
      </span>
      <div className="min-w-0">
        <span className="block truncate text-[13px] font-medium text-ink">{label}</span>
        <span className="mt-0.5 block truncate text-[11px] text-ink-tertiary">{context}</span>
      </div>
      <span className={`max-w-[104px] shrink-0 truncate text-end text-[11px] font-semibold ${statusClass}`} title={status}>{status}</span>
    </li>
  );
}

export function DashboardPanel({ token, projects, isAdmin = false, onNavigate }: DashboardPanelProps) {
  const { t, locale } = useI18n();
  const [performance, setPerformance] = useState<PerformancePayload | null>(null);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [performanceAvailability, setPerformanceAvailability] = useState<TelemetryAvailability>("loading");
  const [healthAvailability, setHealthAvailability] = useState<TelemetryAvailability>("loading");

  useEffect(() => {
    const controller = new AbortController();
    const load = async () => {
      setPerformance(null);
      setHealth(null);
      setPerformanceAvailability("loading");
      setHealthAvailability("loading");

      try {
        const [perfResult, healthResult] = await Promise.allSettled([
          isAdmin
            ? apiRequest<PerformancePayload>("/system/performance", { token, signal: controller.signal })
            : Promise.resolve(null),
          apiRequest<HealthPayload>("/system/health", { token, signal: controller.signal }),
        ]);
        if (controller.signal.aborted) return;

        const performanceAvailable = isAdmin && perfResult.status === "fulfilled" && perfResult.value !== null;
        setPerformance(performanceAvailable ? perfResult.value : null);
        setPerformanceAvailability(performanceAvailable ? "available" : "unavailable");

        const healthAvailable = healthResult.status === "fulfilled" && healthResult.value !== null;
        setHealth(healthAvailable ? healthResult.value : null);
        setHealthAvailability(healthAvailable ? "available" : "unavailable");
      } catch {
        if (controller.signal.aborted) return;
        setPerformance(null);
        setHealth(null);
        setPerformanceAvailability("unavailable");
        setHealthAvailability("unavailable");
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
  const todayCost = performanceAvailability === "available"
    ? authoritativeNumber(daily?.total_cost_usd)
    : null;
  const todayArticles = performanceAvailability === "available"
    ? authoritativeNumber(daily?.article_count)
    : null;
  const threshold = performanceAvailability === "available"
    ? authoritativeNumber(daily?.threshold_usd)
    : null;
  const percent = todayCost !== null && threshold !== null && threshold > 0
    ? Math.min(100, (todayCost / threshold) * 100)
    : null;
  const healthState = classifyHealth(health, healthAvailability);
  const healthLabel = healthState === "healthy"
    ? t("dashboard.systemHealthy")
    : healthState === "degraded"
      ? t("dashboard.systemDegraded")
      : healthState === "unhealthy"
        ? t("dashboard.systemUnhealthy")
        : t("dashboard.unavailable");
  const healthTone: Tone = healthState === "healthy"
    ? "good"
    : healthState === "degraded"
      ? "warning"
      : healthState === "unhealthy"
        ? "danger"
        : "neutral";
  const loading = performanceAvailability === "loading" || healthAvailability === "loading";

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

    if (todayArticles === null) {
      return {
        title: t("dashboard.nextContinueTitle"),
        description: t("dashboard.nextContinueDesc"),
        cta: t("dashboard.actionContinue"),
        page: "studio",
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
      state: todayArticles === null ? "unverified" : todayArticles > 0 ? "done" : "pending",
      context: pipelineCopy.activity,
      status: todayArticles === null
        ? t("dashboard.unavailable")
        : todayArticles > 0
          ? pipelineCopy.complete
          : pipelineCopy.pending,
    },
  ];

  if (loading) {
    return (
      <section className="mx-auto flex min-h-full w-full max-w-[1180px] flex-col gap-5 py-2">
        <div className="h-[76px] border-b border-black/5 pb-5 dark:border-white/10">
          <div className="h-7 w-44 animate-pulse rounded bg-black/10 dark:bg-white/10" />
          <div className="mt-2 h-3 w-64 max-w-full animate-pulse rounded bg-black/10 dark:bg-white/10" />
        </div>
        <div className="smx-panel grid grid-cols-1 gap-px overflow-hidden bg-black/[0.06] sm:grid-cols-2 xl:grid-cols-4 dark:bg-white/10">
          {[1, 2, 3, 4].map((item) => (
            <div key={item} className="h-[108px] bg-[rgb(var(--bg-elevated))] px-5 py-4 dark:bg-[rgb(var(--bg-elevated-dark))]">
              <div className="h-3 w-24 animate-pulse rounded bg-black/10 dark:bg-white/10" />
              <div className="mt-3 h-6 w-16 animate-pulse rounded bg-black/10 dark:bg-white/10" />
              <div className="mt-3 h-3 w-20 animate-pulse rounded bg-black/10 dark:bg-white/10" />
            </div>
          ))}
        </div>
        <div className="grid min-h-[420px] gap-5 xl:grid-cols-[minmax(0,1fr)_336px]">
          <div className="smx-panel min-h-[320px] animate-pulse" />
          <div className="space-y-5">
            <div className="smx-panel h-52 animate-pulse" />
            <div className="smx-panel h-64 animate-pulse" />
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="mx-auto flex min-h-full w-full max-w-[1180px] flex-col gap-5 py-2">
      <header className="flex min-h-[76px] flex-col justify-center gap-4 border-b border-black/5 pb-5 sm:flex-row sm:items-center sm:justify-between dark:border-white/10">
        <div className="min-w-0">
          <h2 className="truncate text-[25px] font-semibold leading-tight tracking-normal text-ink sm:text-[26px]">
            {t("dashboard.commandTitle")}
          </h2>
          <p className="mt-1.5 truncate text-[13px] font-medium text-ink-secondary" dir="auto">
            {t("dashboard.commandSubtitle")}
          </p>
        </div>

        <div className="flex w-full shrink-0 items-center gap-2 sm:w-auto">
          <Button
            variant="outlined"
            size="md"
            onClick={() => onNavigate?.("projects")}
            className="flex-1 sm:flex-none"
          >
            {t("dashboard.secondaryProject")}
          </Button>
          <Button
            variant="primary"
            size="md"
            onClick={() => onNavigate?.("studio")}
            className="flex-1 sm:flex-none"
          >
            {t("dashboard.primaryCreate")}
          </Button>
        </div>
      </header>

      {percent !== null && percent >= 95 && (
        <div className="rounded-lg border border-red-500/20 bg-red-50/70 px-3.5 py-2.5 text-[12px] font-medium text-red-700 dark:bg-red-500/10 dark:text-red-300" role="alert">
          {t("dashboard.costWarning95")}
        </div>
      )}
      {percent !== null && percent >= 80 && percent < 95 && (
        <div className="rounded-lg border border-amber-500/20 bg-amber-50/70 px-3.5 py-2.5 text-[12px] font-medium text-amber-700 dark:bg-amber-500/10 dark:text-amber-300" role="alert">
          {t("dashboard.costWarning80")}
        </div>
      )}

      <section
        className="smx-panel grid grid-cols-1 gap-px overflow-hidden bg-black/[0.06] sm:grid-cols-2 xl:grid-cols-4 dark:bg-white/10"
        aria-label={t("dashboard.commandSubtitle")}
      >
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
          value={todayArticles === null ? "—" : formatNumber(todayArticles)}
          detail={todayArticles === null ? t("dashboard.metricsUnavailable") : t("dashboard.articlesToday")}
          tone={todayArticles !== null && todayArticles > 0 ? "good" : "neutral"}
        />
        <StatusTile
          label={t("dashboard.statusSystem")}
          value={healthLabel}
          detail={healthState === "unavailable"
            ? t("dashboard.healthUnavailable")
            : health?.version
              ? `v${health.version}`
              : t("dashboard.lastUpdated")}
          kind="status"
          tone={healthTone}
        />
      </section>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_336px]">
        <div className="space-y-5">
          <section className="smx-panel border-brand/[0.15] bg-brand/[0.025] p-5 dark:bg-brand/[0.045] sm:p-6">
            <div className="flex flex-col items-start justify-between gap-5 sm:flex-row sm:items-center">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-brand" aria-hidden />
                  <p className="text-[12px] font-semibold text-brand">{t("dashboard.nextStep")}</p>
                </div>
                <h3 className="mt-2 text-[19px] font-semibold tracking-normal text-ink">{nextAction.title}</h3>
                <p className="mt-2 max-w-2xl text-[13px] leading-5 text-ink-secondary">{nextAction.description}</p>
              </div>
              <Button
                variant="primary"
                size="md"
                onClick={() => onNavigate?.(nextAction.page)}
                className="w-full shrink-0 sm:w-auto"
              >
                {nextAction.cta}
              </Button>
            </div>
          </section>

          <section className="smx-panel overflow-hidden">
            <div className="flex items-center justify-between gap-3 border-b border-black/5 px-4 py-3.5 sm:px-5 dark:border-white/10">
              <h3 className="text-[14px] font-semibold text-ink">{t("dashboard.recentWork")}</h3>
              <button
                type="button"
                onClick={() => onNavigate?.("projects")}
                className="rounded-md px-1 py-0.5 text-[12px] font-medium text-brand transition-colors hover:text-brand-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/30"
              >
                {t("dashboard.openProjects")}
              </button>
            </div>
            {recentProjects.length === 0 ? (
              <p className="px-5 py-10 text-center text-[13px] text-ink-tertiary">{t("dashboard.noRecentWork")}</p>
            ) : (
              <div className="divide-y divide-black/5 dark:divide-white/10">
                {recentProjects.map((project) => (
                  <button
                    key={project.id}
                    type="button"
                    className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-4 bg-transparent px-4 py-3.5 text-start transition-colors hover:bg-black/[0.025] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-brand/30 sm:px-5 dark:hover:bg-white/[0.035]"
                    onClick={() => onNavigate?.("projects")}
                  >
                    <span className="min-w-0">
                      <span className="block truncate text-[13px] font-medium text-ink" title={project.name}>
                        {project.name}
                      </span>
                      <span className="mt-1 block truncate text-[12px] text-ink-tertiary" title={project.domain || undefined} dir="ltr">
                        {project.domain || t("projects.noDomain")}
                      </span>
                    </span>
                    <span
                      className={`inline-flex items-center gap-2 text-[11px] font-medium ${
                        project.wordpress_url
                          ? "text-emerald-700 dark:text-emerald-300"
                          : "text-amber-700 dark:text-amber-300"
                      }`}
                    >
                      <span className={`h-1.5 w-1.5 rounded-full ${project.wordpress_url ? "bg-emerald-500" : "bg-amber-400"}`} aria-hidden />
                      {project.wordpress_url ? t("dashboard.connected") : t("dashboard.needsSetup")}
                    </span>
                  </button>
                ))}
                {overflowCount > 0 && (
                  <button
                    type="button"
                    onClick={() => onNavigate?.("projects")}
                    className="w-full bg-transparent px-4 py-2.5 text-start text-[12px] font-medium text-brand transition-colors hover:bg-black/[0.025] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-brand/30 sm:px-5 dark:hover:bg-white/[0.035]"
                  >
                    {t("dashboard.moreProjects", { count: formatNumber(overflowCount) })}
                  </button>
                )}
              </div>
            )}
          </section>
        </div>

        <aside className="space-y-5">
          <section className="smx-panel overflow-hidden">
            <div className="p-5">
              <div className="min-w-0">
                <p className="text-[12px] font-medium text-ink-secondary">{t("dashboard.costTitle")}</p>
                <p className="mt-2 text-[26px] font-semibold leading-none tracking-normal text-ink tabular-nums" dir="ltr">
                  {todayCost === null ? "—" : formatCurrency(todayCost)}
                </p>
              </div>

              <div className="mt-5">
                <div className="mb-2 flex items-center justify-between gap-3">
                  <h3 className="text-[12px] font-medium text-ink-secondary">{t("dashboard.budgetStatus")}</h3>
                  <span className="text-[11px] font-medium text-ink-tertiary tabular-nums">
                    {percent === null ? "—" : `${formatPercent(percent)}%`}
                  </span>
                </div>
                {percent === null || threshold === null ? (
                  <p className="py-2 text-[12px] font-medium text-ink-tertiary">
                    {t("dashboard.metricsUnavailable")}
                  </p>
                ) : (
                  <ProgressBar
                    value={Math.max(percent, todayCost !== null && todayCost > 0 ? 1.5 : 0)}
                    showLabel
                    label={t("dashboard.ofCap", {
                      percent: formatPercent(percent),
                      cap: formatNumber(threshold),
                    })}
                  />
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 border-t border-black/5 dark:border-white/10">
              <div className="border-e border-black/5 px-4 py-3.5 dark:border-white/10">
                <p className="text-[11px] text-ink-tertiary">{t("dashboard.articlesToday")}</p>
                <p className="mt-1.5 text-[19px] font-semibold text-ink tabular-nums">
                  {todayArticles === null ? "—" : formatNumber(todayArticles)}
                </p>
              </div>
              <div className="px-4 py-3.5">
                <p className="text-[11px] text-ink-tertiary">{t("dashboard.totalProjects")}</p>
                <p className="mt-1.5 text-[19px] font-semibold text-ink tabular-nums">{formatNumber(projects.length)}</p>
              </div>
            </div>
          </section>

          <section className="smx-panel overflow-hidden">
            <div className="border-b border-black/5 px-4 py-3.5 dark:border-white/10">
              <h3 className="truncate text-[14px] font-semibold text-ink">{t("dashboard.pipelineTitle")}</h3>
              <p className="mt-1 text-[12px] leading-5 text-ink-tertiary">{pipelineCopy.subtitle}</p>
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
        </aside>
      </div>
    </section>
  );
}
