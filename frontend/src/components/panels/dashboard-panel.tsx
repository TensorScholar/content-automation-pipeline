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

type DashboardDestination = "projects" | "studio" | "tasks";

type DashboardNextAction = {
  title: string;
  description: string;
  cta: string;
  page: DashboardDestination;
};

interface DashboardPanelProps {
  token: string;
  projects: Project[];
  isAdmin?: boolean;
  onNavigate?: (page: DashboardDestination) => void;
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
  if (tone === "good") return "bg-success";
  if (tone === "warning") return "bg-warning";
  if (tone === "danger") return "bg-danger";
  return "bg-ink-muted/35";
}

function numberLocaleFor(locale: string) {
  if (locale === "fa") return "fa-IR";
  if (locale === "ar") return "ar";
  return "en-US";
}

function StatusTile({
  label, value, detail, kind = "number", tone = "neutral",
}: {
  label: string; value: string; detail: string; kind?: "number" | "status"; tone?: Tone;
}) {
  return (
    <div className="min-w-0 px-4 py-4 first:ps-0 last:pe-0 sm:px-5">
      <div className="flex items-center gap-2">
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${toneClasses(tone)}`} aria-hidden />
        <p className="truncate text-xs font-medium text-ink-tertiary">{label}</p>
      </div>
      <p className={`mt-2 truncate font-semibold leading-6 text-ink tabular-nums ${kind === "number" ? "text-metric" : "text-lg"}`} dir="auto">{value}</p>
      <p className="mt-1 truncate text-xs leading-[18px] text-ink-tertiary">{detail}</p>
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
    ? "text-success"
    : state === "unverified"
      ? "text-warning"
      : "text-ink-secondary";

  return (
    <li className="grid min-h-12 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-3 border-t border-line px-4 py-2.5 first:border-t-0">
      <span
        className={`grid h-[18px] w-[18px] shrink-0 place-items-center rounded-full text-xs font-semibold ${
          done
            ? "bg-brand text-white"
            : state === "unverified"
              ? "border border-warning/50 text-warning"
              : "border border-line text-ink-tertiary "
        }`}
        aria-hidden
      >
        {done ? (
          <svg className="h-3 w-3" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="m3.4 8.1 2.7 2.7 6.5-6.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
        ) : state === "unverified" ? (
          <svg className="h-3 w-3" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M6.4 6.2A1.8 1.8 0 0 1 8.2 4.7c1 0 1.9.6 1.9 1.6 0 1.7-2.1 1.7-2.1 3.1M8 11.6h.01" strokeLinecap="round" /></svg>
        ) : (
          <svg className="h-3 w-3" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M4.5 8h7" strokeLinecap="round" /></svg>
        )}
      </span>
      <div className="min-w-0">
        <span className="block truncate text-sm font-medium text-ink">{label}</span>
        <span className="mt-0.5 block truncate text-xs text-ink-tertiary">{context}</span>
      </div>
      <span className={`max-w-[104px] shrink-0 truncate text-end text-xs font-semibold ${statusClass}`} title={status}>{status}</span>
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

  const nextAction = useMemo<DashboardNextAction>(() => {
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
      <section className="smx-page flex min-h-full flex-col gap-6">
        <div className="border-b border-line pb-5">
          <div className="h-7 w-36 animate-pulse rounded-sm bg-ink/[0.07]" />
          <div className="mt-2 h-3 w-56 animate-pulse rounded-sm bg-ink/[0.06]" />
        </div>
        <div className="grid divide-y divide-line border-b border-line sm:grid-cols-2 sm:divide-x sm:divide-y-0 xl:grid-cols-4 ">
          {[1, 2, 3, 4].map((item) => <div key={item} className="h-[98px] animate-pulse bg-ink/[0.018]" />)}
        </div>
        <div className="grid gap-7 xl:grid-cols-[minmax(0,1fr)_320px]">
          <div className="h-64 animate-pulse rounded-md bg-ink/[0.025]" />
          <div className="h-64 animate-pulse rounded-md bg-ink/[0.025]" />
        </div>
      </section>
    );
  }

  return (
    <section className="smx-page flex min-h-full flex-col gap-6">
      <header className="flex flex-col gap-4 border-b border-line pb-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <h2 className="smx-page-title">{t("dashboard.commandTitle")}</h2>
          <p className="mt-1 text-sm leading-5 text-ink-secondary">{t("dashboard.commandSubtitle")}</p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button variant="ghost" size="md" onClick={() => onNavigate?.("projects")}>{t("dashboard.secondaryProject")}</Button>
          <Button variant="primary" size="md" onClick={() => onNavigate?.("studio")}>{t("dashboard.primaryCreate")}</Button>
        </div>
      </header>

      {percent !== null && percent >= 80 ? (
        <div className={`border-s-2 px-3 py-2 text-xs leading-[18px] ${percent >= 95 ? "border-danger bg-danger-subtle text-danger" : "border-warning bg-warning-subtle text-warning"}`} role="alert">
          {percent >= 95 ? t("dashboard.costWarning95") : t("dashboard.costWarning80")}
        </div>
      ) : null}

      <section className="grid divide-y divide-line border-b border-line sm:grid-cols-2 sm:divide-x sm:divide-y-0 xl:grid-cols-4 " aria-label={t("dashboard.commandSubtitle")}>
        <StatusTile label={t("dashboard.statusProject")} value={formatNumber(projects.length)} detail={projects.length > 1 ? t("dashboard.moreProjects", { count: formatNumber(projects.length - 1) }) : (hasProject ? t("dashboard.ready") : t("dashboard.noProjects"))} tone={hasProject ? "good" : "warning"} />
        <StatusTile label={t("dashboard.statusWordpress")} value={hasWp ? t("dashboard.connected") : t("dashboard.notConnected")} detail={hasWp ? t("dashboard.ready") : t("dashboard.needsSetup")} kind="status" tone={hasWp ? "good" : "warning"} />
        <StatusTile label={t("dashboard.statusToday")} value={todayArticles === null ? "—" : formatNumber(todayArticles)} detail={todayArticles === null ? t("dashboard.metricsUnavailable") : t("dashboard.articlesToday")} tone={todayArticles !== null && todayArticles > 0 ? "good" : "neutral"} />
        <StatusTile label={t("dashboard.statusSystem")} value={healthLabel} detail={healthState === "unavailable" ? t("dashboard.healthUnavailable") : health?.version ? `v${health.version}` : t("dashboard.lastUpdated")} kind="status" tone={healthTone} />
      </section>

      <section className="grid items-center gap-5 border-b border-line pb-6 sm:grid-cols-[minmax(0,1fr)_auto]">
        <div className="min-w-0">
          <p className="flex items-center gap-2 text-xs font-medium text-brand"><span className="h-1.5 w-1.5 rounded-full bg-brand" aria-hidden />{t("dashboard.nextStep")}</p>
          <h3 className="mt-2 text-xl font-semibold leading-6 text-ink">{nextAction.title}</h3>
          <p className="mt-1.5 max-w-2xl text-sm leading-5 text-ink-secondary">{nextAction.description}</p>
        </div>
        <Button variant="primary" size="md" onClick={() => onNavigate?.(nextAction.page)}>{nextAction.cta}</Button>
      </section>

      <div className="grid gap-7 xl:grid-cols-[minmax(0,1fr)_320px]">
        <section className="min-w-0">
          <div className="flex items-center justify-between gap-3 pb-3">
            <h3 className="text-base font-semibold text-ink">{t("dashboard.recentWork")}</h3>
            <button type="button" onClick={() => onNavigate?.("projects")} className="text-xs font-medium text-brand hover:text-brand-hover">{t("dashboard.openProjects")}</button>
          </div>
          {recentProjects.length === 0 ? (
            <p className="border-t border-line py-10 text-center text-sm text-ink-tertiary">{t("dashboard.noRecentWork")}</p>
          ) : (
            <div className="border-t border-line">
              {recentProjects.map((project) => (
                <button key={project.id} type="button" className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-4 border-b border-line px-1 py-3.5 text-start transition-colors hover:bg-ink/[0.025]" onClick={() => onNavigate?.("projects")}>
                  <span className="min-w-0"><span className="block truncate text-sm font-medium text-ink">{project.name}</span><span dir="ltr" className="mt-0.5 block truncate text-left text-xs text-ink-tertiary">{project.domain || t("projects.noDomain")}</span></span>
                  <span className={`inline-flex items-center gap-2 text-xs ${project.wordpress_url ? "text-success" : "text-warning"}`}><span className={`h-1.5 w-1.5 rounded-full ${project.wordpress_url ? "bg-success" : "bg-warning"}`} aria-hidden />{project.wordpress_url ? t("dashboard.connected") : t("dashboard.needsSetup")}</span>
                </button>
              ))}
              {overflowCount > 0 ? <button type="button" onClick={() => onNavigate?.("projects")} className="py-3 text-xs font-medium text-brand">{t("dashboard.moreProjects", { count: formatNumber(overflowCount) })}</button> : null}
            </div>
          )}
        </section>

        <aside className="space-y-7">
          <section>
            <p className="text-xs font-medium text-ink-secondary">{t("dashboard.costTitle")}</p>
            <div className="mt-1 flex items-end justify-between gap-3"><p className="text-2xl font-semibold leading-8 text-ink tabular-nums" dir="ltr">{todayCost === null ? "—" : formatCurrency(todayCost)}</p><span className="text-xs tabular-nums text-ink-tertiary">{percent === null ? "—" : `${formatPercent(percent)}%`}</span></div>
            <div className="mt-3">{percent === null || threshold === null ? <p className="text-xs text-ink-tertiary">{t("dashboard.metricsUnavailable")}</p> : <ProgressBar value={Math.max(percent, todayCost !== null && todayCost > 0 ? 1.5 : 0)} showLabel label={t("dashboard.ofCap", { percent: formatPercent(percent), cap: formatNumber(threshold) })} />}</div>
          </section>

          <section>
            <h3 className="text-base font-semibold text-ink">{t("dashboard.pipelineTitle")}</h3>
            <p className="mt-1 text-xs leading-[18px] text-ink-tertiary">{pipelineCopy.subtitle}</p>
            <ul className="mt-3 border-t border-line">
              {pipelineSteps.map((step) => <StepRow key={step.key} label={step.label} state={step.state} context={step.context} status={step.status} />)}
            </ul>
          </section>
        </aside>
      </div>
    </section>
  );
}
