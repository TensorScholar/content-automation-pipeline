"use client";

import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { clsx } from "clsx";
import { apiRequest } from "@/lib/api";
import { LlmOptionsResponse } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { ProgressBar } from "@/components/ui/progress-bar";

interface MonitoringPanelProps {
  token: string;
}

interface HealthPayload {
  status?: string;
  version?: string;
  dependencies?: Record<string, string | { status?: string }>;
}

interface PerformancePayload {
  metrics?: {
    daily_costs?: { total_cost_usd?: number; article_count?: number; threshold_usd?: number };
    db_pool?: { pool_size?: number; checked_out?: number };
  };
}

interface IncidentPayload {
  incidents: Incident[];
  open_count: number;
  critical_count: number;
  warning_count: number;
  generated_at: string;
}

interface Incident {
  id: string;
  severity: "critical" | "warning" | "info" | string;
  source: string;
  status: string;
  user_message: string;
  manager_detail: string;
  created_at: string;
  project_id?: string | null;
  task_id?: string | null;
}

const GRAFANA_URL = process.env.NEXT_PUBLIC_GRAFANA_URL ?? "";

const INCIDENT_COPY = {
  en: {
    title: "Incident Inbox",
    empty: "No open incidents.",
    detail: "Manager detail",
    open: "Open",
    critical: "Critical",
    warning: "Warning",
  },
  fa: {
    title: "صندوق رخدادها",
    empty: "رخداد بازی وجود ندارد.",
    detail: "جزئیات مدیر",
    open: "باز",
    critical: "بحرانی",
    warning: "هشدار",
  },
  ar: {
    title: "صندوق الحوادث",
    empty: "لا توجد حوادث مفتوحة.",
    detail: "تفاصيل المدير",
    open: "مفتوح",
    critical: "حرج",
    warning: "تحذير",
  },
};

const LLM_COPY = {
  en: {
    title: "AI Provider Access",
    active: "Active model",
    configured: "Configured",
    missing: "Missing key",
    selectable: "Selectable models",
    noModels: "No model is currently usable.",
    managerDetail: "Manager detail",
  },
  fa: {
    title: "دسترسی ارائه‌دهنده هوش مصنوعی",
    active: "مدل فعال",
    configured: "پیکربندی‌شده",
    missing: "کلید موجود نیست",
    selectable: "مدل‌های قابل انتخاب",
    noModels: "هیچ مدلی در حال حاضر قابل استفاده نیست.",
    managerDetail: "جزئیات مدیر",
  },
  ar: {
    title: "وصول مزود الذكاء الاصطناعي",
    active: "النموذج النشط",
    configured: "مهيأ",
    missing: "المفتاح مفقود",
    selectable: "النماذج المتاحة",
    noModels: "لا يوجد نموذج قابل للاستخدام حالياً.",
    managerDetail: "تفاصيل المدير",
  },
};

type HealthTone = "good" | "warning" | "critical" | "neutral";

function localeForNumbers(locale: string) {
  if (locale === "fa") return "fa-IR";
  if (locale === "ar") return "ar";
  return "en-US";
}

function toneDotClasses(tone: HealthTone) {
  if (tone === "good") return "bg-emerald-400 shadow-[0_0_16px_rgba(52,211,153,0.45)]";
  if (tone === "warning") return "bg-amber-400 shadow-[0_0_16px_rgba(251,191,36,0.38)]";
  if (tone === "critical") return "bg-rose-400 shadow-[0_0_16px_rgba(244,63,94,0.42)]";
  return "bg-gray-300 dark:bg-white/25";
}

function parseStatusTone(rawStatus: string): HealthTone {
  const normalized = rawStatus.toLowerCase();
  if (normalized.includes("healthy") || normalized === "ok" || normalized === "connected") return "good";
  if (normalized.includes("degraded") || normalized.includes("timeout")) return "warning";
  if (normalized.includes("unhealthy") || normalized.includes("error") || normalized.includes("offline")) return "critical";
  return "neutral";
}

function getStatusCopy(rawStatus: string, t: (key: any, vars?: Record<string, string | number>) => string) {
  const normalized = rawStatus.toLowerCase();

  if (normalized.includes("healthy")) {
    const workerMatch = rawStatus.match(/\((\d+)\s+workers?\s+active\)/i);
    if (workerMatch) {
      return {
        title: t("monitoring.healthy") || "Healthy",
        detail: `${workerMatch[1]} active`,
      };
    }

    return {
      title: t("monitoring.healthy") || "Healthy",
      detail: t("monitoring.connected") || "Connected",
    };
  }

  if (normalized.includes("degraded: no active workers")) {
    return {
      title: t("monitoring.degraded") || "Degraded",
      detail: "No active workers",
    };
  }

  if (normalized.includes("degraded: timeout")) {
    return {
      title: t("monitoring.degraded") || "Degraded",
      detail: t("monitoring.timeout") || "Timeout",
    };
  }

  if (normalized.includes("unhealthy:")) {
    return {
      title: t("monitoring.down") || "Down",
      detail: rawStatus.split(":").slice(1).join(":").trim() || (t("monitoring.error") || "Error"),
    };
  }

  if (normalized === "unknown") {
    return {
      title: t("monitoring.statusUnknown") || "Unknown",
      detail: "Awaiting response",
    };
  }

  return {
    title: rawStatus,
    detail: t("monitoring.lastCheck") || "Last Check",
  };
}

function HealthCard({
  label,
  rawStatus,
  t,
}: {
  label: string;
  rawStatus: string;
  t: (key: any, vars?: Record<string, string | number>) => string;
}) {
  const tone = parseStatusTone(rawStatus);
  const { title, detail } = getStatusCopy(rawStatus, t);

  return (
    <article className="smx-panel min-w-0 p-4">
      <div className="mb-4 flex items-center justify-between gap-3">
        <p className="truncate text-[12px] font-medium text-ink-tertiary">{label}</p>
        <span className={clsx("h-3 w-3 shrink-0 rounded-full", toneDotClasses(tone))} aria-hidden />
      </div>
      <p
        className={clsx(
          "truncate text-[18px] font-semibold leading-tight tracking-normal",
          tone === "critical"
            ? "text-rose-600 dark:text-rose-300"
            : tone === "warning"
              ? "text-amber-700 dark:text-amber-300"
              : "text-ink"
        )}
      >
        {title}
      </p>
      <p className="mt-1.5 truncate text-[12px] font-medium text-ink-secondary">{detail}</p>
    </article>
  );
}

function MetricStat({
  icon,
  label,
  value,
  detail,
  progress,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
  progress?: { value: number; label: string };
}) {
  return (
    <article className="smx-panel min-w-0 p-4">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className="rounded-[12px] border border-black/5 bg-black/[0.02] p-2.5 text-brand dark:border-white/10 dark:bg-white/[0.04]">
          {icon}
        </div>
        <p className="truncate text-[12px] font-medium text-ink-tertiary">{label}</p>
      </div>
      <p className="truncate text-[22px] font-semibold leading-none tracking-normal text-ink tabular-nums" dir="ltr">
        {value}
      </p>
      <p className="mt-1.5 truncate text-[12px] font-medium text-ink-secondary">{detail}</p>
      {progress ? (
        <div className="mt-4">
          <ProgressBar value={progress.value} showLabel label={progress.label} />
        </div>
      ) : null}
    </article>
  );
}

function IncidentInbox({
  incidents,
  copy,
}: {
  incidents: Incident[];
  copy: typeof INCIDENT_COPY.en;
}) {
  return (
    <section className="smx-panel overflow-hidden">
      <div className="flex items-center justify-between gap-3 border-b border-black/5 px-4 py-3 dark:border-white/10">
        <h3 className="text-[14px] font-semibold text-ink">{copy.title}</h3>
        <span className="rounded-md border border-black/5 bg-black/[0.03] px-2 py-1 text-[11px] font-medium text-ink-secondary dark:border-white/10 dark:bg-white/[0.05]">
          {incidents.length} {copy.open}
        </span>
      </div>
      {incidents.length === 0 ? (
        <p className="px-4 py-6 text-[13px] text-ink-tertiary">{copy.empty}</p>
      ) : (
        <div className="divide-y divide-black/5 dark:divide-white/10">
          {incidents.map((incident) => {
            const isCritical = incident.severity === "critical";
            return (
              <article key={incident.id} className="px-4 py-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={clsx(
                      "inline-flex h-6 items-center gap-1.5 rounded-md px-2 text-[11px] font-semibold",
                      isCritical
                        ? "bg-rose-500/10 text-rose-700 dark:text-rose-300"
                        : "bg-amber-500/10 text-amber-700 dark:text-amber-300"
                    )}
                  >
                    <span className={clsx("h-1.5 w-1.5 rounded-full", isCritical ? "bg-rose-500" : "bg-amber-400")} aria-hidden />
                    {isCritical ? copy.critical : copy.warning}
                  </span>
                  <span className="text-[11px] font-medium uppercase tracking-normal text-ink-tertiary">
                    {incident.source}
                  </span>
                </div>
                <p className="mt-2 text-[13px] font-medium leading-5 text-ink">{incident.user_message}</p>
                <p className="mt-1 text-[12px] leading-5 text-ink-secondary">
                  {copy.detail}: {incident.manager_detail}
                </p>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}

function LlmProviderAccess({
  options,
  copy,
}: {
  options: LlmOptionsResponse | null;
  copy: typeof LLM_COPY.en;
}) {
  const providers = options?.providers ?? [];
  const selectableCount = options?.selectable_models.length ?? 0;

  return (
    <section className="smx-panel overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-black/5 px-4 py-3 dark:border-white/10">
        <div className="min-w-0">
          <h3 className="text-[14px] font-semibold text-ink">{copy.title}</h3>
          <p className="mt-1 truncate text-[12px] text-ink-secondary">
            {options ? `${copy.active}: ${options.active_model}` : copy.noModels}
          </p>
        </div>
        <span className="rounded-md border border-black/5 bg-black/[0.03] px-2 py-1 text-[11px] font-medium text-ink-secondary dark:border-white/10 dark:bg-white/[0.05]">
          {selectableCount} {copy.selectable}
        </span>
      </div>

      {options?.manager_detail ? (
        <p className="border-b border-black/5 px-4 py-3 text-[12px] leading-5 text-ink-secondary dark:border-white/10">
          {copy.managerDetail}: {options.manager_detail}
        </p>
      ) : null}

      {providers.length === 0 ? (
        <p className="px-4 py-6 text-[13px] text-ink-tertiary">{copy.noModels}</p>
      ) : (
        <div className="grid gap-0 divide-y divide-black/5 dark:divide-white/10">
          {providers.map((provider) => (
            <article key={provider.provider} className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
              <div className="min-w-0">
                <p className="text-[13px] font-semibold text-ink">{provider.label}</p>
                <p className="mt-1 truncate text-[12px] text-ink-secondary">
                  {provider.models.map((model) => model.label).join(", ") || provider.provider}
                </p>
              </div>
              <span
                className={clsx(
                  "inline-flex h-6 items-center rounded-md px-2 text-[11px] font-semibold",
                  provider.configured
                    ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                    : "bg-amber-500/10 text-amber-700 dark:text-amber-300"
                )}
              >
                {provider.configured ? copy.configured : copy.missing}
              </span>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

export function MonitoringPanel({ token }: MonitoringPanelProps) {
  const { t, locale } = useI18n();
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [performance, setPerformance] = useState<PerformancePayload | null>(null);
  const [incidentPayload, setIncidentPayload] = useState<IncidentPayload | null>(null);
  const [llmOptions, setLlmOptions] = useState<LlmOptionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastCheckTime, setLastCheckTime] = useState<string | null>(null);
  const refreshControllerRef = useRef<AbortController | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const [healthResult, performanceResult, incidentResult, llmResult] = await Promise.allSettled([
        apiRequest<HealthPayload>("/system/health", { token, signal }),
        apiRequest<PerformancePayload>("/system/performance", { token, signal }),
        apiRequest<IncidentPayload>("/system/incidents", { token, signal }),
        apiRequest<LlmOptionsResponse>("/system/llm/options", { token, signal }),
      ]);
      if (signal?.aborted) return;

      setHealth(healthResult.status === "fulfilled" ? healthResult.value : null);
      setPerformance(performanceResult.status === "fulfilled" ? performanceResult.value : null);
      setIncidentPayload(incidentResult.status === "fulfilled" ? incidentResult.value : null);
      setLlmOptions(llmResult.status === "fulfilled" ? llmResult.value : null);
      setLastCheckTime(new Date().toLocaleTimeString(localeForNumbers(locale)));
    } catch {
      if (signal?.aborted) return;
      setHealth(null);
      setPerformance(null);
      setIncidentPayload(null);
      setLlmOptions(null);
    } finally {
      if (signal?.aborted) return;
      setLoading(false);
    }
  }, [locale, token]);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  useEffect(() => {
    return () => refreshControllerRef.current?.abort();
  }, []);

  const refresh = useCallback(() => {
    refreshControllerRef.current?.abort();
    const controller = new AbortController();
    refreshControllerRef.current = controller;
    void load(controller.signal).finally(() => {
      if (refreshControllerRef.current === controller) {
        refreshControllerRef.current = null;
      }
    });
  }, [load]);

  const deps = health?.dependencies ?? {};
  const numberFormatter = useMemo(
    () => new Intl.NumberFormat(localeForNumbers(locale), { maximumFractionDigits: 0 }).format,
    [locale]
  );
  const decimalFormatter = useMemo(
    () =>
      new Intl.NumberFormat(localeForNumbers(locale), {
        maximumFractionDigits: 2,
        minimumFractionDigits: 2,
      }).format,
    [locale]
  );

  const depCards: Array<{ key: string; label: string; rawStatus: string }> = [
    { key: "api", label: t("monitoring.healthApi"), rawStatus: health?.status ?? "unknown" },
    { key: "database", label: t("monitoring.healthDb"), rawStatus: getDependencyStatus(deps.database) },
    { key: "redis", label: t("monitoring.healthRedis"), rawStatus: getDependencyStatus(deps.redis) },
    { key: "celery_workers", label: t("monitoring.healthCelery"), rawStatus: getDependencyStatus(deps.celery_workers) },
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
  const incidentCopy = INCIDENT_COPY[locale];
  const llmCopy = LLM_COPY[locale];
  const incidents = incidentPayload?.incidents ?? [];

  return (
    <section className="mx-auto flex min-h-full w-full max-w-[1120px] flex-col gap-4 py-1">
      <header className="flex min-h-[72px] items-end justify-between gap-4 border-b border-black/5 pb-4 dark:border-white/10">
        <div className="min-w-0">
          <p className="text-[12px] font-medium text-ink-tertiary">{t("monitoring.subtitle") || "Review backend service health"}</p>
          <h2 className="mt-1 text-[24px] font-semibold leading-tight tracking-normal text-ink">
            {t("monitoring.title") || "System Monitoring"}
          </h2>
        </div>

        <div className="smx-panel-subtle flex shrink-0 items-center gap-2 px-2 py-2">
          {lastCheckTime ? (
            <span className="px-2 text-[12px] font-medium text-ink-secondary">
              {t("monitoring.lastCheck") || "Last Check"}: {lastCheckTime}
            </span>
          ) : null}
          <Button variant="outlined" size="sm" onClick={refresh}>
            {t("common.refresh")}
          </Button>
        </div>
      </header>

      {loading ? (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {[1, 2, 3, 4].map((item) => (
              <div key={item} className="smx-panel h-[118px] animate-pulse" />
            ))}
          </div>
          <div className="grid gap-3 lg:grid-cols-3">
            {[1, 2, 3].map((item) => (
              <div key={item} className="smx-panel h-[154px] animate-pulse" />
            ))}
          </div>
        </>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {depCards.map((card) => (
              <HealthCard key={card.key} label={card.label} rawStatus={card.rawStatus} t={t} />
            ))}
          </div>

          <div className="grid gap-3 lg:grid-cols-3">
            <MetricStat
              icon={
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              }
              label={t("monitoring.dailyCost")}
              value={locale === "en" ? `$${todayCost.toFixed(2)}` : `${decimalFormatter(todayCost)} USD`}
              detail={`${t("monitoring.avgCost")} · ${locale === "en" ? `$${avgCostPerArticle.toFixed(3)}` : `${decimalFormatter(avgCostPerArticle)} USD`}`}
              progress={{
                value: Math.max(costPercent, todayCost > 0 ? 1.5 : 0),
                label: t("dashboard.ofCap", {
                  percent: numberFormatter(costPercent),
                  cap: numberFormatter(threshold),
                }),
              }}
            />

            <MetricStat
              icon={
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9.5a2.5 2.5 0 00-2.5-2.5H15M9 11l3 3m0 0l3-3m-3 3V8" />
                </svg>
              }
              label={t("monitoring.articlesToday")}
              value={numberFormatter(todayArticles)}
              detail={avgCostPerArticle > 0 ? `${t("monitoring.avgCost")} · ${locale === "en" ? `$${avgCostPerArticle.toFixed(3)}` : `${decimalFormatter(avgCostPerArticle)} USD`}` : t("dashboard.noUsageToday")}
            />

            <MetricStat
              icon={
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                </svg>
              }
              label={t("monitoring.connectionPool")}
              value={`${numberFormatter(poolUsed)}/${numberFormatter(poolSize)}`}
              detail={t("monitoring.utilized", { percent: numberFormatter(poolPercent) })}
              progress={{
                value: poolPercent,
                label: t("monitoring.utilized", { percent: numberFormatter(poolPercent) }),
              }}
            />
          </div>

          <LlmProviderAccess options={llmOptions} copy={llmCopy} />

          <IncidentInbox incidents={incidents} copy={incidentCopy} />

          {GRAFANA_URL ? (
            <section className="smx-panel overflow-hidden">
              <div className="border-b border-black/5 px-4 py-3 dark:border-white/10">
                <h3 className="text-[14px] font-semibold text-ink">{t("monitoring.grafana") || "Grafana Dashboard"}</h3>
              </div>
              <iframe
                src={GRAFANA_URL}
                title="Grafana Dashboard"
                className="h-[520px] w-full border-0 bg-transparent"
                loading="lazy"
              />
            </section>
          ) : (
            <section className="smx-panel flex items-center justify-between gap-4 px-4 py-3">
              <div className="min-w-0">
                <h3 className="text-[14px] font-semibold text-ink">{t("monitoring.grafana") || "Grafana Dashboard"}</h3>
                <p className="mt-1 text-[12px] text-ink-tertiary">
                  {t("monitoring.grafanaSetup") || "Grafana dashboard is not configured."}
                </p>
                <p className="mt-1 text-[12px] text-ink-secondary">
                  {(health?.version ? `v${health.version}` : t("monitoring.statusUnknown")) || "Unknown"}
                </p>
              </div>
              <span className="shrink-0 rounded-full border border-black/5 bg-black/[0.03] px-2.5 py-1 text-[11px] font-medium text-ink-secondary dark:border-white/10 dark:bg-white/[0.05]">
                {t("monitoring.offline") || "Offline"}
              </span>
            </section>
          )}
        </>
      )}
    </section>
  );
}

function getDependencyStatus(dep: string | { status?: string } | undefined): string {
  if (!dep) return "unknown";
  if (typeof dep === "string") return dep;
  return dep.status ?? "unknown";
}
