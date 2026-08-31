"use client";

import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { clsx } from "clsx";
import { apiRequest } from "@/lib/api";
import { formatModelDisplayName } from "@/lib/model-display";
import { IntegrationOperationalSummary, IntegrationOperationsResponse, LlmOptionsResponse } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import type { MessageKey } from "@/i18n/types";
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
    connection_pool?: { pool_size?: number; checked_out?: number; utilization_percent?: number };
    db_pool?: { pool_size?: number; checked_out?: number; utilization_percent?: number };
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
    workerTitle: "Generation jobs cannot start right now.",
    workerDetail: "No active Celery workers were detected.",
    workerType: "Processing worker",
    genericTitle: "An operational incident needs attention.",
    genericDetail: "Review the technical details for more information.",
    technicalDetails: "Technical details",
  },
  fa: {
    title: "صندوق رخدادها",
    empty: "رخداد باز وجود ندارد.",
    detail: "جزئیات مدیر",
    open: "باز",
    critical: "بحرانی",
    warning: "هشدار",
    workerTitle: "کارهای تولید محتوا فعلاً قابل شروع نیستند.",
    workerDetail: "هیچ پردازشگر Celery فعالی شناسایی نشد.",
    workerType: "پردازشگر",
    genericTitle: "یک رخداد عملیاتی نیازمند بررسی است.",
    genericDetail: "برای اطلاعات بیشتر، جزئیات فنی را بررسی کنید.",
    technicalDetails: "جزئیات فنی",
  },
  ar: {
    title: "صندوق الحوادث",
    empty: "لا توجد حوادث مفتوحة.",
    detail: "تفاصيل المدير",
    open: "مفتوح",
    critical: "حرج",
    warning: "تحذير",
    workerTitle: "لا يمكن بدء مهام إنشاء المحتوى حالياً.",
    workerDetail: "لم يتم العثور على أي عامل Celery نشط.",
    workerType: "عامل المعالجة",
    genericTitle: "توجد حادثة تشغيلية تحتاج إلى المراجعة.",
    genericDetail: "راجع التفاصيل التقنية لمزيد من المعلومات.",
    technicalDetails: "التفاصيل التقنية",
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
    activeProvider: "Active provider",
    providerInventory: "Provider inventory",
    managerDetail: "Manager detail",
    technicalDetails: "Technical details",
    timeout: "The AI provider health check timed out.",
  },
  fa: {
    title: "دسترسی ارائه‌دهنده هوش مصنوعی",
    active: "مدل فعال",
    configured: "پیکربندی‌شده",
    missing: "کلید موجود نیست",
    selectable: "مدل‌های قابل انتخاب",
    noModels: "هیچ مدلی در حال حاضر قابل استفاده نیست.",
    activeProvider: "ارائه‌دهنده فعال",
    providerInventory: "فهرست فنی ارائه‌دهندگان",
    managerDetail: "جزئیات مدیر",
    technicalDetails: "جزئیات فنی",
    timeout: "زمان بررسی سلامت ارائه‌دهنده هوش مصنوعی به پایان رسید.",
  },
  ar: {
    title: "حالة مزود الذكاء الاصطناعي",
    active: "النموذج النشط",
    configured: "مهيأ",
    missing: "المفتاح مفقود",
    selectable: "النماذج المتاحة",
    noModels: "لا يوجد نموذج قابل للاستخدام حالياً.",
    activeProvider: "المزود النشط",
    providerInventory: "قائمة المزودين التقنية",
    managerDetail: "تفاصيل المدير",
    technicalDetails: "التفاصيل التقنية",
    timeout: "انتهت مهلة فحص صحة مزود الذكاء الاصطناعي.",
  },
};

const INTEGRATION_COPY = {
  en: {
    title: "Integration Reliability",
    subtitle: "Durable publishing and Search Console synchronization signals",
    wordpress: "WordPress publishing",
    searchConsole: "Search Console sync",
    active: "Active",
    stale: "Stale",
    successRate: "Success rate",
    failures: "Recent failures",
    recommendations: "Recommended actions",
    empty: "No recent integration activity.",
    noWarnings: "No current reliability warning.",
    unavailable: "Integration reliability signals are temporarily unavailable.",
    healthy: "Healthy",
    idle: "Idle",
    warning: "Needs attention",
    critical: "Critical",
    degraded: "Degraded",
  },
  fa: {
    title: "پایداری یکپارچه‌سازی‌ها",
    subtitle: "وضعیت پایدار انتشار وردپرس و همگام‌سازی سرچ کنسول",
    wordpress: "انتشار وردپرس",
    searchConsole: "همگام‌سازی سرچ کنسول",
    active: "فعال",
    stale: "گیرکرده",
    successRate: "نرخ موفقیت",
    failures: "خطاهای اخیر",
    recommendations: "اقدام‌های پیشنهادی",
    empty: "فعالیت اخیر برای یکپارچه‌سازی‌ها وجود ندارد.",
    noWarnings: "هشدار پایداری فعالی وجود ندارد.",
    unavailable: "سیگنال‌های پایداری یکپارچه‌سازی موقتاً در دسترس نیستند.",
    healthy: "سالم",
    idle: "بدون فعالیت",
    warning: "نیازمند بررسی",
    critical: "بحرانی",
    degraded: "کاهش‌یافته",
  },
  ar: {
    title: "موثوقية التكاملات",
    subtitle: "إشارات النشر الدائم في WordPress ومزامنة Search Console",
    wordpress: "نشر WordPress",
    searchConsole: "مزامنة Search Console",
    active: "نشط",
    stale: "عالق",
    successRate: "نسبة النجاح",
    failures: "الإخفاقات الأخيرة",
    recommendations: "الإجراءات المقترحة",
    empty: "لا يوجد نشاط تكامل حديث.",
    noWarnings: "لا يوجد تحذير موثوقية نشط.",
    unavailable: "إشارات موثوقية التكامل غير متاحة مؤقتًا.",
    healthy: "سليم",
    idle: "خامل",
    warning: "يحتاج إلى انتباه",
    critical: "حرج",
    degraded: "متدهور",
  },
};

const INTEGRATION_REASON_COPY = {
  en: {
    stale_publish_attempts: "Stale publish attempts",
    stale_sync_runs: "Stale sync runs",
    high_failure_rate: "High failure rate",
    recent_failures: "Recent failures",
    connection_attention_required: "Connection needs attention",
    no_successful_sync: "No successful sync recorded",
    truncated_results: "Incomplete Search Console coverage",
  },
  fa: {
    stale_publish_attempts: "تلاش‌های انتشار گیرکرده",
    stale_sync_runs: "همگام‌سازی‌های گیرکرده",
    high_failure_rate: "نرخ خطای بالا",
    recent_failures: "خطاهای اخیر",
    connection_attention_required: "اتصال نیازمند بررسی است",
    no_successful_sync: "همگام‌سازی موفق ثبت نشده است",
    truncated_results: "پوشش ناقص داده‌های سرچ کنسول",
  },
  ar: {
    stale_publish_attempts: "محاولات نشر عالقة",
    stale_sync_runs: "عمليات مزامنة عالقة",
    high_failure_rate: "معدل فشل مرتفع",
    recent_failures: "إخفاقات حديثة",
    connection_attention_required: "الاتصال يحتاج إلى مراجعة",
    no_successful_sync: "لم تُسجّل مزامنة ناجحة",
    truncated_results: "تغطية Search Console غير مكتملة",
  },
};

const INTEGRATION_ACTION_COPY = {
  en: {
    run_wordpress_reconciliation: "Run WordPress reconciliation and verify the integrations worker before new public publishing.",
    review_wordpress_failures: "Review recent WordPress failures and keep public publishing approval-gated.",
    run_search_console_reconciliation: "Run Search Console reconciliation and verify the integrations worker.",
    reconnect_search_console: "Reconnect affected Search Console properties before relying on SEO recommendations.",
    run_initial_search_console_sync: "Run and verify the first completed Search Console synchronization window.",
    review_truncated_syncs: "Review incomplete sync windows before making portfolio decisions.",
  },
  fa: {
    run_wordpress_reconciliation: "بازیابی وردپرس را اجرا و پردازشگر یکپارچه‌سازی را پیش از انتشار عمومی بررسی کنید.",
    review_wordpress_failures: "خطاهای اخیر وردپرس را بررسی و انتشار عمومی را همچنان نیازمند تأیید نگه دارید.",
    run_search_console_reconciliation: "بازیابی سرچ کنسول را اجرا و پردازشگر یکپارچه‌سازی را بررسی کنید.",
    reconnect_search_console: "پیش از اتکا به پیشنهادهای سئو، اتصال‌های مشکل‌دار سرچ کنسول را دوباره برقرار کنید.",
    run_initial_search_console_sync: "نخستین بازه کامل همگام‌سازی سرچ کنسول را اجرا و صحت آن را بررسی کنید.",
    review_truncated_syncs: "پیش از تصمیم‌گیری، بازه‌های همگام‌سازی ناقص را بررسی کنید.",
  },
  ar: {
    run_wordpress_reconciliation: "شغّل تسوية WordPress وتحقق من عامل التكامل قبل أي نشر عام جديد.",
    review_wordpress_failures: "راجع إخفاقات WordPress الأخيرة وأبقِ النشر العام خاضعًا للموافقة.",
    run_search_console_reconciliation: "شغّل تسوية Search Console وتحقق من عامل التكامل.",
    reconnect_search_console: "أعد ربط مواقع Search Console المتأثرة قبل الاعتماد على توصيات SEO.",
    run_initial_search_console_sync: "شغّل أول نافذة مكتملة لمزامنة Search Console وتحقق منها.",
    review_truncated_syncs: "راجع نوافذ المزامنة غير المكتملة قبل اتخاذ قرارات المحتوى.",
  },
};

const HEALTH_DETAIL_COPY = {
  en: {
    noWorkers: "No active processing workers",
    awaiting: "Awaiting response",
    activeWorkers: (count: string) => `${count} active`,
  },
  fa: {
    noWorkers: "هیچ پردازشگر فعالی وجود ندارد",
    awaiting: "در انتظار پاسخ",
    activeWorkers: (count: string) => `${count} پردازشگر فعال`,
  },
  ar: {
    noWorkers: "لا يوجد عامل معالجة نشط",
    awaiting: "في انتظار الاستجابة",
    activeWorkers: (count: string) => `${count} عامل نشط`,
  },
};

type HealthTone = "good" | "warning" | "critical" | "neutral";
type HealthStatusCopy = { title: string; detail: string; technicalDetail?: string };

function localeForNumbers(locale: string) {
  if (locale === "fa") return "fa-IR";
  if (locale === "ar") return "ar";
  return "en-US";
}

function toneDotClasses(tone: HealthTone) {
  if (tone === "good") return "bg-success";
  if (tone === "warning") return "bg-warning";
  if (tone === "critical") return "bg-danger";
  return "bg-ink-muted/35";
}

function parseStatusTone(rawStatus: string): HealthTone {
  const normalized = rawStatus.toLowerCase();
  if (normalized.includes("healthy") || normalized === "ok" || normalized === "connected") return "good";
  if (normalized.includes("degraded") || normalized.includes("timeout")) return "warning";
  if (normalized.includes("unhealthy") || normalized.includes("error") || normalized.includes("offline")) return "critical";
  return "neutral";
}

function getStatusCopy(rawStatus: string, locale: keyof typeof HEALTH_DETAIL_COPY, t: (key: MessageKey, vars?: Record<string, string | number>) => string): HealthStatusCopy {
  const normalized = rawStatus.toLowerCase();
  const detailCopy = HEALTH_DETAIL_COPY[locale];
  const activeWorkerMatch = rawStatus.match(/(\d+)\s+workers?\(s\)\s+are available|(\d+)\s+workers?\s+are available|(\d+)\s+workers?\s+active/i);
  const activeWorkerCount = activeWorkerMatch?.[1] ?? activeWorkerMatch?.[2] ?? activeWorkerMatch?.[3];

  if (normalized.includes("healthy")) {
    if (activeWorkerCount) {
      return {
        title: t("monitoring.healthy"),
        detail: detailCopy.activeWorkers(activeWorkerCount),
      };
    }

    return {
      title: t("monitoring.healthy"),
      detail: t("monitoring.connected"),
    };
  }

  if (activeWorkerCount) {
    return {
      title: t("monitoring.healthy"),
      detail: detailCopy.activeWorkers(activeWorkerCount),
    };
  }

  if (normalized.includes("degraded: no active workers")) {
    return {
      title: t("monitoring.degraded"),
      detail: detailCopy.noWorkers,
    };
  }

  if (normalized.includes("degraded: timeout")) {
    return {
      title: t("monitoring.degraded"),
      detail: t("monitoring.timeout"),
    };
  }

  if (normalized === "degraded" || normalized.includes("degraded")) {
    return {
      title: t("monitoring.degraded"),
      detail: t("monitoring.lastCheck"),
    };
  }

  if (normalized.includes("unhealthy:")) {
    return {
      title: t("monitoring.down"),
      detail: t("monitoring.error"),
      technicalDetail: rawStatus.split(":").slice(1).join(":").trim(),
    };
  }

  if (normalized === "unknown") {
    return {
      title: t("monitoring.statusUnknown"),
      detail: detailCopy.awaiting,
    };
  }

  return {
    title: t("monitoring.statusUnknown"),
    detail: t("monitoring.lastCheck"),
    technicalDetail: rawStatus,
  };
}

function HealthCard({
  label,
  rawStatus,
  locale,
  t,
}: {
  label: string;
  rawStatus: string;
  locale: keyof typeof HEALTH_DETAIL_COPY;
  t: (key: MessageKey, vars?: Record<string, string | number>) => string;
}) {
  const tone = parseStatusTone(rawStatus);
  const { title, detail, technicalDetail } = getStatusCopy(rawStatus, locale, t);

  return (
    <article className="min-w-0 border-t border-line py-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <p className="truncate text-xs font-medium text-ink-tertiary">{label}</p>
        <span className={clsx("h-3 w-3 shrink-0 rounded-full", toneDotClasses(tone))} aria-hidden />
      </div>
      <p
        className={clsx(
          "truncate text-body-lg font-semibold leading-tight tracking-normal",
          tone === "critical"
            ? "text-danger"
            : tone === "warning"
              ? "text-warning"
              : "text-ink"
        )}
      >
        {title}
      </p>
      <p className="mt-1.5 truncate text-xs font-medium text-ink-secondary">{detail}</p>
      {technicalDetail ? (
        <details className="mt-2 text-xs leading-4 text-ink-tertiary">
          <summary className="cursor-pointer">{locale === "fa" ? "جزئیات فنی" : locale === "ar" ? "التفاصيل التقنية" : "Technical details"}</summary>
          <p className="mt-1 break-words" dir="auto">{technicalDetail}</p>
        </details>
      ) : null}
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
    <article className="min-w-0 border-t border-line py-3">
      <div className="mb-2 flex items-start justify-between gap-3">
        <div className="text-brand">
          {icon}
        </div>
        <p className="truncate text-xs font-medium text-ink-tertiary">{label}</p>
      </div>
      <p className="truncate text-xl font-semibold leading-none tracking-normal text-ink tabular-nums" dir="ltr">
        {value}
      </p>
      <p className="mt-1.5 truncate text-xs font-medium text-ink-secondary">{detail}</p>
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
  locale,
}: {
  incidents: Incident[];
  copy: typeof INCIDENT_COPY.en;
  locale: keyof typeof INCIDENT_COPY;
}) {
  return (
    <section className="overflow-hidden border-t border-line">
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
        <h3 className="text-base font-semibold text-ink">{copy.title}</h3>
        <span className="text-xs font-medium text-ink-muted">
          {incidents.length} {copy.open}
        </span>
      </div>
      {incidents.length === 0 ? (
        <p className="px-4 py-6 text-sm text-ink-tertiary">{copy.empty}</p>
      ) : (
        <div className="divide-y divide-line">
          {incidents.map((incident) => {
            const isCritical = incident.severity === "critical";
            const rawIncidentText = `${incident.source} ${incident.user_message} ${incident.manager_detail}`;
            const isWorkerIncident = /worker|celery|generation jobs cannot start/i.test(rawIncidentText);
            const localizedIncident = locale === "en"
              ? {
                  source: incident.source,
                  title: incident.user_message,
                  detail: incident.manager_detail,
                  technicalDetail: null,
                }
              : isWorkerIncident
                ? {
                    source: copy.workerType,
                    title: copy.workerTitle,
                    detail: copy.workerDetail,
                    technicalDetail: null,
                  }
                : {
                    source: copy.warning,
                    title: copy.genericTitle,
                    detail: copy.genericDetail,
                    technicalDetail: rawIncidentText,
                  };
            return (
              <article key={incident.id} className="px-4 py-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={clsx(
                      "inline-flex h-6 items-center gap-1.5 rounded-md px-2 text-xs font-semibold",
                      isCritical
                        ? "bg-danger-subtle text-danger"
                        : "bg-warning/10 text-warning"
                    )}
                  >
                    <span className={clsx("h-1.5 w-1.5 rounded-full", isCritical ? "bg-danger" : "bg-warning")} aria-hidden />
                    {isCritical ? copy.critical : copy.warning}
                  </span>
                  <span className="text-xs font-medium uppercase tracking-normal text-ink-tertiary">
                    {localizedIncident.source}
                  </span>
                </div>
                <p className="mt-2 text-sm font-medium leading-5 text-ink">{localizedIncident.title}</p>
                <p className="mt-1 text-xs leading-5 text-ink-secondary">
                  {copy.detail}: {localizedIncident.detail}
                </p>
                {localizedIncident.technicalDetail ? (
                  <details className="mt-2 text-xs leading-5 text-ink-tertiary">
                    <summary className="cursor-pointer">{copy.technicalDetails}</summary>
                    <p className="mt-1 break-words" dir="ltr">{localizedIncident.technicalDetail}</p>
                  </details>
                ) : null}
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
  const providers = Array.isArray(options?.providers) ? options.providers : [];
  const selectableModels = Array.isArray(options?.selectable_models) ? options.selectable_models : [];
  const selectableCount = selectableModels.length;
  const activeProvider = providers.find(
    (provider) => provider.active || provider.provider === options?.active_provider
  );
  const managerDetail = options?.manager_detail?.toLowerCase().includes("llm ping timed out")
    ? copy.timeout
    : options?.manager_detail;

  return (
    <section className="overflow-hidden border-t border-line">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-ink">{copy.title}</h3>
          <p className="mt-1 truncate text-xs text-ink-secondary">
            {options ? `${copy.active}: ${formatModelDisplayName(options.active_model)}` : copy.noModels}
          </p>
        </div>
        <span className="text-xs font-medium text-ink-muted">
          {selectableCount} {copy.selectable}
        </span>
      </div>

      {managerDetail ? (
        <details className="border-b border-line px-4 py-3 text-xs leading-5 text-ink-secondary">
          <summary className="cursor-pointer font-medium text-ink-secondary">{copy.technicalDetails}</summary>
          <p className="mt-2" dir="auto">{copy.managerDetail}: {managerDetail}</p>
        </details>
      ) : null}

      {!activeProvider ? (
        <p className="px-4 py-6 text-sm text-ink-tertiary">{copy.noModels}</p>
      ) : (
        <article className="flex flex-wrap items-center justify-between gap-3 px-4 py-4">
          <div className="min-w-0">
            <p className="text-sm font-semibold text-ink">{activeProvider.label}</p>
            <p className="mt-1 truncate text-xs text-ink-secondary">
              {formatModelDisplayName(options?.active_model)}
            </p>
          </div>
          <span className="inline-flex h-6 items-center rounded-md bg-success/10 px-2 text-xs font-semibold text-success">
            {copy.activeProvider}
          </span>
        </article>
      )}

      {providers.length > 0 ? (
        <details className="border-t border-line">
          <summary className="cursor-pointer px-4 py-3 text-xs font-medium text-ink-secondary">
            {copy.providerInventory}
          </summary>
          <div className="divide-y divide-line border-t border-line">
            {providers.map((provider) => {
              const models = Array.isArray(provider.models) ? provider.models : [];

              return (
                <article key={provider.provider} className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-ink">{provider.label}</p>
                    <p className="mt-1 truncate text-xs text-ink-secondary">
                      {models.map((model) => formatModelDisplayName(model.model)).join(", ") || provider.label}
                    </p>
                  </div>
                  <span
                    className={clsx(
                      "inline-flex h-6 items-center rounded-md px-2 text-xs font-semibold",
                      provider.configured
                        ? "bg-success/10 text-success"
                        : "bg-warning/10 text-warning"
                    )}
                  >
                    {provider.configured ? copy.configured : copy.missing}
                  </span>
                </article>
              );
            })}
          </div>
        </details>
      ) : null}
    </section>
  );
}

function integrationTone(status: string) {
  if (status === "healthy") return "border-success/20 bg-success/[0.06] text-success";
  if (status === "critical") return "border-danger/20 bg-danger-subtle text-danger";
  if (status === "warning" || status === "degraded") return "border-warning/20 bg-warning/[0.07] text-warning";
  return "border-line bg-ink/[0.035] text-ink-secondary";
}

function IntegrationSummaryCard({
  label,
  summary,
  copy,
  locale,
}: {
  label: string;
  summary: IntegrationOperationalSummary;
  copy: typeof INTEGRATION_COPY.en;
  locale: string;
}) {
  const successRate = summary.recent_total > 0
    ? Math.max(0, Math.min(100, (summary.recent_succeeded / summary.recent_total) * 100))
    : null;
  const statusLabel = copy[summary.status as keyof typeof copy] ?? summary.status;
  return (
    <article className="rounded-lg border border-line bg-ink/[0.018] p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h4 className="text-sm font-semibold text-ink">{label}</h4>
          <p className="mt-1 text-xs text-ink-tertiary">
            {summary.reasons.length
              ? summary.reasons.map((reason) => INTEGRATION_REASON_COPY[locale as keyof typeof INTEGRATION_REASON_COPY]?.[reason as keyof typeof INTEGRATION_REASON_COPY.en] ?? reason).join(" · ")
              : summary.active_count > 0 || summary.recent_total > 0
                ? copy.noWarnings
                : copy.empty}
          </p>
        </div>
        <span className={clsx("rounded-full border px-2.5 py-1 text-xs font-semibold", integrationTone(summary.status))}>
          {statusLabel}
        </span>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-2">
        <div>
          <p className="text-xs text-ink-tertiary">{copy.active}</p>
          <p className="mt-1 text-xl font-semibold text-ink">{summary.active_count}</p>
        </div>
        <div>
          <p className="text-xs text-ink-tertiary">{copy.stale}</p>
          <p className={clsx("mt-1 text-xl font-semibold", summary.stale_count ? "text-danger" : "text-ink")}>
            {summary.stale_count}
          </p>
        </div>
        <div>
          <p className="text-xs text-ink-tertiary">{copy.successRate}</p>
          <p className="mt-1 text-xl font-semibold text-ink">
            {successRate === null
              ? "—"
              : `${new Intl.NumberFormat(localeForNumbers(locale), { maximumFractionDigits: 0 }).format(successRate)}%`}
          </p>
        </div>
      </div>
      {summary.recent_failures.length > 0 ? (
        <details className="mt-4 border-t border-line pt-3 text-xs">
          <summary className="cursor-pointer font-medium text-ink-secondary">
            {copy.failures} ({summary.recent_failures.length})
          </summary>
          <div className="mt-2 space-y-2">
            {summary.recent_failures.slice(0, 3).map((failure) => (
              <p key={failure.id} className="break-words rounded-lg bg-ink/[0.025] px-3 py-2 leading-5 text-ink-secondary" dir="auto">
                <span className="font-semibold text-ink">{failure.error_category}</span>: {failure.error_message}
              </p>
            ))}
          </div>
        </details>
      ) : null}
    </article>
  );
}

function IntegrationOperationsBoard({
  payload,
  locale,
}: {
  payload: IntegrationOperationsResponse | null;
  locale: keyof typeof INTEGRATION_COPY;
}) {
  if (!payload) return null;
  const copy = INTEGRATION_COPY[locale] ?? INTEGRATION_COPY.en;
  return (
    <section className="overflow-hidden border-t border-line" aria-live="polite">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-line px-4 py-3">
        <div>
          <h3 className="text-base font-semibold text-ink">{copy.title}</h3>
          <p className="mt-1 text-xs text-ink-secondary">{copy.subtitle}</p>
        </div>
        <span className={clsx("rounded-full border px-2.5 py-1 text-xs font-semibold", integrationTone(payload.overall_status))}>
          {copy[payload.overall_status as keyof typeof copy] ?? payload.overall_status}
        </span>
      </div>
      <div className="grid gap-3 p-4 lg:grid-cols-2">
        <IntegrationSummaryCard label={copy.wordpress} summary={payload.integrations.wordpress} copy={copy} locale={locale} />
        <IntegrationSummaryCard label={copy.searchConsole} summary={payload.integrations.search_console} copy={copy} locale={locale} />
      </div>
      {payload.recommendations.length > 0 ? (
        <div className="border-t border-line px-4 py-3">
          <p className="text-xs font-semibold text-ink">{copy.recommendations}</p>
          <div className="mt-2 space-y-2">
            {payload.recommendations.map((recommendation) => (
              <div key={`${recommendation.integration}-${recommendation.code}`} className="flex gap-2 text-xs leading-5 text-ink-secondary">
                <span className={clsx("mt-1.5 h-2 w-2 shrink-0 rounded-full", recommendation.priority === "critical" ? "bg-danger" : recommendation.priority === "high" ? "bg-warning" : "bg-brand")} aria-hidden />
                <p>{INTEGRATION_ACTION_COPY[locale]?.[recommendation.code as keyof typeof INTEGRATION_ACTION_COPY.en] ?? recommendation.message}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

export function MonitoringPanel({ token }: MonitoringPanelProps) {
  const { t, locale } = useI18n();
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [performance, setPerformance] = useState<PerformancePayload | null>(null);
  const [incidentPayload, setIncidentPayload] = useState<IncidentPayload | null>(null);
  const [llmOptions, setLlmOptions] = useState<LlmOptionsResponse | null>(null);
  const [integrationOperations, setIntegrationOperations] = useState<IntegrationOperationsResponse | null>(null);
  const [integrationOperationsUnavailable, setIntegrationOperationsUnavailable] = useState(false);
  const [loading, setLoading] = useState(true);
  const [lastCheckTime, setLastCheckTime] = useState<string | null>(null);
  const refreshControllerRef = useRef<AbortController | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const [healthResult, performanceResult, incidentResult, llmResult, integrationResult] = await Promise.allSettled([
        apiRequest<HealthPayload>("/system/health", { token, signal }),
        apiRequest<PerformancePayload>("/system/performance", { token, signal }),
        apiRequest<IncidentPayload>("/system/incidents", { token, signal }),
        apiRequest<LlmOptionsResponse>("/system/llm/options", { token, signal }),
        apiRequest<IntegrationOperationsResponse>("/system/integrations/operations", { token, signal }),
      ]);
      if (signal?.aborted) return;

      setHealth(healthResult.status === "fulfilled" ? healthResult.value : null);
      setPerformance(performanceResult.status === "fulfilled" ? performanceResult.value : null);
      setIncidentPayload(incidentResult.status === "fulfilled" ? incidentResult.value : null);
      setLlmOptions(llmResult.status === "fulfilled" ? llmResult.value : null);
      setIntegrationOperations(integrationResult.status === "fulfilled" ? integrationResult.value : null);
      setIntegrationOperationsUnavailable(integrationResult.status === "rejected");
      setLastCheckTime(new Date().toLocaleTimeString(localeForNumbers(locale)));
    } catch {
      if (signal?.aborted) return;
      setHealth(null);
      setPerformance(null);
      setIncidentPayload(null);
      setLlmOptions(null);
      setIntegrationOperations(null);
      setIntegrationOperationsUnavailable(true);
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

  const pool = performance?.metrics?.connection_pool ?? performance?.metrics?.db_pool;
  const poolSize = pool?.pool_size ?? 0;
  const poolUsed = pool?.checked_out ?? 0;
  const poolPercent =
    typeof pool?.utilization_percent === "number"
      ? Math.min(100, Math.max(0, pool.utilization_percent))
      : poolSize > 0
        ? Math.min(100, (poolUsed / poolSize) * 100)
        : 0;
  const incidentCopy = INCIDENT_COPY[locale] ?? INCIDENT_COPY.en;
  const llmCopy = LLM_COPY[locale] ?? LLM_COPY.en;
  const incidents = Array.isArray(incidentPayload?.incidents) ? incidentPayload.incidents : [];

  return (
    <section className="smx-page flex min-h-full flex-col gap-4">
      <header className="smx-page-header">
        <div className="min-w-0">
          <p className="text-xs font-medium text-ink-tertiary">{t("monitoring.subtitle")}</p>
          <h2 className="smx-page-title">
            {t("monitoring.title")}
          </h2>
        </div>

        <div className="smx-toolbar shrink-0">
          {lastCheckTime ? (
            <span className="px-2 text-xs font-medium text-ink-secondary">
              {t("monitoring.lastCheck")}: {lastCheckTime}
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
              <div key={item} className="h-[90px] animate-pulse border-t border-line bg-ink/[0.02]" />
            ))}
          </div>
          <div className="grid gap-3 lg:grid-cols-3">
            {[1, 2, 3].map((item) => (
              <div key={item} className="h-[120px] animate-pulse border-t border-line bg-ink/[0.02]" />
            ))}
          </div>
        </>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {depCards.map((card) => (
              <HealthCard key={card.key} label={card.label} rawStatus={card.rawStatus} locale={locale} t={t} />
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

          <IntegrationOperationsBoard payload={integrationOperations} locale={locale} />
          {integrationOperationsUnavailable ? (
            <section
              className="border-s-2 border-s-warning px-4 py-3 text-sm text-warning"
              role="status"
            >
              {(INTEGRATION_COPY[locale] ?? INTEGRATION_COPY.en).unavailable}
            </section>
          ) : null}

          <IncidentInbox incidents={incidents} copy={incidentCopy} locale={locale} />

          {GRAFANA_URL ? (
            <section className="overflow-hidden border-t border-line">
              <div className="border-b border-line px-4 py-3">
                <h3 className="text-base font-semibold text-ink">{t("monitoring.grafana")}</h3>
              </div>
              <iframe
                src={GRAFANA_URL}
                title={t("monitoring.grafana")}
                className="h-[520px] w-full border-0 bg-transparent"
                loading="lazy"
              />
            </section>
          ) : (
            <section className="flex items-center justify-between gap-4 border-t border-line py-4">
              <div className="min-w-0">
                <h3 className="text-base font-semibold text-ink">{t("monitoring.grafana")}</h3>
                <p className="mt-1 text-xs text-ink-tertiary">
                  {t("monitoring.grafanaSetup")}
                </p>
                <p className="mt-1 text-xs text-ink-secondary">
                  {health?.version ? `v${health.version}` : t("monitoring.statusUnknown")}
                </p>
              </div>
              <span className="shrink-0 text-xs font-medium text-ink-muted">
                {t("monitoring.offline")}
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
