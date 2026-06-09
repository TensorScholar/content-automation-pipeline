"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import { TaskHistoryItem, TaskStatusResponse, ArticleDetail, DraftRiskAssessment } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { Modal } from "@/components/ui/modal";

interface TasksPanelProps {
  token: string;
}

interface QualityMetricsResponse {
  seo_score?: {
    score?: number;
    recommendations?: string[];
    component_scores?: Record<string, number>;
  };
  structure_score?: {
    score?: number;
    details?: Record<string, unknown>;
  };
  readability_grade?: string;
  overall_quality?: {
    score?: number;
    grade?: string;
  };
}

type FilterTab = "all" | "SUCCESS" | "FAILURE" | "RUNNING";
type DetailTab = "content" | "seo" | "export";
type ContentView = "reader" | "raw" | "edit";

const RISK_COPY = {
  en: {
    title: "Draft Risk",
    low: "Low risk",
    medium: "Needs review",
    high: "High risk",
    blocked: "Blocked",
    loading: "Checking publish risk...",
    blockedPublish: "Publishing is blocked until the critical issue is fixed.",
  },
  fa: {
    title: "ریسک پیش‌نویس",
    low: "ریسک پایین",
    medium: "نیازمند بررسی",
    high: "ریسک بالا",
    blocked: "مسدود",
    loading: "در حال بررسی ریسک انتشار...",
    blockedPublish: "تا رفع مورد بحرانی، انتشار مسدود است.",
  },
  ar: {
    title: "مخاطر المسودة",
    low: "مخاطر منخفضة",
    medium: "يحتاج مراجعة",
    high: "مخاطر عالية",
    blocked: "محظور",
    loading: "جارٍ فحص مخاطر النشر...",
    blockedPublish: "النشر محظور حتى إصلاح المشكلة الحرجة.",
  },
};

export function TasksPanel({ token }: TasksPanelProps) {
  const { t, locale } = useI18n();
  const { showToast } = useToast();
  const [tasks, setTasks] = useState<TaskHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterTab>("all");
  const [search, setSearch] = useState("");
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [liveStatus, setLiveStatus] = useState<TaskStatusResponse | null>(null);
  const [streamActive, setStreamActive] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // deep view state
  const [detailArticle, setDetailArticle] = useState<ArticleDetail | null>(null);
  const [detailTab, setDetailTab] = useState<DetailTab>("content");
  const [contentView, setContentView] = useState<ContentView>("reader");
  const [editContent, setEditContent] = useState("");
  const [wpPublishing, setWpPublishing] = useState(false);
  const [wpResult, setWpResult] = useState<string | null>(null);
  const [riskAssessment, setRiskAssessment] = useState<DraftRiskAssessment | null>(null);
  const [riskLoading, setRiskLoading] = useState(false);
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetricsResponse | null>(null);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [qualityError, setQualityError] = useState<string | null>(null);

  const loadTasks = useCallback(async (signal?: AbortSignal) => {
    try {
      const res = await apiRequest<TaskHistoryItem[]>("/content/tasks", { token, signal });
      if (signal?.aborted) return;
      setTasks(Array.isArray(res) ? res : []);
    } catch {
      if (signal?.aborted) return;
      setTasks([]);
    } finally {
      if (signal?.aborted) return;
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    const controller = new AbortController();
    void loadTasks(controller.signal);
    return () => controller.abort();
  }, [loadTasks]);

  // Auto-refresh without overlapping requests.
  // FIX: Prevent overlapping polls
  useEffect(() => {
    if (!autoRefresh) return;
    const controller = new AbortController();
    let mounted = true;
    let isPolling = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const schedule = () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      timeoutId = setTimeout(() => { void poll(); }, 30000);
    };

    const poll = async () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      isPolling = true;
      try {
        await loadTasks(controller.signal);
      } catch (error) {
        if (!controller.signal.aborted) {
          console.error("Task polling error:", error);
        }
      } finally {
        isPolling = false;
      }
      if (mounted && !controller.signal.aborted) {
        schedule();
      }
    };

    schedule();
    return () => {
      mounted = false;
      isPolling = false;
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
      controller.abort();
    };
  }, [autoRefresh, loadTasks]);

  // Poll selected task without exposing bearer tokens in URLs.
  // FIX: Prevent overlapping polls by tracking polling state with ref
  useEffect(() => {
    if (!selectedTaskId) {
      setStreamActive(false);
      setLiveStatus(null);
      return;
    }

    const controller = new AbortController();
    let mounted = true;
    let isPolling = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    setStreamActive(true);

    const schedule = () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      timeoutId = setTimeout(() => { void poll(); }, 4000);
    };

    const poll = async () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      isPolling = true;

      try {
        const payload = await apiRequest<TaskStatusResponse>(`/content/task/${selectedTaskId}`, {
          token,
          signal: controller.signal,
          timeoutMs: 8000,
        });
        if (!mounted || controller.signal.aborted) return;
        setLiveStatus(payload);
        if (payload.ready) {
          setStreamActive(false);
          isPolling = false;
          void loadTasks(controller.signal);
          return;
        }
        isPolling = false;
        if (mounted && !controller.signal.aborted) {
          schedule();
        }
      } catch (error) {
        if (!mounted || controller.signal.aborted) return;
        setStreamActive(false);
        isPolling = false;
      }
    };

    void poll();

    return () => {
      mounted = false;
      isPolling = false;
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
      controller.abort();
    };
  }, [selectedTaskId, token, loadTasks]);

  // Load article detail when task is SUCCESS
  useEffect(() => {
    const articleId = liveStatus?.result?.article_id;
    if (!articleId) {
      setDetailArticle(null);
      setRiskAssessment(null);
      setRiskLoading(false);
      setQualityMetrics(null);
      setQualityLoading(false);
      setQualityError(null);
      return;
    }
    const controller = new AbortController();
    const load = async () => {
      setRiskLoading(true);
      try {
        const [articleResult, riskResult] = await Promise.allSettled([
          apiRequest<ArticleDetail>(`/content/${articleId}`, {
            token,
            signal: controller.signal,
          }),
          apiRequest<DraftRiskAssessment>(`/content/${articleId}/risk-assessment`, {
            token,
            signal: controller.signal,
          }),
        ]);
        if (controller.signal.aborted) return;
        if (articleResult.status === "fulfilled") {
          setDetailArticle(articleResult.value);
          setEditContent(articleResult.value.content ?? "");
          setQualityMetrics(null);
          setQualityError(null);
        } else {
          setDetailArticle(null);
        }
        setRiskAssessment(riskResult.status === "fulfilled" ? riskResult.value : null);
      } catch {
        if (!controller.signal.aborted) {
          setDetailArticle(null);
          setRiskAssessment(null);
        }
      } finally {
        if (!controller.signal.aborted) setRiskLoading(false);
      }
    };
    void load();
    return () => controller.abort();
  }, [liveStatus?.result?.article_id, token]);

  useEffect(() => {
    const articleId = detailArticle?.id;
    if (detailTab !== "seo" || !articleId || qualityMetrics || qualityLoading) return;

    const controller = new AbortController();
    setQualityLoading(true);
    setQualityError(null);

    apiRequest<QualityMetricsResponse>(`/content/${articleId}/quality`, {
      token,
      signal: controller.signal,
      timeoutMs: 20000,
    })
      .then((payload) => {
        if (!controller.signal.aborted) setQualityMetrics(payload);
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setQualityMetrics(null);
          setQualityError(error instanceof ApiError ? error.detail : t("common.unexpectedError"));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setQualityLoading(false);
      });

    return () => controller.abort();
  }, [detailArticle?.id, detailTab, qualityLoading, qualityMetrics, t, token]);

  // KPI counters
  const kpis = useMemo(() => {
    const total = tasks.length;
    const success = tasks.filter((t) => t.status?.toUpperCase() === "SUCCESS").length;
    const failure = tasks.filter((t) => ["FAILURE", "FAILED"].includes(t.status?.toUpperCase() ?? "")).length;
    const running = total - success - failure;
    return { total, success, failure, running };
  }, [tasks]);

  const filtered = useMemo(() => {
    let list = tasks;
    if (filter === "RUNNING") {
      list = list.filter((t) => !["SUCCESS", "FAILURE", "FAILED"].includes(t.status?.toUpperCase() ?? ""));
    } else if (filter !== "all") {
      list = list.filter((t) => t.status?.toUpperCase() === filter || (filter === "FAILURE" && t.status?.toUpperCase() === "FAILED"));
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter((t) => t.task_id.toLowerCase().includes(q) || (t.topic ?? "").toLowerCase().includes(q));
    }
    return list;
  }, [tasks, filter, search]);

  const onDeleteTask = async (taskId: string) => {
    setDeleteConfirmId(null);
    try {
      await apiRequest<void>(`/content/task/${taskId}`, { method: "DELETE", token });
      showToast("success", t("tasks.taskDeleted") || "Task deleted");
      if (selectedTaskId === taskId) { setSelectedTaskId(null); setLiveStatus(null); }
      await loadTasks();
    } catch (e) {
      showToast("error", e instanceof ApiError ? e.detail : (t("common.unexpectedError") || "Unexpected error"));
    }
  };

  const onBulkDownload = async () => {
    const successful = tasks.filter((t) => t.status?.toUpperCase() === "SUCCESS").slice(0, 20);
    const results: string[] = [];
    for (const task of successful) {
      const articleId = (task.result as Record<string, unknown> | undefined)?.article_id;
      if (!articleId) continue;
      try {
        const article = await apiRequest<ArticleDetail>(`/content/${String(articleId)}`, { token });
        results.push(`--- ${article.title} ---\n\n${article.content}\n\n`);
      } catch { /* skip */ }
    }
    if (results.length === 0) return;
    const blob = new Blob([results.join("\n\n")], { type: "text/plain;charset=utf-8" });
    downloadBlob(blob, "articles-bulk.txt");
  };

  const onWpPublish = async (status: "draft" | "publish") => {
    if (!detailArticle) return;
    const publishProjectId = detailArticle.project_id ?? liveStatus?.result?.project_id;
    if (!publishProjectId) {
      setWpResult(t("tasks.wpMissingProject" as any) || "Missing project ID for WordPress publish.");
      return;
    }
    if (riskAssessment?.risk_level === "blocked") {
      setWpResult(RISK_COPY[locale].blockedPublish);
      return;
    }
    setWpPublishing(true);
    setWpResult(null);
    try {
      await apiRequest(`/content/${detailArticle.id}/publish/wordpress`, {
        method: "POST", token
      }, {
        project_id: publishProjectId,
        post_status: status,
      });
      setWpResult(t("tasks.wpPublished") || "Published to WordPress successfully.");
    } catch (e) {
      setWpResult(e instanceof ApiError ? e.detail : (t("tasks.wpPublishError") || "Failed to publish"));
    } finally {
      setWpPublishing(false);
    }
  };

  const filterTabs: Array<{ key: FilterTab; label: string; count: number }> = [
    { key: "all", label: t("common.all"), count: kpis.total },
    { key: "SUCCESS", label: t("common.success"), count: kpis.success },
    { key: "FAILURE", label: t("common.failure"), count: kpis.failure },
    { key: "RUNNING", label: t("common.running"), count: kpis.running },
  ];
  const seoFallback =
    locale === "fa"
      ? "برای این مقاله هنوز داده سئو ذخیره نشده است."
      : locale === "ar"
        ? "لا توجد بيانات سيو محفوظة لهذه المقالة بعد."
        : "No SEO data is stored for this article yet.";
  const riskCopy = RISK_COPY[locale];

  /* ════════════════════════════════════════════════════════════════════════
     Master-Detail Layout: Smooth Dynamic Drawers and Logical Properties Only
     ════════════════════════════════════════════════════════════════════════ */
  return (
    <section className="macos-content-scope animate-fade-in relative flex min-h-[calc(100vh-96px)] flex-col space-y-4 bg-transparent p-3 md:p-4" dir="auto">

      {/* ── Apple-Style Header & Toolbar ── */}
      <div className="flex flex-col justify-between gap-4 pb-1 md:flex-row md:items-start">
        <div className="flex-1">
          <h2 className="text-[20px] font-semibold text-gray-900 dark:text-gray-100">{t("tasks.title") || "Task History"}</h2>
          <p className="mt-1 text-[13px] text-gray-500 dark:text-gray-300">{t("tasks.subtitle") || "Review, export, and monitor pipeline progress."}</p>
        </div>

        <div className="flex w-full flex-wrap items-center gap-2 rounded-xl border border-black/5 bg-white/[0.95] p-1.5 dark:border-white/10 dark:bg-surface md:mt-0 md:w-auto">
          {/* iOS Toggle Switch for Auto Refresh */}
          <div className="flex items-center gap-3 px-3">
            <span className="text-[13px] font-semibold text-gray-700 dark:text-gray-200">{t("tasks.autoRefresh") || "Auto-refresh"}</span>
            <button
              type="button"
              role="switch"
              aria-checked={autoRefresh}
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={clsx(
                "relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2",
                autoRefresh ? "bg-teal-500" : "bg-gray-200 dark:bg-white/10"
              )}
            >
              <span
                aria-hidden="true"
                className={clsx(
                  "pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white/90 dark:bg-white/80 shadow ring-0 transition duration-200 ease-in-out",
                  autoRefresh ? (locale === "ar" || locale === "fa" ? "-translate-x-5" : "translate-x-5") : "translate-x-0"
                )}
              />
            </button>
          </div>

          <div className="w-px h-6 bg-gray-200 dark:bg-white/10 mx-1 hidden sm:block" />

          <button
            type="button"
            onClick={() => void loadTasks()}
            className="flex h-8 w-8 items-center justify-center rounded-md bg-gray-50 text-gray-600 transition-colors hover:bg-gray-100 hover:text-gray-900 dark:bg-surface-alt dark:text-gray-200 dark:hover:bg-white/[0.12] dark:hover:text-gray-100"
            title={t("common.refresh")}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
          </button>

          {kpis.success > 0 && (
            <Button variant="outlined" onClick={() => void onBulkDownload()} className="h-8 rounded-md border-gray-200 bg-white px-3 text-[13px] shadow-none hover:border-teal-500 hover:bg-teal-50/50 hover:text-teal-700 dark:border-white/10 dark:bg-surface-alt dark:hover:bg-teal-500/10 dark:hover:text-teal-300">
              <svg className="w-4 h-4 mie-2 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
              {t("tasks.bulkDownload") || "Bulk Download"}
            </Button>
          )}
        </div>
      </div>

      {/* Interactive KPI Filter Chips */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {[
            { key: "all", label: t("tasks.kpiTotal") || "Total", value: kpis.total, text: "text-slate-900 dark:text-gray-100", icon: <svg className="w-5 h-5 text-slate-400 dark:text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" /></svg> },
            { key: "SUCCESS", label: t("tasks.kpiSuccess") || "Success", value: kpis.success, text: "text-emerald-700 dark:text-emerald-300", icon: <svg className="w-5 h-5 text-emerald-500 dark:text-emerald-300" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg> },
            { key: "FAILURE", label: t("tasks.kpiFailure") || "Failed", value: kpis.failure, text: "text-red-700 dark:text-red-300", icon: <svg className="w-5 h-5 text-red-500 dark:text-red-300" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg> },
            { key: "RUNNING", label: t("tasks.kpiRunning") || "Running", value: kpis.running, text: "text-teal-700 dark:text-teal-300", icon: <svg className="w-5 h-5 text-teal-500 dark:text-teal-300 animate-spin-slow" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg> },
          ].map((card) => {
          const isActive = filter === card.key;
          return (
            <button
              key={card.key}
              onClick={() => setFilter(card.key as FilterTab)}
              className={clsx(
                "relative group flex flex-col justify-between rounded-xl border p-4 text-start transition-colors duration-150 outline-none focus:ring-2 focus:ring-teal-500/20",
                "border-black/5 dark:border-white/10",
                isActive ? "bg-white ring-1 ring-teal-400/50 dark:bg-surface-alt" : "cursor-pointer bg-white hover:bg-gray-50 dark:bg-surface dark:hover:bg-surface-alt"
              )}
            >
              <div className="mb-6 flex w-full items-start justify-end">
                <div className="rounded-lg border border-black/5 bg-gray-50 p-2 dark:border-white/10 dark:bg-white/[0.06]">
                  {card.icon}
                </div>
              </div>
              <div className="flex w-full items-end justify-between">
                <span className={clsx("text-[24px] font-semibold leading-none tabular-nums", card.text)}>{card.value}</span>
                <span className="pb-1 text-end text-[11px] font-medium text-slate-500 dark:text-gray-300">{card.label}</span>
              </div>
            </button>
          )
        })}
      </div>

      <Modal open={Boolean(deleteConfirmId)} onClose={() => setDeleteConfirmId(null)} title={t("tasks.deleteTask") || "Delete Task"} footer={
        <>
          <Button variant="outlined" onClick={() => setDeleteConfirmId(null)}>{t("common.cancel")}</Button>
          <Button variant="danger" onClick={() => deleteConfirmId && void onDeleteTask(deleteConfirmId)}>{t("common.delete")}</Button>
        </>
      }>
        <p className="text-[14px] text-gray-600 dark:text-gray-300">{t("tasks.confirmDeleteTask") || "Are you sure you want to permanently delete this task data?"}</p>
      </Modal>

      {/* ── Search Bar ── */}
      <div className="flex flex-wrap items-center justify-end w-full">
        {/* Search Input with properly aligned Icon (pis) */}
        <div className="relative w-full md:w-80 shrink-0 group">
          <input
            placeholder={t("tasks.searchPlaceholder") || "Search tasks..."}
            className="w-full rounded-xl border border-black/5 bg-white ps-10 pe-3 py-2 text-[14px] font-medium text-slate-700 outline-none transition-colors duration-150 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:ring-2 focus:ring-teal-500/20 dark:border-white/10 dark:bg-surface dark:text-gray-100 dark:placeholder:text-gray-400 dark:focus:bg-surface-alt"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
            <svg className="absolute start-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400 dark:text-gray-300 group-focus-within:text-teal-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      {/* ── Dynamic Master-Detail Layout Wrapper ── */}
      <div className={clsx(
        "grid w-full flex-1 gap-4 transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)]",
        selectedTaskId ? "grid-cols-1 xl:grid-cols-[1fr_450px]" : "grid-cols-1"
      )}>

        {/* Master: Data Table */}
        <div className="flex min-w-0 flex-col overflow-hidden rounded-xl border border-black/5 bg-white dark:border-white/10 dark:bg-surface">
          <div className="flex-1 overflow-auto rounded-xl">
            <table className="w-full text-start border-collapse">
              <thead className="sticky top-0 z-10 border-b border-gray-200/80 bg-gray-50 dark:border-white/10 dark:bg-surface-alt">
                <tr className="text-[12px] font-semibold text-slate-400 dark:text-gray-400">
                  <th className="px-6 py-5 text-start font-bold w-1/2">{t("tasks.topic") || "Topic"}</th>
                  <th className="px-6 py-5 text-start font-bold w-1/4">{t("tasks.status") || "Status"}</th>
                  <th className="px-6 py-5 text-start font-bold w-1/4">{t("tasks.created") || "Date"}</th>
                  <th className="px-6 py-5 text-end font-bold sr-only w-16">{t("users.action") || "Action"}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-white/10">
                {loading ? (
                  [1, 2, 3, 4, 5].map((i) => (
                    <tr key={i} className="animate-pulse">
                      <td className="px-6 py-5">
                        <div className="h-5 bg-slate-100 dark:bg-white/10 rounded-md w-3/4 mb-2"></div>
                        <div className="h-3 bg-slate-50 dark:bg-white/10 rounded-md w-1/3"></div>
                      </td>
                      <td className="px-6 py-5"><div className="h-7 w-24 bg-slate-100 dark:bg-white/10 rounded-full"></div></td>
                      <td className="px-6 py-5">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full bg-slate-100 dark:bg-white/10"></div>
                          <div className="h-4 w-20 bg-slate-100 dark:bg-white/10 rounded-md"></div>
                        </div>
                      </td>
                      <td className="px-6 py-5 text-end"><div className="h-8 w-8 bg-slate-100 dark:bg-white/10 rounded-full ms-auto"></div></td>
                    </tr>
                  ))
                ) : filtered.length === 0 ? (
                  <tr className="hover:bg-transparent">
                    <td colSpan={4} className="px-6 py-24 text-center">
                      <svg className="mx-auto w-16 h-16 text-gray-200 dark:text-gray-600 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                      </svg>
                      <p className="text-[15px] font-semibold text-gray-500 dark:text-gray-300">{t("tasks.noTasks") || "No tasks found"}</p>
                    </td>
                  </tr>
                ) : (
                  filtered.map((task) => {
                    const isSelected = task.task_id === selectedTaskId;
                    const statusUpper = task.status?.toUpperCase() ?? "";
                    return (
                      <tr
                        key={task.task_id}
                        className={clsx(
                          "border-b border-gray-100 dark:border-white/10 transition-colors duration-200 cursor-pointer",
                          isSelected ? "bg-teal-50/50 dark:bg-teal-500/10" : "hover:bg-gray-50/50 dark:hover:bg-white/[0.05]"
                        )}
                        onClick={() => { setSelectedTaskId(task.task_id); setDetailArticle(null); setDetailTab("content"); setContentView("reader"); setWpResult(null); }}
                      >
                        <td className="px-6 py-4">
                          <div className="flex flex-col">
                            <span className={clsx("text-[14px] font-semibold truncate max-w-sm", isSelected ? "text-teal-900 dark:text-teal-200" : "text-gray-900 dark:text-gray-100")}>
                              {task.topic || task.task_name || task.task_id.slice(0, 12)}
                            </span>
                            <code className="text-[11px] text-gray-400 dark:text-gray-300 mt-0.5 truncate max-w-sm" dir="ltr">#{task.task_id}</code>
                          </div>
                        </td>
                        <td className="px-6 py-4"><StatusBadge status={statusUpper} /></td>
                        <td className="px-6 py-4 text-[13px] text-gray-500 dark:text-gray-400 font-medium">{formatDate(task.created_at)}</td>
                        <td className="px-6 py-4 text-end">
                          <button
                            type="button"
                            onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(task.task_id); }}
                            className="w-8 h-8 inline-flex items-center justify-center rounded-full text-gray-400 dark:text-gray-300 hover:text-red-600 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-500/10 transition-colors"
                            title={t("common.delete") || "Delete"}
                          >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Detail: Slide-over Context Panel */}
        {selectedTaskId && (
          <aside className="animate-slide-in-end sticky top-4 flex h-[calc(100vh-168px)] min-w-0 flex-col rounded-xl border border-black/5 bg-white dark:border-white/10 dark:bg-surface">
            <div className="p-6 border-b border-black/5 dark:border-white/10 flex items-center justify-between">
              <h3 className="text-[15px] font-semibold text-gray-900 dark:text-gray-100">{t("tasks.detail") || "Task Analysis"}</h3>
              <div className="flex gap-2">
                {streamActive && <span className="flex items-center gap-1.5 text-[12px] font-bold text-emerald-500 tracking-wider uppercase"><span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" /> Live</span>}
                <button onClick={() => setSelectedTaskId(null)} className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-100 dark:bg-white/10 hover:bg-gray-200 dark:hover:bg-white/15 transition-colors text-gray-600 dark:text-gray-300">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" /></svg>
                </button>
              </div>
            </div>

            <div className="p-6 flex-1 overflow-y-auto space-y-6">
              {/* Status Block */}
              {liveStatus ? (
                <div className="space-y-4">
                  <div className="space-y-3 rounded-xl border border-black/5 bg-gray-50 p-4 dark:border-white/10 dark:bg-surface-alt">
                    <div className="flex items-center justify-between">
                      <StatusBadge status={liveStatus.state} />
                      <button onClick={() => void navigator.clipboard.writeText(selectedTaskId)} className="text-[12px] text-gray-400 dark:text-gray-300 hover:text-teal-600 dark:hover:text-teal-300 font-mono transition-colors active:scale-95 flex items-center gap-1">
                        {t("tasks.copyId") || "Copy ID"}
                      </button>
                    </div>
                    {liveStatus.status && <p className="text-[14px] text-gray-800 dark:text-gray-200 font-medium leading-relaxed">{liveStatus.status}</p>}

                    {/* Progress bar if numerical */}
                    {typeof liveStatus.progress === "number" && liveStatus.progress > 0 && liveStatus.progress < 100 && (
                      <div className="h-1.5 w-full bg-gray-200 dark:bg-white/10 rounded-full overflow-hidden mt-2">
                        <div className="h-full bg-teal-500 transition-all duration-500 ease-out" style={{ width: `${liveStatus.progress}%` }} />
                      </div>
                    )}
                  </div>

                  {/* Failure Trace */}
                  {liveStatus.state === "FAILURE" && (
                    <div className="rounded-xl border border-red-100 bg-red-50/50 p-5 dark:border-red-500/20 dark:bg-red-500/10">
                      <h4 className="text-[12px] font-medium text-red-700 dark:text-red-300 mb-2">{t("tasks.failureTrace") || "Failure Trace"}</h4>
                      <pre className="text-[11px] text-red-600 dark:text-red-300 font-mono whitespace-pre-wrap max-h-40 overflow-auto" dir="ltr">
                        {liveStatus.error ?? liveStatus.last_error ?? (t("common.unexpectedError") || "Unknown error occurred.")}
                      </pre>
                    </div>
                  )}

                  {/* Success Article Payload */}
                  {liveStatus.state === "SUCCESS" && detailArticle && (
                    <div className="space-y-6">
                      {/* Metric Chips */}
                      <div className="grid grid-cols-3 gap-3">
                        <div className="rounded-lg border border-gray-100 bg-gray-50 p-3 text-center dark:border-white/10 dark:bg-surface-alt">
                          <span className="block text-[11px] font-medium text-gray-400 dark:text-gray-300">{t("tasks.wordCount") || "Words"}</span>
                          <span className="block text-[18px] font-bold text-gray-900 dark:text-gray-100 mt-1">{detailArticle.word_count ?? "—"}</span>
                        </div>
                        <div className="rounded-lg border border-gray-100 bg-gray-50 p-3 text-center dark:border-white/10 dark:bg-surface-alt">
                          <span className="block text-[11px] font-medium text-gray-400 dark:text-gray-300">{t("tasks.qualityScore") || "Quality"}</span>
                          <span className={clsx("block text-[18px] font-bold mt-1", (detailArticle.quality_score ?? 0) >= 80 ? "text-emerald-600" : "text-amber-600")}>{detailArticle.quality_score ?? "—"}</span>
                        </div>
                        <div className="rounded-lg border border-gray-100 bg-gray-50 p-3 text-center dark:border-white/10 dark:bg-surface-alt">
                          <span className="block text-[11px] font-medium text-gray-400 dark:text-gray-300">{t("tasks.cost") || "Cost"}</span>
                          <span className="block text-[18px] font-bold text-teal-600 mt-1">{detailArticle.cost_usd ? `$${detailArticle.cost_usd.toFixed(3)}` : "—"}</span>
                        </div>
                      </div>

                      {/* Inner Sub-Navigation (Segmented) */}
                      <div className="inline-flex w-full rounded-md bg-gray-100 p-1 dark:bg-white/10">
                        {[
                          { key: "content" as DetailTab, label: t("tasks.contentTab") || "Content" },
                          { key: "seo" as DetailTab, label: t("tasks.seoTab") || "SEO" },
                          { key: "export" as DetailTab, label: t("tasks.exportTab") || "Export" },
                        ].map((tab) => (
                          <button key={tab.key} onClick={() => setDetailTab(tab.key)} className={clsx("flex-1 rounded-lg py-1.5 text-[13px] font-semibold transition-all", detailTab === tab.key ? "bg-white text-gray-900 shadow-sm dark:bg-white/15 dark:text-gray-100" : "text-gray-500 hover:text-gray-700 dark:text-gray-300 dark:hover:text-gray-100")}>
                            {tab.label}
                          </button>
                        ))}
                      </div>

                      {/* Content Views */}
                      {detailTab === "content" && (
                        <div className="flex flex-col gap-3">
                          <div className="flex gap-2 mb-2">
                            {(["reader", "raw", "edit"] as ContentView[]).map(cv => (
                              <button key={cv} onClick={() => setContentView(cv)} className={clsx("text-[12px] font-bold uppercase tracking-wider px-3 py-1 rounded-full transition-colors", contentView === cv ? "bg-teal-50 dark:bg-teal-500/15 text-teal-700 dark:text-teal-300" : "text-gray-400 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-white/[0.06]")}>
                                {cv === "reader" ? t("tasks.readerMode") || "Reader" : cv === "raw" ? t("tasks.rawHtml") || "Raw" : t("tasks.editMode") || "Edit"}
                              </button>
                            ))}
                          </div>
                          <div className="rounded-xl border border-gray-200 bg-white dark:border-white/10 dark:bg-white/[0.05]">
                            {contentView === "reader" && (
                              <article className="prose prose-sm prose-teal max-w-none whitespace-pre-wrap p-5 font-sans text-[14px] leading-relaxed text-gray-800 dark:text-gray-200" dir={locale === "en" ? "ltr" : "rtl"}>
                                {toReaderText(detailArticle.html_content ?? detailArticle.content)}
                              </article>
                            )}
                            {contentView === "raw" && (
                              <pre className="max-h-96 overflow-auto rounded-xl border border-black/5 bg-slate-950 p-4 font-mono text-[12px] text-slate-100 whitespace-pre-wrap select-all dark:border-white/10 dark:bg-slate-950" dir="ltr">{detailArticle.html_content ?? detailArticle.content}</pre>
                            )}
                            {contentView === "edit" && (
                              <textarea className="w-full h-96 p-4 outline-none resize-none bg-transparent font-mono text-[13px] text-gray-700 dark:text-gray-200 leading-relaxed" dir="auto" value={editContent} onChange={(e) => setEditContent(e.target.value)} />
                            )}
                          </div>
                        </div>
                      )}

                      {detailTab === "seo" && (() => {
                        const storedSeo = detailArticle.seo_analysis;
                        const seoScore = qualityMetrics?.seo_score?.score ?? storedSeo?.score;
                        const componentScores = qualityMetrics?.seo_score?.component_scores ?? {};
                        const recommendations = qualityMetrics?.seo_score?.recommendations ?? storedSeo?.recommendations ?? [];
                        const checklist = storedSeo?.checklist ?? [];
                        const hasSeoData =
                          typeof seoScore === "number" ||
                          Object.keys(componentScores).length > 0 ||
                          recommendations.length > 0 ||
                          checklist.length > 0 ||
                          qualityMetrics?.readability_grade;

                        return (
                          <div className="space-y-4">
                            <section className="rounded-xl border border-gray-200 bg-white p-4 dark:border-white/10 dark:bg-white/[0.05]">
                              <div className="flex flex-wrap items-start justify-between gap-3">
                                <div>
                                  <h4 className="text-[14px] font-bold text-gray-900 dark:text-gray-100">{t("tasks.seoTab") || "SEO"}</h4>
                                  <p className="mt-1 text-[12px] text-gray-500 dark:text-gray-300">
                                    {qualityLoading
                                      ? t("common.loading")
                                      : qualityMetrics?.readability_grade
                                        ? qualityMetrics.readability_grade
                                        : seoFallback}
                                  </p>
                                </div>
                                <span className="rounded-lg border border-black/5 bg-gray-50 px-3 py-2 text-[18px] font-bold tabular-nums text-gray-900 dark:border-white/10 dark:bg-white/[0.06] dark:text-gray-100">
                                  {formatPercentScore(seoScore)}
                                </span>
                              </div>
                            </section>

                            {qualityError && (
                              <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-[13px] font-medium text-red-700 dark:text-red-300">
                                {qualityError}
                              </div>
                            )}

                            {Object.keys(componentScores).length > 0 && (
                              <section className="rounded-xl border border-gray-200 bg-white p-4 dark:border-white/10 dark:bg-white/[0.05]">
                                <h5 className="mb-3 text-[13px] font-bold text-gray-900 dark:text-gray-100">{t("tasks.seoScore") || "SEO Score"}</h5>
                                <div className="grid gap-2">
                                  {Object.entries(componentScores).map(([key, value]) => (
                                    <div key={key} className="flex items-center justify-between gap-3 rounded-lg bg-gray-50 px-3 py-2 dark:bg-white/[0.04]">
                                      <span className="text-[12px] font-medium text-gray-600 dark:text-gray-300">{humanizeMetricKey(key)}</span>
                                      <span className="text-[12px] font-bold tabular-nums text-gray-900 dark:text-gray-100">{formatPercentScore(value)}</span>
                                    </div>
                                  ))}
                                </div>
                              </section>
                            )}

                            {checklist.length > 0 && (
                              <section className="rounded-xl border border-gray-200 bg-white p-4 dark:border-white/10 dark:bg-white/[0.05]">
                                <h5 className="mb-3 text-[13px] font-bold text-gray-900 dark:text-gray-100">{t("tasks.seoChecklist") || "Checklist"}</h5>
                                <div className="space-y-2">
                                  {checklist.map((item) => (
                                    <div key={item.label} className="flex items-start gap-2 text-[12px] leading-5 text-gray-600 dark:text-gray-300">
                                      <span className={clsx("mt-1 h-2 w-2 shrink-0 rounded-full", item.passed ? "bg-emerald-500" : "bg-amber-400")} />
                                      <span>
                                        <span className="font-semibold text-gray-900 dark:text-gray-100">{item.label}</span>
                                        {item.detail ? `: ${item.detail}` : ""}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </section>
                            )}

                            {recommendations.length > 0 && (
                              <section className="rounded-xl border border-gray-200 bg-white p-4 dark:border-white/10 dark:bg-white/[0.05]">
                                <h5 className="mb-3 text-[13px] font-bold text-gray-900 dark:text-gray-100">{t("tasks.recommendations") || "Recommendations"}</h5>
                                <div className="space-y-2">
                                  {recommendations.map((recommendation) => (
                                    <p key={recommendation} className="rounded-lg bg-amber-500/10 px-3 py-2 text-[12px] leading-5 text-amber-800 dark:text-amber-200">
                                      {recommendation}
                                    </p>
                                  ))}
                                </div>
                              </section>
                            )}

                            {!qualityLoading && !qualityError && !hasSeoData && (
                              <div className="rounded-xl border border-gray-200 bg-white p-5 text-center text-[13px] text-gray-500 dark:border-white/10 dark:bg-white/[0.05] dark:text-gray-300">
                                {seoFallback}
                              </div>
                            )}
                          </div>
                        );
                      })()}
                      {detailTab === "export" && (
                        <div className="space-y-4">
                          <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-white/10 dark:bg-white/[0.05]">
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <h4 className="text-[14px] font-bold text-gray-900 dark:text-gray-100">{riskCopy.title}</h4>
                                <p className="mt-1 text-[12px] text-gray-500 dark:text-gray-300">
                                  {riskLoading
                                    ? riskCopy.loading
                                    : riskAssessment
                                      ? `${riskAssessment.overall_score}/100`
                                      : t("monitoring.statusUnknown" as any) || "Unknown"}
                                </p>
                              </div>
                              {riskAssessment && (
                                <span
                                  className={clsx(
                                    "inline-flex h-7 items-center rounded-md px-2.5 text-[11px] font-bold",
                                    riskAssessment.risk_level === "blocked" || riskAssessment.risk_level === "high"
                                      ? "bg-red-500/10 text-red-700 dark:text-red-300"
                                      : riskAssessment.risk_level === "medium"
                                        ? "bg-amber-500/10 text-amber-700 dark:text-amber-300"
                                        : "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                                  )}
                                >
                                  {riskAssessment.risk_level === "blocked"
                                    ? riskCopy.blocked
                                    : riskAssessment.risk_level === "high"
                                      ? riskCopy.high
                                      : riskAssessment.risk_level === "medium"
                                        ? riskCopy.medium
                                        : riskCopy.low}
                                </span>
                              )}
                            </div>
                            {riskAssessment?.issues?.length ? (
                              <div className="mt-3 space-y-2">
                                {riskAssessment.issues.slice(0, 3).map((issue) => (
                                  <p key={issue.id} className="text-[12px] leading-5 text-gray-600 dark:text-gray-300">
                                    <span className="font-semibold text-gray-900 dark:text-gray-100">{issue.category}: </span>
                                    {issue.message}
                                  </p>
                                ))}
                              </div>
                            ) : null}
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <Button variant="outlined" onClick={() => downloadContent(detailArticle, "txt")}>{t("tasks.downloadTxt") || "Text"}</Button>
                            <Button variant="outlined" onClick={() => downloadContent(detailArticle, "html")}>{t("tasks.downloadHtml") || "HTML"}</Button>
                            <Button variant="outlined" onClick={() => void navigator.clipboard.writeText(contentView === "edit" ? editContent : detailArticle.content)} className="col-span-2">{t("tasks.copyContent") || "Copy Full Content"}</Button>
                          </div>
                          <div className="rounded-xl border border-blue-100 bg-blue-50/50 p-5 dark:border-blue-500/20 dark:bg-blue-500/10">
                            <h4 className="text-[14px] font-bold text-blue-900 dark:text-blue-200 mb-3">{t("tasks.wpPublish") || "WordPress Publish"}</h4>
                            <div className="flex gap-3">
                              <Button variant="outlined" size="sm" loading={wpPublishing} disabled={riskAssessment?.risk_level === "blocked"} onClick={() => void onWpPublish("draft")}>{t("tasks.wpDraft") || "Draft"}</Button>
                              <Button variant="primary" size="sm" loading={wpPublishing} disabled={riskAssessment?.risk_level === "blocked"} onClick={() => void onWpPublish("publish")}>{t("tasks.wpLive") || "Publish Live"}</Button>
                            </div>
                            {wpResult && <p className={clsx("mt-3 text-[12px] font-medium", wpResult.includes("error") || wpResult.includes("خطا") ? "text-red-600 dark:text-red-300" : "text-blue-600 dark:text-blue-300")}>{wpResult}</p>}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-4 animate-pulse">
                  <div className="h-24 w-full rounded-xl bg-gray-100 dark:bg-white/10" />
                  <div className="h-64 w-full rounded-xl bg-gray-100 dark:bg-white/10" />
                </div>
              )}
            </div>
          </aside>
        )}
      </div>

    </section>
  );
}

/* ─── Helper Components ─── */
function StatusBadge({ status }: { status: string }) {
  const s = status.toUpperCase();
  const cls = s === "SUCCESS"
    ? "border-emerald-200/60 bg-emerald-50 text-emerald-700 dark:border-emerald-400/30 dark:bg-emerald-500/[0.12] dark:text-emerald-200"
    : ["FAILURE", "FAILED"].includes(s)
      ? "border-red-200/60 bg-red-50 text-red-700 dark:border-red-400/30 dark:bg-red-500/[0.12] dark:text-red-200"
      : "border-teal-200/60 bg-teal-50 text-teal-700 animate-pulse-soft dark:border-teal-400/30 dark:bg-teal-500/[0.12] dark:text-teal-200";

  return (
    <span className={clsx("inline-flex items-center justify-center rounded-lg border px-2.5 py-1 text-[11px] font-bold uppercase tracking-wider", cls)}>
      {status}
    </span>
  );
}

/* ─── Helper Functions ─── */
function formatDate(d?: string): string {
  if (!d) return "—";
  try { return new Date(d).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
  catch { return d; }
}

function formatPercentScore(value?: number): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const normalized = value <= 1 ? value * 100 : value;
  return `${Math.round(normalized)}%`;
}

function humanizeMetricKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function toReaderText(content?: string): string {
  if (!content) return "";
  return content
    .replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?>[\s\S]*?<\/style>/gi, "")
    .replace(/<\/(p|div|h[1-6]|li|ul|ol|blockquote|br)>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, "\"")
    .replace(/&#39;/g, "'")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function downloadContent(article: ArticleDetail, format: "txt" | "html") {
  const content = format === "html" ? (article.html_content ?? article.content) : article.content;
  const blob = new Blob([content], { type: format === "html" ? "text/html;charset=utf-8" : "text/plain;charset=utf-8" });
  downloadBlob(blob, `${article.title || "article"}.${format}`);
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}
