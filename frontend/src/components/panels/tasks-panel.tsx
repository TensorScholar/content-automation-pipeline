"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import {
  TaskHistoryItem,
  TaskStatusResponse,
  ArticleDetail,
  DraftRiskAssessment,
  ArticleReviewAction,
  ArticleReviewState,
  ProjectReadiness,
} from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { StatusBadge as UiStatusBadge } from "@/components/ui/status-badge";
import { useToast } from "@/components/ui/toast";
import { Modal } from "@/components/ui/modal";
import { ToggleSwitch } from "@/components/ui/toggle-switch";
import { TASK_COPY, RISK_COPY, REVIEW_COPY, PUBLISH_COPY } from "./tasks/task-constants";
import type { TaskLocale } from "./tasks/task-constants";
import {
  isWordPressPublishReadinessItem,
  DiagnosticItem,
  StatusBadge,
  ReviewPanel,
  reviewVariant,
  reviewLabel,
  localizeTaskStatus,
  localizeTaskResult,
  formatDiagnosticNumber,
  formatWordRange,
  formatDiagnosticLanguage,
  formatBoolean,
  formatQualityFindingActual,
  localizeQualityFinding,
  formatPublishResult,
  localizeRiskCategory,
  localizeRiskMessage,
  resolveArticleDirection,
  formatDate,
  formatPercentScore,
  readFiniteNumber,
  qualityGrade,
  humanizeMetricKey,
  toReaderText,
  downloadContent,
  downloadBlob,
} from "./tasks/task-helpers";

interface TasksPanelProps {
  token: string;
  canReview?: boolean;
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

interface ContentHistoryResponse {
  current_version: Record<string, unknown>;
  revisions: Array<{
    id: string;
    content: string;
    revision_note: string;
    created_at: string;
    word_count: number;
  }>;
  total_revisions: number;
}

type FilterTab = "all" | "SUCCESS" | "FAILURE" | "RUNNING";
type DetailTab = "content" | "seo" | "export" | "history";
type ContentView = "reader" | "raw" | "edit";
type ReadinessItem = ProjectReadiness["blocking_items"][number];

export function TasksPanel({ token, canReview = false }: TasksPanelProps) {
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
  const publishControllerRef = useRef<AbortController | null>(null);
  const [wpResult, setWpResult] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      publishControllerRef.current?.abort();
    };
  }, []);
  const [riskAssessment, setRiskAssessment] = useState<DraftRiskAssessment | null>(null);
  const [riskLoading, setRiskLoading] = useState(false);
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetricsResponse | null>(null);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [qualityError, setQualityError] = useState<string | null>(null);
  const [reviewState, setReviewState] = useState<ArticleReviewState | null>(null);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [reviewAction, setReviewAction] = useState<Exclude<ArticleReviewAction, "approve"> | null>(null);
  const [reviewNote, setReviewNote] = useState("");
  const [reviewSubmitting, setReviewSubmitting] = useState(false);
  const [publishReadiness, setPublishReadiness] = useState<ProjectReadiness | null>(null);
  const [publishReadinessLoading, setPublishReadinessLoading] = useState(false);

  const [savingEdit, setSavingEdit] = useState(false);
  const [articleHistory, setArticleHistory] = useState<ContentHistoryResponse | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const loadHistory = useCallback(async (articleId: string, signal?: AbortSignal) => {
    setLoadingHistory(true);
    try {
      const res = await apiRequest<ContentHistoryResponse>(`/content/${articleId}/history`, { token, signal });
      if (signal?.aborted) return;
      setArticleHistory(res);
    } catch (e) {
      if (signal?.aborted) return;
      console.error(e);
      setArticleHistory(null);
    } finally {
      if (signal?.aborted) return;
      setLoadingHistory(false);
    }
  }, [token]);

  useEffect(() => {
    if (detailTab === "history" && detailArticle?.id) {
      const controller = new AbortController();
      void loadHistory(detailArticle.id, controller.signal);
      return () => controller.abort();
    }
  }, [detailTab, detailArticle?.id, loadHistory]);

  const handleSaveEdit = async () => {
    if (!detailArticle || editContent === detailArticle.content) return;
    setSavingEdit(true);
    try {
      const saved = await apiRequest<{
        content: string;
        word_count: number;
      }>(`/content/${detailArticle.id}`, {
        method: "PUT",
        token,
        body: { content: editContent, revision_note: TASK_COPY[locale].revisionNote },
      });
      setEditContent(saved.content);
      setDetailArticle({
        ...detailArticle,
        content: saved.content,
        word_count: saved.word_count,
      });
      showToast("success", TASK_COPY[locale].editSaved);
      // If history was already loaded, refresh it so the new revision appears
      if (articleHistory) {
        void loadHistory(detailArticle.id);
      }
    } catch (e) {
      console.error("Failed to save edit", e);
      showToast(
        "error",
        e instanceof ApiError ? e.detail : TASK_COPY[locale].editSaveFailed,
      );
    } finally {
      setSavingEdit(false);
    }
  };

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
      setReviewState(null);
      setReviewLoading(false);
      setReviewError(null);
      return;
    }
    const controller = new AbortController();
    const load = async () => {
      setRiskLoading(true);
      setReviewLoading(true);
      setQualityLoading(true);
      setReviewError(null);
      try {
        const [articleResult, riskResult, reviewResult, qualityResult] = await Promise.allSettled([
          apiRequest<ArticleDetail>(`/content/${articleId}`, {
            token,
            signal: controller.signal,
          }),
          apiRequest<DraftRiskAssessment>(`/content/${articleId}/risk-assessment`, {
            token,
            signal: controller.signal,
          }),
          apiRequest<ArticleReviewState>(`/content/${articleId}/review`, {
            token,
            signal: controller.signal,
          }),
          apiRequest<QualityMetricsResponse>(`/content/${articleId}/quality`, {
            token,
            signal: controller.signal,
            timeoutMs: 20000,
          }),
        ]);
        if (controller.signal.aborted) return;
        if (articleResult.status === "fulfilled") {
          setDetailArticle(articleResult.value);
          setEditContent(articleResult.value.content ?? "");
        } else {
          setDetailArticle(null);
        }
        if (qualityResult.status === "fulfilled") {
          setQualityMetrics(qualityResult.value);
          setQualityError(null);
        } else {
          setQualityMetrics(null);
          setQualityError(null);
        }
        setRiskAssessment(riskResult.status === "fulfilled" ? riskResult.value : null);
        if (reviewResult.status === "fulfilled") {
          setReviewState(reviewResult.value);
        } else {
          setReviewState(null);
          setReviewError(REVIEW_COPY[locale].unavailable);
        }
      } catch {
        if (!controller.signal.aborted) {
          setDetailArticle(null);
          setRiskAssessment(null);
          setReviewState(null);
          setReviewError(REVIEW_COPY[locale].unavailable);
        }
      } finally {
        if (!controller.signal.aborted) setRiskLoading(false);
        if (!controller.signal.aborted) setReviewLoading(false);
        if (!controller.signal.aborted) setQualityLoading(false);
      }
    };
    void load();
    return () => controller.abort();
  }, [liveStatus?.result?.article_id, locale, token]);

  const publishProjectId = detailArticle?.project_id ?? liveStatus?.result?.project_id;

  useEffect(() => {
    if (!publishProjectId) {
      setPublishReadiness(null);
      setPublishReadinessLoading(false);
      return;
    }

    const controller = new AbortController();
    setPublishReadiness(null);
    setPublishReadinessLoading(true);
    apiRequest<ProjectReadiness>(`/projects/${publishProjectId}/readiness`, {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => {
        if (!controller.signal.aborted) setPublishReadiness(payload);
      })
      .catch(() => {
        if (!controller.signal.aborted) setPublishReadiness(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setPublishReadinessLoading(false);
      });

    return () => controller.abort();
  }, [publishProjectId, token]);

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
      showToast("success", t("tasks.taskDeleted"));
      if (selectedTaskId === taskId) { setSelectedTaskId(null); setLiveStatus(null); }
      await loadTasks();
    } catch (e) {
      showToast("error", e instanceof ApiError ? e.detail : (t("common.unexpectedError")));
    }
  };

  const onBulkDownload = async () => {
    const allSuccessful = tasks.filter((t) => t.status?.toUpperCase() === "SUCCESS");
    const successful = allSuccessful.slice(0, 20);
    const results: string[] = [];
    let omitted = allSuccessful.length - successful.length;
    for (const task of successful) {
      const articleId = (task.result as Record<string, unknown> | undefined)?.article_id;
      if (!articleId) {
        omitted += 1;
        continue;
      }
      try {
        const article = await apiRequest<ArticleDetail>(`/content/${String(articleId)}`, { token });
        results.push(`--- ${article.title} ---\n\n${article.content}\n\n`);
      } catch {
        omitted += 1;
      }
    }
    if (results.length === 0) {
      showToast("error", TASK_COPY[locale].bulkDownloadUnavailable);
      return;
    }
    const blob = new Blob([results.join("\n\n")], { type: "text/plain;charset=utf-8" });
    downloadBlob(blob, "articles-bulk.txt");
    showToast(
      omitted > 0 ? "warning" : "success",
      omitted > 0
        ? TASK_COPY[locale].bulkDownloadPartial(results.length, omitted)
        : TASK_COPY[locale].bulkDownloadComplete(results.length),
    );
  };

  const onWpPublish = async (status: "draft" | "publish") => {
    if (!detailArticle || !publishProjectId) return;
    if (riskAssessment?.risk_level === "blocked") {
      setWpResult(RISK_COPY[locale].blockedPublish);
      return;
    }
    if (status === "publish" && reviewState?.status !== "approved") {
      setWpResult(REVIEW_COPY[locale].liveBlocked);
      return;
    }
    if (wordpressActionBlocked) {
      setWpResult(
        publishReadinessLoading
          ? PUBLISH_COPY[locale].checkingWordPress
          : PUBLISH_COPY[locale].wordpressBlocked,
      );
      return;
    }

    publishControllerRef.current?.abort();
    const controller = new AbortController();
    publishControllerRef.current = controller;
    const wait = (ms: number) => new Promise<void>((resolve, reject) => {
      if (controller.signal.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      const timer = window.setTimeout(() => {
        controller.signal.removeEventListener("abort", onAbort);
        resolve();
      }, ms);
      const onAbort = () => {
        window.clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      };
      controller.signal.addEventListener("abort", onAbort, { once: true });
    });

    setWpPublishing(true);
    setWpResult(null);
    try {
      const queued = await apiRequest<{ status: string; publish_status?: string }>(
        `/content/${detailArticle.id}/publish/wordpress`,
        { method: "POST", token, timeoutMs: 15000, signal: controller.signal },
        { project_id: publishProjectId, post_status: status },
      );
      if (queued.status === "success") {
        setWpResult(PUBLISH_COPY[locale].completed);
        return;
      }
      setWpResult(PUBLISH_COPY[locale].queued);
      const terminalSuccess = new Set(["published_as_draft", "published_scheduled", "published_public"]);
      const terminalFailure = new Set(["publish_failed", "publish_validation_failed"]);
      for (let attempt = 0; attempt < 40; attempt += 1) {
        await wait(1500);
        const publishState = await apiRequest<{
          publish_status: string;
          publish_error_message?: string | null;
        }>(`/content/${detailArticle.id}/publish/status`, { token, timeoutMs: 10000, signal: controller.signal });
        if (terminalSuccess.has(publishState.publish_status)) {
          setWpResult(PUBLISH_COPY[locale].completed);
          return;
        }
        if (terminalFailure.has(publishState.publish_status)) {
          throw new Error(publishState.publish_error_message || t("tasks.wpPublishError"));
        }
        if (publishState.publish_status === "publish_retrying") {
          setWpResult(PUBLISH_COPY[locale].retrying);
        }
      }
      setWpResult(PUBLISH_COPY[locale].accepted);
    } catch (error) {
      if (controller.signal.aborted) return;
      setWpResult(formatPublishResult(error instanceof ApiError ? error.detail : error, t("tasks.wpPublishError")));
    } finally {
      if (publishControllerRef.current === controller) publishControllerRef.current = null;
      if (!controller.signal.aborted) setWpPublishing(false);
    }
  };

  const submitReview = async (action: ArticleReviewAction, note?: string) => {
    if (!detailArticle) return;
    setReviewSubmitting(true);
    setReviewError(null);
    try {
      const updated = await apiRequest<ArticleReviewState, { action: ArticleReviewAction; note?: string }>(
        `/content/${detailArticle.id}/review`,
        {
          method: "POST",
          token,
          body: { action, note },
        }
      );
      setReviewState(updated);
      setReviewAction(null);
      setReviewNote("");
      showToast("success", REVIEW_COPY[locale].reviewUpdated);
    } catch (error) {
      const message = error instanceof ApiError ? error.detail : REVIEW_COPY[locale].reviewError;
      setReviewError(message);
      showToast("error", message);
    } finally {
      setReviewSubmitting(false);
    }
  };

  const filterTabs: Array<{ key: FilterTab; label: string; count: number }> = [
    { key: "all", label: t("common.all"), count: kpis.total },
    { key: "SUCCESS", label: t("common.success"), count: kpis.success },
    { key: "FAILURE", label: t("common.failure"), count: kpis.failure },
    { key: "RUNNING", label: t("common.running"), count: kpis.running },
  ];
  const seoFallback =
    TASK_COPY[locale].seoEmpty;
  const riskCopy = RISK_COPY[locale];
  const reviewCopy = REVIEW_COPY[locale];
  const publishCopy = PUBLISH_COPY[locale];
  const taskQualityScore = detailArticle?.quality_score ?? qualityMetrics?.overall_quality?.score;
  const taskCost = readFiniteNumber(liveStatus?.result?.cost) ?? detailArticle?.cost_usd;
  const sourceIsHtml = Boolean(detailArticle?.html_content?.trim());
  const sourceContent = sourceIsHtml ? detailArticle?.html_content ?? "" : detailArticle?.content ?? "";
  const articleDirection = resolveArticleDirection(detailArticle?.language, sourceContent);
  const wordpressReadinessUnavailable = Boolean(publishProjectId) && !publishReadinessLoading && !publishReadiness;
  const wordpressReadinessBlocked = publishReadiness ? !publishReadiness.can_publish : false;
  const wordpressBlockingItem = publishReadiness?.blocking_items.find(isWordPressPublishReadinessItem);
  const wordpressPublishBlocked = !publishProjectId || wordpressReadinessUnavailable || wordpressReadinessBlocked || Boolean(wordpressBlockingItem);
  const wordpressActionBlocked = publishReadinessLoading || wordpressPublishBlocked;
  const publicPublishBlocked = riskAssessment?.risk_level === "blocked" || reviewState?.status !== "approved" || publishReadinessLoading || wordpressPublishBlocked;
  const publicPublishLabel = reviewState?.status !== "approved"
    ? reviewCopy.publishNeedsApproval
    : publishReadinessLoading
      ? publishCopy.checkingWordPress
      : wordpressPublishBlocked
        ? publishCopy.wordpressRequired
        : t("tasks.wpLive");
  const publicPublishReason = reviewState?.status !== "approved"
    ? reviewCopy.liveBlocked
    : wordpressPublishBlocked
      ? publishCopy.wordpressBlocked
      : null;
  const wordpressActionReason = wordpressActionBlocked
    ? publishReadinessLoading
      ? publishCopy.checkingWordPress
      : publishCopy.wordpressBlocked
    : null;

  /* ════════════════════════════════════════════════════════════════════════
     Master-Detail Layout: Smooth Dynamic Drawers and Logical Properties Only
     ════════════════════════════════════════════════════════════════════════ */
  return (
    <section className="smx-page !max-w-none relative flex h-full min-h-0 min-w-0 flex-col gap-4 overflow-hidden" dir="auto">

      {/* Content header and toolbar */}
      <div className="smx-page-header">
        <div className="min-w-0 flex-1">
          <h2 className="smx-page-title">{t("tasks.title")}</h2>
          <p className="mt-1 text-sm text-ink-muted">{t("tasks.subtitle")}</p>
        </div>

        <div className="smx-toolbar min-w-0 flex-wrap md:w-auto">
          <div className="flex items-center px-1">
            <ToggleSwitch checked={autoRefresh} onChange={setAutoRefresh} label={t("tasks.autoRefresh")} />
          </div>

          <div className="hidden h-5 w-px bg-line sm:block" />

          <button
            type="button"
            onClick={() => void loadTasks()}
            className="smx-icon-button"
            title={t("common.refresh")}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
          </button>

          {kpis.success > 0 && (
            <Button variant="outlined" onClick={() => void onBulkDownload()} className="h-8 px-3 text-sm">
              <svg className="w-4 h-4 me-2 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
              {t("tasks.bulkDownload")}
            </Button>
          )}
        </div>
      </div>

      {/* Compact status filters: same filtering capability, less dashboard chrome. */}
      <div className="flex shrink-0 flex-wrap items-center gap-x-1 gap-y-2 border-b border-line pb-3">
        {[
          { key: "all", label: t("tasks.kpiTotal"), value: kpis.total },
          { key: "SUCCESS", label: t("tasks.kpiSuccess"), value: kpis.success },
          { key: "RUNNING", label: t("tasks.kpiRunning"), value: kpis.running },
          { key: "FAILURE", label: t("tasks.kpiFailure"), value: kpis.failure },
        ].map((item) => {
          const isActive = filter === item.key;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => setFilter(item.key as FilterTab)}
              aria-pressed={isActive}
              className={clsx(
                "flex min-h-9 items-center gap-2 rounded-md px-3 text-sm font-medium transition-colors duration-fast",
                isActive ? "bg-brand/[0.075] text-ink" : "text-ink-muted hover:bg-ink/[0.035] hover:text-ink",
              )}
            >
              <span>{item.label}</span>
              <span className="tabular-nums text-xs text-ink-tertiary">{item.value}</span>
            </button>
          );
        })}
      </div>

      <Modal open={Boolean(deleteConfirmId)} onClose={() => setDeleteConfirmId(null)} title={t("tasks.deleteTask")} footer={
        <>
          <Button variant="outlined" onClick={() => setDeleteConfirmId(null)}>{t("common.cancel")}</Button>
          <Button variant="danger" onClick={() => deleteConfirmId && void onDeleteTask(deleteConfirmId)}>{t("common.delete")}</Button>
        </>
      }>
        <p className="text-base text-ink-secondary">{t("tasks.confirmDeleteTask")}</p>
      </Modal>

      <Modal
        open={Boolean(reviewAction)}
        onClose={() => {
          if (!reviewSubmitting) {
            setReviewAction(null);
            setReviewNote("");
          }
        }}
        title={reviewAction ? reviewCopy[reviewAction] : reviewCopy.title}
        footer={
          <>
            <Button
              variant="outlined"
              disabled={reviewSubmitting}
              onClick={() => {
                setReviewAction(null);
                setReviewNote("");
              }}
            >
              {t("common.cancel")}
            </Button>
            <Button
              variant={reviewAction === "reject" ? "danger" : "primary"}
              loading={reviewSubmitting}
              disabled={!reviewNote.trim()}
              onClick={() => reviewAction && void submitReview(reviewAction, reviewNote)}
            >
              {reviewAction ? reviewCopy[reviewAction] : t("common.confirm")}
            </Button>
          </>
        }
      >
        <label className="block">
          <span className="text-sm font-semibold text-ink">
            {reviewCopy.noteLabel}
          </span>
          <textarea
            value={reviewNote}
            onChange={(event) => setReviewNote(event.target.value)}
            className="mt-2 min-h-[112px] w-full resize-none rounded-lg border border-line bg-surface px-3 py-2.5 text-base leading-6 text-ink outline-none transition-colors placeholder:text-ink-muted focus:border-brand focus:ring-2 focus:ring-brand/20"
            placeholder={reviewCopy.notePlaceholder}
            maxLength={2000}
          />
        </label>
        {!reviewNote.trim() && (
          <p className="mt-2 text-xs font-medium text-warning">
            {reviewCopy.noteRequired}
          </p>
        )}
      </Modal>

      {/* ── Search Bar ── */}
      <div className="flex flex-wrap items-center justify-end w-full">
        {/* Search Input with properly aligned Icon (pis) */}
        <div className="relative w-full md:w-80 shrink-0 group">
          <input
            aria-label={t("tasks.searchPlaceholder")}
            placeholder={t("tasks.searchPlaceholder")}
            className="smx-input w-full ps-10 pe-3 text-sm font-medium"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
            <svg className="absolute start-4 top-1/2 -translate-y-1/2 h-5 w-5 text-ink-muted group-focus-within:text-brand transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      {/* ── Dynamic Master-Detail Layout Wrapper ── */}
      <div className={clsx(
        "grid min-h-0 min-w-0 w-full flex-1 gap-0",
        selectedTaskId
          ? "grid-cols-1 lg:grid-cols-[minmax(220px,0.32fr)_minmax(0,0.68fr)] xl:grid-cols-[minmax(240px,0.28fr)_minmax(0,0.72fr)]"
          : "grid-cols-1"
      )}>

        {/* Master: Data Table */}
        <div className="flex min-h-0 min-w-0 flex-col overflow-hidden bg-surface">
          <div className="min-h-0 flex-1 overflow-auto">
            {!loading && filtered.length === 0 ? (
              <div className="flex w-full flex-col items-center justify-center px-6 py-24 text-center">
                <svg className="mb-4 h-16 w-16 text-ink-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                </svg>
                <p className="text-body-lg font-semibold text-ink-muted">{t("tasks.noTasks")}</p>
              </div>
            ) : (
            <table
              className={clsx(
                "w-full text-start border-collapse",
                selectedTaskId ? "table-fixed" : "min-w-[640px]"
              )}
            >
              <thead className="sticky top-0 z-10 border-b border-line bg-[rgb(var(--bg-secondary))]">
                <tr className="text-xs font-semibold text-ink-muted">
                  <th className={clsx("text-start font-bold", selectedTaskId ? "w-auto px-4 py-4" : "w-1/2 px-6 py-5")}>
                    {t("tasks.topic")}
                  </th>
                  <th className={clsx("text-start font-bold", selectedTaskId ? "w-28 px-4 py-4" : "w-1/4 px-6 py-5")}>
                    {t("tasks.status")}
                  </th>
                  {!selectedTaskId && (
                    <>
                      <th className="w-1/4 px-6 py-5 text-start font-bold">{t("tasks.created")}</th>
                      <th className="sr-only w-16 px-6 py-5 text-end font-bold">{t("users.action")}</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {loading ? (
                  [1, 2, 3, 4, 5].map((i) => (
                    <tr key={i} className="animate-pulse">
                      <td className={selectedTaskId ? "px-4 py-4" : "px-6 py-5"}>
                        <div className="h-5 bg-surface-tertiary rounded-md w-3/4 mb-2"></div>
                        <div className="h-3 bg-surface-alt rounded-md w-1/3"></div>
                      </td>
                      <td className={selectedTaskId ? "px-4 py-4" : "px-6 py-5"}>
                        <div className="h-7 w-20 max-w-full rounded-full bg-surface-tertiary"></div>
                      </td>
                      {!selectedTaskId && (
                        <>
                          <td className="px-6 py-5">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 rounded-full bg-surface-tertiary"></div>
                              <div className="h-4 w-20 bg-surface-tertiary rounded-md"></div>
                            </div>
                          </td>
                          <td className="px-6 py-5 text-end"><div className="h-8 w-8 bg-surface-tertiary rounded-full ms-auto"></div></td>
                        </>
                      )}
                    </tr>
                  ))
                ) : (
                  filtered.map((task) => {
                    const isSelected = task.task_id === selectedTaskId;
                    const statusUpper = task.status?.toUpperCase() ?? "";
                    return (
                      <tr
                        key={task.task_id}
                        className={clsx(
                          "border-b border-line  transition-colors duration-200 cursor-pointer",
                          isSelected ? "bg-ink/[0.05]" : "hover:bg-ink/[0.03]"
                        )}
                        tabIndex={0}
                        aria-selected={isSelected}
                        onClick={() => { setSelectedTaskId(task.task_id); setDetailArticle(null); setDetailTab("content"); setContentView("reader"); setWpResult(null); }}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            setSelectedTaskId(task.task_id); setDetailArticle(null); setDetailTab("content"); setContentView("reader"); setWpResult(null);
                          }
                        }}
                      >
                        <td className={selectedTaskId ? "px-4 py-4" : "px-6 py-4"}>
                          <div className="flex min-w-0 items-start gap-2">
                            <div className="flex min-w-0 flex-1 flex-col">
                              <span className={clsx("truncate text-base font-semibold", isSelected ? "text-ink" : "text-ink")}>
                                {task.topic || task.task_name || task.task_id.slice(0, 12)}
                              </span>
                              {selectedTaskId && (
                                <span className="mt-1 truncate text-xs font-medium text-ink-muted">
                                  {formatDate(task.created_at, locale)}
                                </span>
                              )}
                            </div>
                            {selectedTaskId && (
                              <button
                                type="button"
                                onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(task.task_id); }}
                                className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-ink-muted transition-colors hover:bg-danger-subtle hover:text-danger"
                                title={t("common.delete")}
                                aria-label={t("common.delete")}
                              >
                                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                              </button>
                            )}
                          </div>
                        </td>
                        <td className={selectedTaskId ? "px-4 py-4 align-top" : "px-6 py-4"}><StatusBadge status={statusUpper} locale={locale} /></td>
                        {!selectedTaskId && (
                          <>
                            <td className="px-6 py-4 text-sm text-ink-muted font-medium">{formatDate(task.created_at, locale)}</td>
                            <td className="px-6 py-4 text-end">
                              <button
                                type="button"
                                onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(task.task_id); }}
                                className="w-8 h-8 inline-flex items-center justify-center rounded-full text-ink-muted hover:text-danger hover:bg-danger-subtle transition-colors"
                                title={t("common.delete")}
                                aria-label={t("common.delete")}
                              >
                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                              </button>
                            </td>
                          </>
                        )}
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
            )}
          </div>
        </div>

        {/* Detail: Slide-over Context Panel */}
        {selectedTaskId && (
          <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden border-s border-line bg-[rgb(var(--bg-secondary))] lg:sticky lg:top-0 lg:max-h-full">
            <div className="flex items-center justify-between border-b border-line p-5 lg:p-6">
              <h3 className="text-body-lg font-semibold text-ink">{t("tasks.detail")}</h3>
              <div className="flex gap-2">
                {streamActive && <span className="flex items-center gap-1.5 text-xs font-medium text-success"><span className="w-2 h-2 rounded-full bg-success animate-pulse" /> {TASK_COPY[locale].live}</span>}
                <button type="button" onClick={() => setSelectedTaskId(null)} aria-label={t("common.close")} className="flex h-8 w-8 items-center justify-center rounded-md text-ink-muted transition-colors hover:bg-ink/[0.05] hover:text-ink">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" /></svg>
                </button>
              </div>
            </div>

            <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-5 lg:p-6">
              {/* Status Block */}
              {liveStatus ? (
                <div className="space-y-4">
                  <div className="space-y-3 border-t border-line pt-4">
                    <div className="flex items-center justify-between">
                      <StatusBadge status={liveStatus.state} locale={locale} />
                      <button type="button" onClick={() => void navigator.clipboard.writeText(selectedTaskId)} className="text-xs text-ink-muted hover:text-brand font-mono transition-colors active:scale-95 flex items-center gap-1">
                        {t("tasks.copyId")}
                      </button>
                    </div>
                    {liveStatus.status && (
                      <p className="text-base text-ink-muted font-medium leading-relaxed">
                        {localizeTaskResult(liveStatus.status, locale)}
                      </p>
                    )}

                    {/* Progress bar if numerical */}
                    {typeof liveStatus.progress === "number" && liveStatus.progress > 0 && liveStatus.progress < 100 && (
                      <div className="h-1.5 w-full bg-surface-tertiary rounded-full overflow-hidden mt-2">
                        <div className="h-full bg-brand transition-all duration-500 ease-out" style={{ width: `${liveStatus.progress}%` }} />
                      </div>
                    )}
                  </div>

                  {/* Failure Trace */}
                  {liveStatus.state === "FAILURE" && (
                    <div className="rounded-lg border border-danger/25 border-s-4 border-s-danger bg-danger-subtle p-4 border-danger/20 bg-danger/10">
                      <h4 className="text-sm font-semibold text-danger">{TASK_COPY[locale].statuses.FAILURE}</h4>
                      <p className="mt-1 text-xs leading-5 text-danger">{TASK_COPY[locale].failedSummary}</p>
                      {liveStatus.quality_diagnostics && (
                        <section className="mt-4 border-t border-danger/70 pt-3 border-danger/20">
                          <h5 className="text-xs font-semibold text-danger">{TASK_COPY[locale].qualityDiagnostics}</h5>
                          <dl className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                            <DiagnosticItem label={TASK_COPY[locale].actualWordCount} value={formatDiagnosticNumber(liveStatus.quality_diagnostics.actual_word_count, locale)} />
                            <DiagnosticItem label={TASK_COPY[locale].allowedWordRange} value={formatWordRange(liveStatus.quality_diagnostics, locale)} />
                            <DiagnosticItem label={TASK_COPY[locale].headings} value={formatDiagnosticNumber(liveStatus.quality_diagnostics.headings_count, locale)} />
                            <DiagnosticItem label={TASK_COPY[locale].paragraphs} value={formatDiagnosticNumber(liveStatus.quality_diagnostics.paragraphs_count, locale)} />
                            <DiagnosticItem label={TASK_COPY[locale].language} value={formatDiagnosticLanguage(liveStatus.quality_diagnostics.language, locale)} />
                            <DiagnosticItem label={TASK_COPY[locale].regenerationAttempted} value={formatBoolean(liveStatus.quality_diagnostics.regeneration_attempted, locale)} />
                          </dl>
                          {liveStatus.quality_diagnostics.findings?.length ? (
                            <div className="mt-3 border-t border-danger/70 pt-3 border-danger/20">
                              <h6 className="text-xs font-semibold text-danger">{TASK_COPY[locale].findings}</h6>
                              <ul className="mt-2 space-y-1.5 text-xs leading-5 text-danger">
                                {liveStatus.quality_diagnostics.findings.map((finding, index) => (
                                  <li key={`${finding.code ?? "finding"}-${index}`}>
                                    {localizeQualityFinding(finding.code, finding.message, locale)}
                                    {formatQualityFindingActual(finding.code, liveStatus.quality_diagnostics, finding.actual, locale)}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          ) : null}
                        </section>
                      )}
                      <details className="mt-3 text-xs text-danger">
                        <summary className="cursor-pointer font-medium">{TASK_COPY[locale].technicalDetails}</summary>
                        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap font-mono" dir="ltr">
                          {liveStatus.error ?? liveStatus.last_error ?? (t("common.unexpectedError"))}
                        </pre>
                      </details>
                    </div>
                  )}

                  {/* Success Article Payload */}
                  {liveStatus.state === "SUCCESS" && detailArticle && (
                    <div className="space-y-6">
                      {/* Article Metadata Header */}
                      <div className="border-t border-line pt-4">
                        <h4 className="text-lg font-semibold text-ink leading-snug" dir="auto">
                          {detailArticle.title || "\u2014"}
                        </h4>
                        <div className="mt-2.5 flex flex-wrap gap-2">
                          {detailArticle.language && (
                            <span className="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-1 text-xs font-medium text-ink-secondary" dir="auto">
                              {TASK_COPY[locale].language}: {detailArticle.language}
                            </span>
                          )}
                          {detailArticle.primary_keyword && (
                            <span className="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-1 text-xs font-medium text-ink-secondary" dir="auto">
                              {TASK_COPY[locale].keyword}: {detailArticle.primary_keyword}
                            </span>
                          )}
                          {detailArticle.generated_at && (
                            <span className="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-1 text-xs font-medium text-ink-secondary">
                              {TASK_COPY[locale].generatedAt}: {formatDate(detailArticle.generated_at, locale)}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Metric Chips */}
                      <div className="grid grid-cols-3 divide-x divide-line border-y border-line">
                        <div className="px-3 py-4 text-center">
                          <span className="block text-xs font-medium text-ink-muted">{t("tasks.wordCount")}</span>
                          <span className="block text-xl font-bold text-ink mt-1">{detailArticle.word_count ?? "—"}</span>
                        </div>
                        <div className="px-3 py-4 text-center">
                          <span className="block text-xs font-medium text-ink-muted">{t("tasks.qualityScore")}</span>
                          {typeof taskQualityScore === "number" ? (
                            <>
                              <span className={clsx("block text-xl font-bold mt-1", qualityGrade(taskQualityScore, locale).color)}>{taskQualityScore}</span>
                              <span className={clsx("block text-xs font-semibold mt-0.5", qualityGrade(taskQualityScore, locale).color)}>{qualityGrade(taskQualityScore, locale).label}</span>
                            </>
                          ) : (
                            <span className="mt-2 block text-xs font-medium leading-5 text-ink-muted">{TASK_COPY[locale].notRecorded}</span>
                          )}
                        </div>
                        <div className="px-3 py-4 text-center">
                          <span className="block text-xs font-medium text-ink-muted">{t("tasks.cost")}</span>
                          {typeof taskCost === "number" ? (
                            <span className="block text-xl font-bold text-brand mt-1">${taskCost.toFixed(3)}</span>
                          ) : (
                            <span className="mt-2 block text-xs font-medium leading-5 text-ink-muted">{TASK_COPY[locale].notRecorded}</span>
                          )}
                        </div>
                      </div>

                      <ReviewPanel
                        canReview={canReview}
                        reviewState={reviewState}
                        loading={reviewLoading}
                        error={reviewError}
                        copy={reviewCopy}
                        onApprove={() => void submitReview("approve")}
                        onRequestChanges={() => {
                          setReviewAction("request_changes");
                          setReviewNote("");
                        }}
                        onReject={() => {
                          setReviewAction("reject");
                          setReviewNote("");
                        }}
                        submitting={reviewSubmitting}
                      />

                      {/* Inner Sub-Navigation (Segmented) */}
                      <div className="flex w-full border-b border-line">
                        {[
                          { key: "content" as DetailTab, label: t("tasks.contentTab") },
                          { key: "seo" as DetailTab, label: t("tasks.seoTab") },
                          { key: "export" as DetailTab, label: t("tasks.exportTab") },
                          { key: "history" as DetailTab, label: TASK_COPY[locale].history },
                        ].map((tab) => (
                          <button type="button" key={tab.key} onClick={() => setDetailTab(tab.key)} className={clsx("relative flex-1 px-2 py-2.5 text-sm font-medium transition-colors after:absolute after:inset-x-2 after:-bottom-px after:h-px", detailTab === tab.key ? "text-ink after:bg-brand" : "text-ink-muted after:bg-transparent hover:text-ink")}>
                            {tab.label}
                          </button>
                        ))}
                      </div>

                      {/* Content Views */}
                      {detailTab === "content" && (
                        <div className="flex flex-col gap-3">
                          <div className="mb-2 flex flex-wrap gap-2">
                            {(["reader", "raw", "edit"] as ContentView[]).map(cv => (
                              <button type="button" key={cv} onClick={() => setContentView(cv)} className={clsx("border-b px-2 py-1.5 text-xs font-medium transition-colors", contentView === cv ? "border-brand text-ink" : "border-transparent text-ink-muted hover:text-ink")}>
                                {cv === "reader"
                                  ? t("tasks.readerMode")
                                  : cv === "raw"
                                    ? sourceIsHtml
                                      ? TASK_COPY[locale].htmlSource
                                      : TASK_COPY[locale].markdownSource
                                    : t("tasks.editMode")}
                              </button>
                            ))}
                          </div>
                          <div className="rounded-xl border border-line bg-surface">
                            {contentView === "reader" && (
                              <article className="max-w-none whitespace-pre-wrap p-5 font-sans text-base leading-relaxed text-ink" dir={articleDirection}>
                                {toReaderText(sourceContent)}
                              </article>
                            )}
                            {contentView === "raw" && (
                              <pre className="max-h-96 overflow-auto rounded-lg border border-line bg-surface-alt p-4 font-mono text-xs leading-5 text-ink whitespace-pre-wrap select-all" dir={articleDirection}>{sourceContent}</pre>
                            )}
                            {contentView === "edit" && (
                              <div className="relative group">
                                <textarea aria-label={t("tasks.editMode")} className="w-full h-96 p-4 outline-none resize-none bg-transparent font-mono text-sm text-ink-muted leading-relaxed" dir="auto" value={editContent} onChange={(e) => setEditContent(e.target.value)} />
                                <div className="absolute bottom-4 end-4 flex gap-2 opacity-100 transition-opacity">
                                  {editContent !== detailArticle.content && (
                                    <span className="text-xs text-warning flex items-center font-medium bg-warning-subtle px-2 py-1 rounded-md">{TASK_COPY[locale].unsavedChanges}</span>
                                  )}
                                  <Button size="sm" variant="primary" loading={savingEdit} disabled={editContent === detailArticle.content} onClick={() => void handleSaveEdit()}>
                                    {TASK_COPY[locale].save}
                                  </Button>
                                </div>
                              </div>
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
                            <section className="border-t border-line pt-4">
                              <div className="flex flex-wrap items-start justify-between gap-3">
                                <div>
                                  <h4 className="text-base font-bold text-ink">{t("tasks.seoTab")}</h4>
                                  <p className="mt-1 text-xs text-ink-muted">
                                    {qualityLoading
                                      ? TASK_COPY[locale].seoLoading
                                      : qualityMetrics?.readability_grade
                                        ? qualityMetrics.readability_grade
                                        : seoFallback}
                                  </p>
                                </div>
                                <span className="rounded-lg border border-line bg-surface-alt px-3 py-2 text-xl font-bold tabular-nums text-ink">
                                  {formatPercentScore(seoScore)}
                                </span>
                              </div>
                            </section>

                            {qualityError && (
                              <div className="rounded-xl border border-danger/20 bg-danger/10 px-4 py-3 text-sm font-medium text-danger">
                                <p>{TASK_COPY[locale].seoError}</p>
                                <details className="mt-2 text-xs opacity-80">
                                  <summary className="cursor-pointer">{TASK_COPY[locale].technicalDetails}</summary>
                                  <p className="mt-1 break-words" dir="auto">{qualityError}</p>
                                </details>
                              </div>
                            )}

                            {Object.keys(componentScores).length > 0 && (
                              <section className="border-t border-line pt-4">
                                <h5 className="mb-3 text-sm font-bold text-ink">{t("tasks.seoScore")}</h5>
                                <div className="grid gap-2">
                                  {Object.entries(componentScores).map(([key, value]) => (
                                    <div key={key} className="flex items-center justify-between gap-3 rounded-lg bg-surface-alt px-3 py-2">
                                      <span className="text-xs font-medium text-ink-secondary">{humanizeMetricKey(key)}</span>
                                      <span className="text-xs font-bold tabular-nums text-ink">{formatPercentScore(value)}</span>
                                    </div>
                                  ))}
                                </div>
                              </section>
                            )}

                            {checklist.length > 0 && (
                              <section className="border-t border-line pt-4">
                                <h5 className="mb-3 text-sm font-bold text-ink">{t("tasks.seoChecklist")}</h5>
                                <div className="space-y-2">
                                  {checklist.map((item) => (
                                    <div key={item.label} className="flex items-start gap-2 text-xs leading-5 text-ink-secondary">
                                      <span className={clsx("mt-1 h-2 w-2 shrink-0 rounded-full", item.passed ? "bg-success" : "bg-warning")} />
                                      <span>
                                        <span className="font-semibold text-ink">{item.label}</span>
                                        {item.detail ? `: ${item.detail}` : ""}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </section>
                            )}

                            {recommendations.length > 0 && (
                              <section className="border-t border-line pt-4">
                                <h5 className="mb-3 text-sm font-bold text-ink">{t("tasks.recommendations")}</h5>
                                <div className="space-y-2">
                                  {recommendations.map((recommendation) => (
                                    <p key={recommendation} className="rounded-lg bg-warning/10 px-3 py-2 text-xs leading-5 text-warning">
                                      {recommendation}
                                    </p>
                                  ))}
                                </div>
                              </section>
                            )}

                            {!qualityLoading && !qualityError && !hasSeoData && (
                              <div className="rounded-lg border border-line bg-surface p-5 text-center text-sm text-ink-muted">
                                {seoFallback}
                              </div>
                            )}
                          </div>
                        );
                      })()}
                      {detailTab === "export" && (
                        <div className="space-y-4">
                          <div className="border-t border-line pt-4">
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <h4 className="text-base font-bold text-ink">{riskCopy.title}</h4>
                                <p className="mt-1 text-xs text-ink-muted">
                                  {riskLoading
                                    ? riskCopy.loading
                                    : riskAssessment
                                      ? `${riskAssessment.overall_score}/100`
                                      : t("monitoring.statusUnknown")}
                                </p>
                              </div>
                              {riskAssessment && (
                                <span
                                  className={clsx(
                                    "inline-flex h-7 items-center rounded-md px-2.5 text-xs font-bold",
                                    riskAssessment.risk_level === "blocked" || riskAssessment.risk_level === "high"
                                      ? "bg-danger/10 text-danger"
                                      : riskAssessment.risk_level === "medium"
                                        ? "bg-warning/10 text-warning"
                                        : "bg-success/10 text-success"
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
                                  <p key={issue.id} className="text-xs leading-5 text-ink-secondary">
                                    <span className="font-semibold text-ink">{localizeRiskCategory(issue.category, locale)}: </span>
                                    {localizeRiskMessage(issue.message, locale)}
                                  </p>
                                ))}
                              </div>
                            ) : null}
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <Button variant="outlined" onClick={() => downloadContent(detailArticle, "txt")}>
                              <svg className="w-4 h-4 me-2 inline shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                              {t("tasks.downloadTxt")}
                            </Button>
                            <Button
                              variant="outlined"
                              onClick={() => downloadContent(detailArticle, sourceIsHtml ? "html" : "markdown")}
                            >
                              <svg className="w-4 h-4 me-2 inline shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                              {sourceIsHtml ? TASK_COPY[locale].downloadHtml : TASK_COPY[locale].downloadMarkdown}
                            </Button>
                            <Button variant="outlined" onClick={() => void navigator.clipboard.writeText(contentView === "edit" ? editContent : detailArticle.content)} className="col-span-2">
                              <svg className="w-4 h-4 me-2 inline shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" /></svg>
                              {t("tasks.copyContent")}
                            </Button>
                          </div>
                          <div className="border-t border-line pt-5">
                            <h4 className="text-base font-bold text-info mb-3">{t("tasks.wpPublish")}</h4>
                            <div className="flex gap-3">
                              <Button variant="outlined" size="sm" loading={wpPublishing} disabled={riskAssessment?.risk_level === "blocked" || wordpressActionBlocked} onClick={() => void onWpPublish("draft")}>{t("tasks.wpDraft")}</Button>
                              <Button
                                variant={publicPublishBlocked ? "outlined" : "primary"}
                                size="sm"
                                loading={wpPublishing}
                                disabled={publicPublishBlocked}
                                onClick={() => void onWpPublish("publish")}
                                className={publicPublishBlocked ? "border-warning/25 bg-warning-subtle text-warning shadow-none hover:bg-warning-subtle border-warning/20 bg-warning/10 text-warning" : undefined}
                              >
                                {publicPublishLabel}
                              </Button>
                            </div>
                            {publicPublishReason && (
                              <p className="mt-3 text-xs font-medium text-warning">
                                {publicPublishReason}
                              </p>
                            )}
                            {wordpressActionReason && wordpressActionReason !== publicPublishReason && (
                              <p className="mt-2 text-xs font-medium text-warning">
                                {wordpressActionReason}
                              </p>
                            )}
                            {wpResult && <p className={clsx("mt-3 text-xs font-medium", wpResult.includes("error") || wpResult.includes("خطا") ? "text-danger" : "text-info")}>{wpResult}</p>}
                          </div>
                        </div>
                      )}
                      {detailTab === "history" && (
                        <div className="space-y-4">
                          <h4 className="text-base font-bold text-ink">{TASK_COPY[locale].revisionHistory}</h4>
                          {loadingHistory ? (
                            <div className="animate-pulse space-y-3">
                              <div className="h-16 w-full rounded-xl bg-surface-tertiary" />
                              <div className="h-16 w-full rounded-xl bg-surface-tertiary" />
                            </div>
                          ) : !articleHistory?.revisions?.length ? (
                            <p className="px-4 py-10 text-center text-sm text-ink-muted">{TASK_COPY[locale].noRevisions}</p>
                          ) : (
                            <div className="space-y-3">
                              {articleHistory.revisions.map((rev) => (
                                <div key={rev.id} className="border-t border-line pt-4">
                                  <div className="flex justify-between items-start mb-2">
                                    <div className="text-sm font-medium text-ink">
                                      {rev.revision_note || TASK_COPY[locale].manualEdit}
                                    </div>
                                    <div className="text-xs text-ink-muted">
                                      {formatDate(rev.created_at, locale)}
                                    </div>
                                  </div>
                                  <div className="text-xs text-ink-muted mb-3">
                                    {formatDiagnosticNumber(rev.word_count, locale)} {TASK_COPY[locale].wordUnit}
                                  </div>
                                  <details className="text-xs">
                                    <summary className="cursor-pointer text-brand font-medium select-none">{TASK_COPY[locale].viewPastContent}</summary>
                                    <div className="mt-3 max-h-64 overflow-auto rounded-lg bg-surface-alt p-3 font-mono text-xs text-ink-secondary" dir="auto">
                                      {rev.content}
                                    </div>
                                  </details>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-4 animate-pulse">
                  <div className="h-24 w-full rounded-xl bg-surface-tertiary" />
                  <div className="h-64 w-full rounded-xl bg-surface-tertiary" />
                  <p className="text-center text-xs font-medium text-ink-muted">{TASK_COPY[locale].loadingDetail}</p>
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
