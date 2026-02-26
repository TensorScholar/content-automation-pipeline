"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, apiRequest, API_BASE_URL } from "@/lib/api";
import { TaskHistoryItem, TaskStatusResponse, ArticleDetail } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { TabBar } from "@/components/ui/tab-bar";
import { StatusBadge as SharedStatusBadge } from "@/components/ui/status-badge";
import { Modal } from "@/components/ui/modal";

interface TasksPanelProps {
  token: string;
}

type FilterTab = "all" | "SUCCESS" | "FAILURE" | "RUNNING";
type DetailTab = "content" | "seo" | "export";
type ContentView = "reader" | "raw" | "edit";

export function TasksPanel({ token }: TasksPanelProps) {
  const { t } = useI18n();
  const [tasks, setTasks] = useState<TaskHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterTab>("all");
  const [search, setSearch] = useState("");
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [liveStatus, setLiveStatus] = useState<TaskStatusResponse | null>(null);
  const [streamActive, setStreamActive] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // deep view state
  const [detailArticle, setDetailArticle] = useState<ArticleDetail | null>(null);
  const [detailTab, setDetailTab] = useState<DetailTab>("content");
  const [contentView, setContentView] = useState<ContentView>("reader");
  const [editContent, setEditContent] = useState("");
  const [wpPublishing, setWpPublishing] = useState(false);
  const [wpResult, setWpResult] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);

  const loadTasks = useCallback(async () => {
    try {
      const res = await apiRequest<TaskHistoryItem[]>("/content/tasks", { token });
      setTasks(Array.isArray(res) ? res : []);
    } catch {
      setTasks([]);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => { void loadTasks(); }, [loadTasks]);

  // Auto-refresh interval
  useEffect(() => {
    if (!autoRefresh) return;
    const iv = window.setInterval(() => { void loadTasks(); }, 30000);
    return () => window.clearInterval(iv);
  }, [autoRefresh, loadTasks]);

  // SSE streaming for selected task
  useEffect(() => {
    if (!selectedTaskId) { setStreamActive(false); return; }
    const es = new EventSource(`${API_BASE_URL}/content/task/${selectedTaskId}/stream?token=${token}`);
    eventSourceRef.current = es;
    setStreamActive(true);
    es.addEventListener("status", (event) => {
      try {
        const payload = JSON.parse(event.data) as TaskStatusResponse;
        setLiveStatus(payload);
        if (payload.ready) { es.close(); setStreamActive(false); void loadTasks(); }
      } catch { /* ignore parse errors */ }
    });
    es.onerror = () => {
      es.close();
      setStreamActive(false);
      void pollTask(selectedTaskId, token, setLiveStatus);
    };
    return () => { es.close(); setStreamActive(false); };
  }, [selectedTaskId, token, loadTasks]);

  // Load article detail when task is SUCCESS
  useEffect(() => {
    const articleId = liveStatus?.result?.article_id;
    if (!articleId) { setDetailArticle(null); return; }
    const load = async () => {
      try {
        const article = await apiRequest<ArticleDetail>(`/content/${articleId}`, { token });
        setDetailArticle(article);
        setEditContent(article.content ?? "");
      } catch { setDetailArticle(null); }
    };
    void load();
  }, [liveStatus?.result?.article_id, token]);

  // KPI counters
  const kpis = useMemo(() => {
    const total = tasks.length;
    const success = tasks.filter((t) => t.status?.toUpperCase() === "SUCCESS").length;
    const failure = tasks.filter((t) => t.status?.toUpperCase() === "FAILURE" || t.status?.toUpperCase() === "FAILED").length;
    const running = tasks.filter((t) => {
      const s = t.status?.toUpperCase() ?? "";
      return s !== "SUCCESS" && s !== "FAILURE" && s !== "FAILED";
    }).length;
    return { total, success, failure, running };
  }, [tasks]);

  const filtered = useMemo(() => {
    let list = tasks;
    if (filter === "RUNNING") {
      list = list.filter((t) => { const s = t.status?.toUpperCase() ?? ""; return s !== "SUCCESS" && s !== "FAILURE" && s !== "FAILED"; });
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
    setError(null);
    try {
      await apiRequest<void>(`/content/task/${taskId}`, { method: "DELETE", token });
      setMessage(t("tasks.taskDeleted"));
      if (selectedTaskId === taskId) { setSelectedTaskId(null); setLiveStatus(null); }
      await loadTasks();
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : t("common.unexpectedError"));
    }
  };

  const onBulkDownload = async () => {
    const successful = tasks.filter((t) => t.status?.toUpperCase() === "SUCCESS");
    const results: string[] = [];
    for (const task of successful.slice(0, 20)) {
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
    setWpPublishing(true);
    setWpResult(null);
    try {
      await apiRequest(`/content/${detailArticle.id}/publish/wordpress`, {
        method: "POST", token, body: { status }
      });
      setWpResult(t("tasks.wpPublished"));
    } catch (e) {
      setWpResult(e instanceof ApiError ? e.detail : t("tasks.wpPublishError"));
    } finally {
      setWpPublishing(false);
    }
  };

  const filterTabs: Array<{ key: FilterTab; label: string; count: number; colorClass: string }> = [
    { key: "all", label: t("common.all"), count: kpis.total, colorClass: "" },
    { key: "SUCCESS", label: t("common.success"), count: kpis.success, colorClass: "text-success" },
    { key: "FAILURE", label: t("common.failure"), count: kpis.failure, colorClass: "text-danger" },
    { key: "RUNNING", label: t("common.running"), count: kpis.running, colorClass: "text-info" },
  ];

  const inputCls = "w-full rounded-xl border border-border bg-surface-secondary px-3 py-2.5 text-body-md text-ink outline-none focus:border-border-focus transition-colors duration-fast";

  return (
    <section className="animate-fade-in space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-display-lg text-ink">{t("tasks.title")}</h2>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-body-sm text-ink-secondary cursor-pointer select-none">
            <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} className="accent-accent rounded" />
            {t("tasks.autoRefresh")}
          </label>
          <button
            type="button"
            onClick={() => void loadTasks()}
            className="rounded-xl border border-border bg-surface px-4 py-2 text-body-md text-ink transition-colors duration-fast hover:bg-surface-secondary"
          >
            {t("common.refresh")}
          </button>
          {kpis.success > 0 && (
            <button
              type="button"
              onClick={() => void onBulkDownload()}
              className="rounded-xl border border-accent/30 bg-accent-subtle px-4 py-2 text-body-md text-accent font-semibold transition-colors duration-fast hover:bg-accent/10"
            >
              {t("tasks.bulkDownload")}
            </button>
          )}
        </div>
      </div>

      {message && <p className="rounded-xl border border-success/20 bg-success-subtle px-4 py-2 text-body-md text-success">{message}</p>}
      {error && <p className="rounded-xl border border-danger/20 bg-danger-subtle px-4 py-2 text-body-md text-danger">{error}</p>}

      {/* KPI cards */}
      <div className="grid gap-3 grid-cols-2 md:grid-cols-4">
        {([
          { label: t("tasks.kpiTotal"), value: kpis.total, dot: "bg-ink-tertiary" },
          { label: t("tasks.kpiSuccess"), value: kpis.success, dot: "bg-success" },
          { label: t("tasks.kpiFailure"), value: kpis.failure, dot: "bg-danger" },
          { label: t("tasks.kpiRunning"), value: kpis.running, dot: "bg-info animate-pulse-soft" },
        ]).map((card) => (
          <div key={card.label} className="elevated-card p-4">
            <div className="flex items-center gap-2 mb-1">
              <div className={`h-2 w-2 rounded-full ${card.dot}`} />
              <span className="text-body-sm font-semibold text-ink-tertiary uppercase tracking-wider">{card.label}</span>
            </div>
            <span className="text-display-xl text-ink">{card.value}</span>
          </div>
        ))}
      </div>

      {/* Delete Confirmation */}
      {deleteConfirmId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 backdrop-blur-sm animate-fade-in" onClick={() => setDeleteConfirmId(null)}>
          <div className="glass-card mx-4 w-full max-w-md p-6 animate-scale-in" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-heading-md text-ink text-center">{t("tasks.deleteTask")}</h3>
            <p className="mt-2 text-body-md text-ink-tertiary text-center">{t("tasks.confirmDeleteTask")}</p>
            <div className="mt-6 flex gap-3 justify-center">
              <button type="button" onClick={() => setDeleteConfirmId(null)} className="rounded-xl border border-border px-5 py-2.5 text-body-md text-ink-secondary hover:bg-surface-tertiary transition-colors duration-fast">{t("common.cancel")}</button>
              <button type="button" onClick={() => void onDeleteTask(deleteConfirmId)} className="rounded-xl bg-danger px-5 py-2.5 text-body-md font-semibold text-ink-inverse hover:brightness-110 transition-colors duration-fast">{t("common.delete")}</button>
            </div>
          </div>
        </div>
      )}

      {/* Search + filter tabs */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          placeholder={t("tasks.searchPlaceholder")}
          className={`max-w-xs ${inputCls}`}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <div className="inline-flex rounded-full border border-border bg-surface-secondary p-1 gap-0.5">
          {filterTabs.map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => setFilter(tab.key)}
              className={`rounded-full px-3 py-1.5 text-body-sm font-semibold transition-all duration-fast ease-apple ${filter === tab.key
                ? "bg-accent text-ink-inverse shadow-elevation-1"
                : "text-ink-tertiary hover:bg-surface-tertiary"
                }`}
            >
              {tab.label} ({tab.count})
            </button>
          ))}
        </div>
      </div>

      {/* Main area: list + detail panel */}
      <div className="grid gap-5 xl:grid-cols-[1fr_420px]">
        {/* Task list */}
        <div className="elevated-card overflow-hidden">
          <div className="overflow-auto max-h-[60vh]">
            <table className="w-full text-start">
              <thead className="sticky top-0 bg-surface-secondary/95 backdrop-blur-sm z-10 border-b border-border">
                <tr className="text-body-sm font-semibold text-ink-tertiary uppercase tracking-wider">
                  <th className="px-4 py-3 text-start">{t("tasks.topic")}</th>
                  <th className="px-4 py-3 text-start">{t("tasks.status")}</th>
                  <th className="px-4 py-3 text-start">{t("tasks.created")}</th>
                  <th className="px-4 py-3 text-end">{t("users.action")}</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <tr key={i}><td colSpan={4} className="px-4 py-3"><div className="skeleton h-4 w-full" /></td></tr>
                  ))
                ) : filtered.length === 0 ? (
                  <tr><td colSpan={4} className="px-4 py-8 text-center text-body-md text-ink-tertiary">{t("tasks.noTasks")}</td></tr>
                ) : (
                  filtered.map((task) => {
                    const isSelected = task.task_id === selectedTaskId;
                    const statusUpper = task.status?.toUpperCase() ?? "";
                    return (
                      <tr
                        key={task.task_id}
                        className={`border-b border-border transition-colors duration-fast cursor-pointer ${isSelected ? "bg-accent-subtle" : "hover:bg-surface-secondary"}`}
                        onClick={() => { setSelectedTaskId(task.task_id); setDetailArticle(null); setDetailTab("content"); setContentView("reader"); setWpResult(null); }}
                      >
                        <td className="px-4 py-3 text-body-md text-ink truncate max-w-[200px]">{task.topic || task.task_name || task.task_id.slice(0, 12)}</td>
                        <td className="px-4 py-3">
                          <StatusBadge status={statusUpper} />
                        </td>
                        <td className="px-4 py-3 text-body-sm text-ink-tertiary">{formatDate(task.created_at)}</td>
                        <td className="px-4 py-3 text-end">
                          <button
                            type="button"
                            onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(task.task_id); }}
                            className="rounded-lg border border-danger/20 px-2.5 py-1 text-body-sm text-danger transition-colors duration-fast hover:bg-danger-subtle"
                          >
                            {t("common.delete")}
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

        {/* ── Detail side panel ── */}
        <div className="glass-card p-5 flex flex-col gap-4 max-h-[80vh] overflow-y-auto">
          {!selectedTaskId && (
            <p className="text-body-md text-ink-tertiary text-center py-8">{t("tasks.selectTask")}</p>
          )}
          {selectedTaskId && (
            <>
              {/* Header */}
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-heading-sm text-ink truncate">{t("tasks.detail")}</h3>
                <div className="flex items-center gap-2">
                  {streamActive && <span className="text-body-sm text-success animate-pulse-soft">{t("tasks.streaming")}</span>}
                  <button type="button" onClick={() => void pollTask(selectedTaskId, token, setLiveStatus)} className="rounded-lg border border-border px-2.5 py-1 text-body-sm text-ink-secondary hover:bg-surface-tertiary transition-colors duration-fast">{t("tasks.pollNow")}</button>
                  <button type="button" onClick={() => void navigator.clipboard.writeText(selectedTaskId)} className="rounded-lg border border-border px-2.5 py-1 text-body-sm text-ink-secondary hover:bg-surface-tertiary transition-colors duration-fast">{t("tasks.copyId")}</button>
                </div>
              </div>
              <code className="block rounded-lg bg-surface-tertiary px-3 py-2 font-mono text-body-sm text-ink-secondary break-all">{selectedTaskId}</code>

              {/* Status */}
              {liveStatus && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <StatusBadge status={liveStatus.state} />
                    {typeof liveStatus.progress === "number" && liveStatus.progress > 0 && (
                      <span className="text-body-sm text-ink-tertiary">{t("tasks.progress")}: {liveStatus.progress}%</span>
                    )}
                  </div>
                  {liveStatus.status && <p className="text-body-md text-ink-secondary">{liveStatus.status}</p>}

                  {/* ── SUCCESS: show article deep view ── */}
                  {liveStatus.state === "SUCCESS" && detailArticle && (
                    <>
                      {/* Metadata row */}
                      <div className="grid grid-cols-3 gap-2">
                        <div className="rounded-lg bg-surface-tertiary px-3 py-2 text-center">
                          <p className="text-body-sm text-ink-tertiary">{t("tasks.wordCount")}</p>
                          <p className="text-body-md font-semibold text-ink">{detailArticle.word_count ?? "—"}</p>
                        </div>
                        <div className="rounded-lg bg-surface-tertiary px-3 py-2 text-center">
                          <p className="text-body-sm text-ink-tertiary">{t("tasks.qualityScore")}</p>
                          <p className="text-body-md font-semibold text-ink">{detailArticle.quality_score ?? "—"}</p>
                        </div>
                        <div className="rounded-lg bg-surface-tertiary px-3 py-2 text-center">
                          <p className="text-body-sm text-ink-tertiary">{t("tasks.cost")}</p>
                          <p className="text-body-md font-semibold text-ink">{detailArticle.cost_usd ? `$${detailArticle.cost_usd.toFixed(3)}` : "—"}</p>
                        </div>
                      </div>

                      {/* Deep view tabs */}
                      <div className="inline-flex rounded-full border border-border bg-surface-secondary p-1 gap-0.5">
                        {([
                          { key: "content" as DetailTab, label: t("tasks.contentTab") },
                          { key: "seo" as DetailTab, label: t("tasks.seoTab") },
                          { key: "export" as DetailTab, label: t("tasks.exportTab") },
                        ]).map((tab) => (
                          <button key={tab.key} type="button" onClick={() => setDetailTab(tab.key)}
                            className={`rounded-full px-3 py-1 text-body-sm font-semibold transition-all duration-fast ease-apple ${detailTab === tab.key ? "bg-accent text-ink-inverse shadow-elevation-1" : "text-ink-tertiary hover:bg-surface-tertiary"}`}
                          >{tab.label}</button>
                        ))}
                      </div>

                      {/* Content tab */}
                      {detailTab === "content" && (
                        <div className="space-y-3">
                          <div className="flex gap-2">
                            {(["reader", "raw", "edit"] as ContentView[]).map((cv) => (
                              <button key={cv} type="button" onClick={() => setContentView(cv)}
                                className={`rounded-lg px-3 py-1 text-body-sm transition-colors duration-fast ${contentView === cv ? "bg-surface-tertiary text-ink font-semibold" : "text-ink-tertiary hover:text-ink"}`}
                              >{cv === "reader" ? t("tasks.readerMode") : cv === "raw" ? t("tasks.rawHtml") : t("tasks.editMode")}</button>
                            ))}
                          </div>
                          {contentView === "reader" && (
                            <article className="prose prose-sm max-w-none rounded-xl bg-surface p-4 border border-border text-body-md text-ink leading-relaxed" dangerouslySetInnerHTML={{ __html: detailArticle.html_content ?? detailArticle.content }} />
                          )}
                          {contentView === "raw" && (
                            <pre className="max-h-64 overflow-auto rounded-xl bg-ink p-4 text-body-sm text-ink-inverse font-mono">{detailArticle.html_content ?? detailArticle.content}</pre>
                          )}
                          {contentView === "edit" && (
                            <textarea className={`${inputCls} h-64 resize-y font-mono text-body-sm`} value={editContent} onChange={(e) => setEditContent(e.target.value)} />
                          )}
                        </div>
                      )}

                      {/* SEO tab */}
                      {detailTab === "seo" && (
                        <div className="space-y-3">
                          {detailArticle.seo_analysis ? (
                            <>
                              <div className="flex items-center gap-3">
                                <span className="text-body-sm font-semibold text-ink-tertiary">{t("tasks.seoScore")}</span>
                                <SeoScoreBadge score={detailArticle.seo_analysis.score} />
                              </div>
                              {detailArticle.seo_analysis.checklist && detailArticle.seo_analysis.checklist.length > 0 && (
                                <div className="space-y-1.5">
                                  <h4 className="text-body-sm font-semibold text-ink-tertiary">{t("tasks.seoChecklist")}</h4>
                                  {detailArticle.seo_analysis.checklist.map((item, i) => (
                                    <div key={i} className={`flex items-center gap-2 rounded-lg border px-3 py-2 ${item.passed ? "border-success/20 bg-success-subtle" : "border-danger/20 bg-danger-subtle"}`}>
                                      <span className={`text-body-md ${item.passed ? "text-success" : "text-danger"}`}>{item.passed ? "✓" : "✗"}</span>
                                      <span className="text-body-sm text-ink">{item.label}</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                              {detailArticle.seo_analysis.recommendations && detailArticle.seo_analysis.recommendations.length > 0 && (
                                <div className="space-y-1.5">
                                  <h4 className="text-body-sm font-semibold text-ink-tertiary">{t("tasks.recommendations")}</h4>
                                  <ul className="space-y-1 ps-4 list-disc list-outside text-body-sm text-ink-secondary">
                                    {detailArticle.seo_analysis.recommendations.map((rec, i) => <li key={i}>{rec}</li>)}
                                  </ul>
                                </div>
                              )}
                            </>
                          ) : (
                            <p className="text-body-md text-ink-tertiary">{t("common.noData")}</p>
                          )}
                        </div>
                      )}

                      {/* Export tab */}
                      {detailTab === "export" && (
                        <div className="space-y-4">
                          <div className="flex flex-wrap gap-2">
                            <button type="button" onClick={() => downloadContent(detailArticle, "txt")} className="rounded-xl border border-border px-4 py-2 text-body-sm text-ink-secondary hover:bg-surface-tertiary transition-colors duration-fast">{t("tasks.downloadTxt")}</button>
                            <button type="button" onClick={() => downloadContent(detailArticle, "html")} className="rounded-xl border border-border px-4 py-2 text-body-sm text-ink-secondary hover:bg-surface-tertiary transition-colors duration-fast">{t("tasks.downloadHtml")}</button>
                            <button type="button" onClick={() => void navigator.clipboard.writeText(contentView === "edit" ? editContent : detailArticle.content)} className="rounded-xl border border-border px-4 py-2 text-body-sm text-ink-secondary hover:bg-surface-tertiary transition-colors duration-fast">{t("tasks.copyContent")}</button>
                          </div>
                          {/* WordPress publish */}
                          <div className="rounded-xl border border-border bg-surface p-4 space-y-3">
                            <h4 className="text-body-md font-semibold text-ink">{t("tasks.wpPublish")}</h4>
                            <div className="flex gap-2">
                              <button type="button" disabled={wpPublishing} onClick={() => void onWpPublish("draft")} className="rounded-xl border border-border px-4 py-2 text-body-sm text-ink-secondary hover:bg-surface-tertiary transition-colors duration-fast disabled:opacity-50">{t("tasks.wpDraft")}</button>
                              <button type="button" disabled={wpPublishing} onClick={() => void onWpPublish("publish")} className="rounded-xl bg-accent px-4 py-2 text-body-sm font-semibold text-ink-inverse hover:bg-accent-hover transition-colors duration-fast disabled:opacity-50">{t("tasks.wpLive")}</button>
                            </div>
                            {wpResult && <p className={`text-body-sm ${wpResult.includes("error") || wpResult.includes("خطا") ? "text-danger" : "text-success"}`}>{wpResult}</p>}
                          </div>
                        </div>
                      )}
                    </>
                  )}

                  {/* ── FAILURE state ── */}
                  {liveStatus.state === "FAILURE" && (
                    <div className="space-y-2">
                      <h4 className="text-body-sm font-semibold text-danger">{t("tasks.failureTrace")}</h4>
                      <pre className="max-h-40 overflow-auto rounded-xl bg-danger-subtle border border-danger/20 px-4 py-3 font-mono text-body-sm text-danger">
                        {liveStatus.error ?? liveStatus.last_error ?? t("common.unexpectedError")}
                      </pre>
                    </div>
                  )}

                  {/* ── Pending / In progress ── */}
                  {liveStatus.state !== "SUCCESS" && liveStatus.state !== "FAILURE" && (
                    <div className="flex flex-col items-center gap-3 py-4">
                      <span className="h-8 w-8 rounded-full border-2 border-accent/30 border-t-accent animate-spin" />
                      <p className="text-body-md text-ink-tertiary">{t("tasks.pendingNotice")}</p>
                    </div>
                  )}
                </div>
              )}
              {!liveStatus && (
                <div className="flex flex-col items-center gap-3 py-8">
                  <div className="skeleton h-4 w-3/4" /><div className="skeleton h-4 w-1/2" /><div className="skeleton h-4 w-2/3" />
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}

/* ─── Helper Components ─── */

function StatusBadge({ status }: { status: string }) {
  const s = status.toUpperCase();
  const cls = s === "SUCCESS"
    ? "bg-success-subtle text-success border-success/20"
    : s === "FAILURE" || s === "FAILED"
      ? "bg-danger-subtle text-danger border-danger/20"
      : "bg-info-subtle text-info border-info/20 animate-pulse-soft";
  return <span className={`inline-block rounded-full border px-2.5 py-0.5 text-body-sm font-semibold ${cls}`}>{status}</span>;
}

function SeoScoreBadge({ score }: { score: number | undefined }) {
  if (score === undefined) return <span className="text-body-md text-ink-tertiary">—</span>;
  const cls = score >= 80 ? "text-success" : score >= 50 ? "text-warning" : "text-danger";
  return <span className={`text-display-lg font-bold ${cls}`}>{score}/100</span>;
}

/* ─── Helper Functions ─── */

async function pollTask(taskId: string, token: string, setter: (p: TaskStatusResponse) => void) {
  try {
    const payload = await apiRequest<TaskStatusResponse>(`/content/task/${taskId}`, { token });
    setter(payload);
  } catch { /* ignore */ }
}

function formatDate(d?: string): string {
  if (!d) return "—";
  try { return new Date(d).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
  catch { return d; }
}

function downloadContent(article: ArticleDetail, format: "txt" | "html") {
  const content = format === "html" ? (article.html_content ?? article.content) : article.content;
  const blob = new Blob([content], { type: format === "html" ? "text/html;charset=utf-8" : "text/plain;charset=utf-8" });
  downloadBlob(blob, `${article.title || "article"}.${format}`);
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
