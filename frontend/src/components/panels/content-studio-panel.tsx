"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { ApiError, apiRequest } from "@/lib/api";
import { TaskStatusResponse } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { TabBar } from "@/components/ui/tab-bar";
import { StatusBadge as SharedStatusBadge } from "@/components/ui/status-badge";

interface ContentStudioPanelProps {
  token: string;
  selectedProjectId: string | null;
}

interface JsonLdResponse {
  article_id: string;
  schema: Record<string, unknown>;
}

interface HtmlExportResponse {
  article_id: string;
  title: string;
  schema: Record<string, unknown>;
  html: string;
}

interface BatchStatusResponse {
  batch_id: string;
  total: number;
  completed: number;
  failed: number;
  status: string;
  tasks?: Array<{ topic: string; task_id?: string; status: string }>;
}

type StudioTab = "generate" | "bulk" | "social" | "schema";

const TONE_OPTIONS = [
  { value: "professional", fa: "حرفه‌ای", ar: "مهني", en: "Professional" },
  { value: "friendly", fa: "دوستانه", ar: "ودي", en: "Friendly" },
  { value: "formal", fa: "رسمی", ar: "رسمي", en: "Formal" },
  { value: "persuasive", fa: "متقاعدکننده", ar: "مقنع", en: "Persuasive" },
  { value: "educational", fa: "آموزشی", ar: "تعليمي", en: "Educational" },
];

const STRUCTURE_OPTIONS = [
  { value: "standard", fa: "استاندارد", ar: "قياسي", en: "Standard" },
  { value: "listicle", fa: "فهرستی", ar: "قائمة", en: "Listicle" },
  { value: "howto", fa: "آموزشی (How-to)", ar: "كيفية (How-to)", en: "How-to Guide" },
  { value: "comparison", fa: "مقایسه‌ای", ar: "مقارنة", en: "Comparison" },
  { value: "pillar", fa: "ستونی (Pillar)", ar: "ركيزة (Pillar)", en: "Pillar Page" },
];

const POV_OPTIONS = [
  { value: "first_person", fa: "اول شخص", ar: "ضمير المتكلم", en: "First person" },
  { value: "second_person", fa: "دوم شخص", ar: "ضمير المخاطب", en: "Second person" },
  { value: "third_person", fa: "سوم شخص", ar: "ضمير الغائب", en: "Third person" },
];

const AUDIENCE_OPTIONS = [
  { value: "general", fa: "عمومی", ar: "عام", en: "General" },
  { value: "technical", fa: "فنی و تخصصی", ar: "تقني ومتخصص", en: "Technical" },
  { value: "beginner", fa: "مبتدی", ar: "مبتدئ", en: "Beginner" },
  { value: "business", fa: "مدیران و کسب‌وکار", ar: "رجال الأعمال", en: "Business professionals" },
];

export function ContentStudioPanel({ token, selectedProjectId }: ContentStudioPanelProps) {
  const { t, locale } = useI18n();
  const [activeTab, setActiveTab] = useState<StudioTab>("generate");
  const [topic, setTopic] = useState("");
  const [keyword, setKeyword] = useState("");
  const [language, setLanguage] = useState<"fa" | "ar" | "en">("fa");
  const [competitorUrl, setCompetitorUrl] = useState("");
  const [extraInstructions, setExtraInstructions] = useState("");
  const [sourceUrls, setSourceUrls] = useState("");
  const [tone, setTone] = useState("professional");
  const [customTone, setCustomTone] = useState("");
  const [structure, setStructure] = useState("standard");
  const [customStructure, setCustomStructure] = useState("");
  const [pov, setPov] = useState("second_person");
  const [audience, setAudience] = useState("general");
  const [customAudience, setCustomAudience] = useState("");
  const [wordCountMin, setWordCountMin] = useState("");
  const [wordCountMax, setWordCountMax] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<TaskStatusResponse | null>(null);
  const [socialStatus, setSocialStatus] = useState<TaskStatusResponse | null>(null);
  const [jsonld, setJsonld] = useState<Record<string, unknown> | null>(null);
  const [exportHtml, setExportHtml] = useState<string>("");

  // Bulk queue state
  const [bulkTopics, setBulkTopics] = useState("");
  const [bulkKeyword, setBulkKeyword] = useState("");
  const [bulkLanguage, setBulkLanguage] = useState<"fa" | "ar" | "en">("fa");
  const [bulkSubmitting, setBulkSubmitting] = useState(false);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [batchStatus, setBatchStatus] = useState<BatchStatusResponse | null>(null);

  const labelFor = (opts: Array<{ value: string; fa: string; ar: string; en: string }>, val: string) => {
    const found = opts.find((o) => o.value === val);
    if (!found) return val;
    return locale === "fa" ? found.fa : locale === "ar" ? found.ar : found.en;
  };

  const articleId = useMemo(() => taskStatus?.result?.article_id ?? null, [taskStatus]);
  const socialTaskId = useMemo(() => {
    if (!taskStatus?.result?.social_task_id) return null;
    return String(taskStatus.result.social_task_id);
  }, [taskStatus]);

  useEffect(() => {
    setTaskStatus(null);
    setSocialStatus(null);
    setJsonld(null);
    setExportHtml("");
    setMessage(null);
    setError(null);
    setBatchId(null);
    setBatchStatus(null);
  }, [selectedProjectId]);

  // Poll main task
  useEffect(() => {
    if (!taskStatus?.task_id || taskStatus.ready) return;
    const interval = window.setInterval(() => {
      void refreshTask(taskStatus.task_id, setTaskStatus, token);
    }, 4000);
    return () => window.clearInterval(interval);
  }, [taskStatus?.ready, taskStatus?.task_id, token]);

  // Poll social task
  useEffect(() => {
    if (!socialTaskId) return;
    const poll = async () => { await refreshTask(socialTaskId, setSocialStatus, token); };
    void poll();
    const interval = window.setInterval(() => { void poll(); }, 4000);
    return () => window.clearInterval(interval);
  }, [socialTaskId, token]);

  // Load schema + export HTML when article is ready
  useEffect(() => {
    if (!articleId) return;
    const loadSchema = async () => {
      try {
        const [schemaPayload, htmlPayload] = await Promise.all([
          apiRequest<JsonLdResponse>(`/content/${articleId}/schema/jsonld`, { token }),
          apiRequest<HtmlExportResponse>(`/content/${articleId}/export/html`, { token })
        ]);
        setJsonld(schemaPayload.schema);
        setExportHtml(htmlPayload.html);
      } catch {
        setJsonld(null);
        setExportHtml("");
      }
    };
    void loadSchema();
  }, [articleId, token]);

  // Poll batch status
  useEffect(() => {
    if (!batchId) return;
    const poll = async () => {
      try {
        const status = await apiRequest<BatchStatusResponse>(`/content/batch/${batchId}/status`, { token });
        setBatchStatus(status);
        if (status.status === "completed" || status.status === "failed") {
          setBatchId(null);
        }
      } catch { /* ignore */ }
    };
    void poll();
    const interval = window.setInterval(() => { void poll(); }, 5000);
    return () => window.clearInterval(interval);
  }, [batchId, token]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedProjectId) {
      setError(t("studio.selectProjectFirst"));
      return;
    }
    setSubmitting(true);
    setMessage(null);
    setError(null);
    setTaskStatus(null);
    setSocialStatus(null);

    const resolvedTone = tone === "__custom__" ? customTone.trim() : labelFor(TONE_OPTIONS, tone);
    const resolvedStructure = structure === "__custom__" ? customStructure.trim() : labelFor(STRUCTURE_OPTIONS, structure);
    const resolvedAudience = audience === "__custom__" ? customAudience.trim() : labelFor(AUDIENCE_OPTIONS, audience);
    const backendLanguage = language === "en" ? "en" : "fa";
    const instructions = [
      extraInstructions.trim(),
      competitorUrl.trim() ? `Competitor URL (Skyscraper mode): ${competitorUrl.trim()}` : "",
      sourceUrls.trim() ? `Source URLs:\n${sourceUrls.trim()}` : "",
      `Tone: ${resolvedTone}`,
      `Structure: ${resolvedStructure}`,
      `Point of view: ${labelFor(POV_OPTIONS, pov)}`,
      `Target audience: ${resolvedAudience}`,
      "After article generation, preserve social repurposing outputs for LinkedIn/Twitter/Instagram.",
      "Ensure schema-friendly structure for FAQ/HowTo rich snippets."
    ].filter((line) => line.length > 0).join("\n");

    try {
      const payload = await apiRequest<{ task_id: string; status: string }>(
        "/content/generate/async",
        {
          method: "POST",
          token,
          body: {
            project_id: selectedProjectId,
            topic: topic.trim(),
            priority: "high",
            primary_keyword: keyword.trim(),
            custom_instructions: instructions,
            language: backendLanguage,
            temperature,
            ...(wordCountMin || wordCountMax ? {
              word_count_min: wordCountMin ? Number(wordCountMin) : undefined,
              word_count_max: wordCountMax ? Number(wordCountMax) : undefined,
            } : {}),
            seo_settings: {
              auto_schema: true,
              competitor_takedown: competitorUrl.trim().length > 0
            }
          }
        }
      );
      setTaskStatus({
        task_id: payload.task_id,
        state: "PENDING",
        ready: false,
        status: payload.status
      });
      setMessage(t("studio.taskQueued").replace("{taskId}", payload.task_id));
      setActiveTab("generate");
    } catch (e) {
      setError(extractError(e));
    } finally {
      setSubmitting(false);
    }
  };

  const onSubmitBatch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedProjectId) {
      setError(t("studio.selectProjectFirst"));
      return;
    }
    const topics = bulkTopics.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);
    if (topics.length === 0 || topics.length > 20) {
      setError(t("studio.maxTopics"));
      return;
    }
    setBulkSubmitting(true);
    setError(null);
    setMessage(null);
    setBatchStatus(null);
    const backendLanguage = bulkLanguage === "en" ? "en" : "fa";
    try {
      const payload = await apiRequest<{ batch_id: string }>("/content/generate/batch", {
        method: "POST",
        token,
        body: {
          project_id: selectedProjectId,
          topics,
          shared_keyword: bulkKeyword.trim() || undefined,
          language: backendLanguage,
        }
      });
      setBatchId(payload.batch_id);
      setMessage(t("studio.batchProgress"));
    } catch (e) {
      setError(extractError(e));
    } finally {
      setBulkSubmitting(false);
    }
  };

  const scanning = submitting && competitorUrl.trim().length > 0;
  const inputCls = "w-full rounded-xl border border-border bg-surface-secondary px-3 py-2.5 text-body-md text-ink outline-none focus:border-border-focus transition-colors duration-fast";

  const tabEntries: Array<{ key: StudioTab; label: string }> = [
    { key: "generate", label: t("studio.generateTab") },
    { key: "bulk", label: t("studio.bulkTab") },
    { key: "social", label: t("studio.socialTab") },
    { key: "schema", label: t("studio.schemaTab") },
  ];

  return (
    <section className="animate-fade-in space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-display-lg text-ink">{t("studio.title")}</h2>
        <div className="inline-flex rounded-full border border-border bg-surface-secondary p-1 gap-0.5">
          {tabEntries.map((entry) => (
            <button
              key={entry.key}
              type="button"
              onClick={() => setActiveTab(entry.key)}
              className={`rounded-full px-4 py-1.5 text-body-sm font-semibold transition-all duration-fast ease-apple ${activeTab === entry.key
                ? "bg-accent text-ink-inverse shadow-elevation-1"
                : "text-ink-tertiary hover:bg-surface-tertiary hover:text-ink-secondary"
                }`}
            >
              {entry.label}
            </button>
          ))}
        </div>
      </div>

      {message && (
        <p className="rounded-xl border border-success/20 bg-success-subtle px-4 py-2 text-body-md text-success">{message}</p>
      )}
      {error && (
        <p className="rounded-xl border border-danger/20 bg-danger-subtle px-4 py-2 text-body-md text-danger">{error}</p>
      )}

      {/* ── Generate Tab ── */}
      {activeTab === "generate" && (
        <div className="grid gap-5 xl:grid-cols-[1fr_380px]">
          <article className="glass-card p-5">
            <h3 className="text-heading-sm text-ink mb-4">{t("studio.contentGeneration")}</h3>
            <form className="space-y-3" onSubmit={onSubmit}>
              <input required placeholder={t("studio.articleTopic")} className={inputCls} value={topic} onChange={(e) => setTopic(e.target.value)} />
              <input required placeholder={t("studio.primaryKeyword")} className={inputCls} value={keyword} onChange={(e) => setKeyword(e.target.value)} />
              <select className={inputCls} value={language} onChange={(e) => setLanguage(e.target.value as "fa" | "ar" | "en")}>
                <option value="fa">Persian — RTL</option>
                <option value="ar">Arabic — RTL</option>
                <option value="en">English — LTR</option>
              </select>

              {/* Strategy controls */}
              <div className="grid gap-3 md:grid-cols-2">
                <div>
                  <label className="block text-body-sm font-semibold text-ink-tertiary mb-1">{t("studio.tone")}</label>
                  <select className={inputCls} value={tone} onChange={(e) => setTone(e.target.value)}>
                    {TONE_OPTIONS.map((o) => <option key={o.value} value={o.value}>{labelFor(TONE_OPTIONS, o.value)}</option>)}
                    <option value="__custom__">{t("studio.toneCustom")}</option>
                  </select>
                  {tone === "__custom__" && <input className={`${inputCls} mt-2`} placeholder={t("studio.toneCustom")} value={customTone} onChange={(e) => setCustomTone(e.target.value)} />}
                </div>
                <div>
                  <label className="block text-body-sm font-semibold text-ink-tertiary mb-1">{t("studio.structure")}</label>
                  <select className={inputCls} value={structure} onChange={(e) => setStructure(e.target.value)}>
                    {STRUCTURE_OPTIONS.map((o) => <option key={o.value} value={o.value}>{labelFor(STRUCTURE_OPTIONS, o.value)}</option>)}
                    <option value="__custom__">{t("studio.structureCustom")}</option>
                  </select>
                  {structure === "__custom__" && <input className={`${inputCls} mt-2`} placeholder={t("studio.structureCustom")} value={customStructure} onChange={(e) => setCustomStructure(e.target.value)} />}
                </div>
                <div>
                  <label className="block text-body-sm font-semibold text-ink-tertiary mb-1">{t("studio.pointOfView")}</label>
                  <select className={inputCls} value={pov} onChange={(e) => setPov(e.target.value)}>
                    {POV_OPTIONS.map((o) => <option key={o.value} value={o.value}>{labelFor(POV_OPTIONS, o.value)}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-body-sm font-semibold text-ink-tertiary mb-1">{t("studio.audience")}</label>
                  <select className={inputCls} value={audience} onChange={(e) => setAudience(e.target.value)}>
                    {AUDIENCE_OPTIONS.map((o) => <option key={o.value} value={o.value}>{labelFor(AUDIENCE_OPTIONS, o.value)}</option>)}
                    <option value="__custom__">{t("studio.audienceCustom")}</option>
                  </select>
                  {audience === "__custom__" && <input className={`${inputCls} mt-2`} placeholder={t("studio.audienceCustom")} value={customAudience} onChange={(e) => setCustomAudience(e.target.value)} />}
                </div>
              </div>

              {/* Word count range */}
              <div>
                <label className="block text-body-sm font-semibold text-ink-tertiary mb-1">{t("studio.wordCount")}</label>
                <div className="grid grid-cols-2 gap-3">
                  <input type="number" placeholder={t("studio.wordCountMin")} className={inputCls} value={wordCountMin} onChange={(e) => setWordCountMin(e.target.value)} min="100" />
                  <input type="number" placeholder={t("studio.wordCountMax")} className={inputCls} value={wordCountMax} onChange={(e) => setWordCountMax(e.target.value)} min="100" />
                </div>
              </div>

              {/* Creativity slider */}
              <div>
                <label className="block text-body-sm font-semibold text-ink-tertiary mb-1">{t("studio.creativity")}: {temperature.toFixed(1)}</label>
                <input type="range" min="0" max="1" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="w-full accent-accent" />
              </div>

              {/* Competitor Takedown */}
              <div className="relative">
                <input
                  placeholder={t("studio.competitor")}
                  className={`${inputCls} pe-10`}
                  value={competitorUrl}
                  onChange={(e) => setCompetitorUrl(e.target.value)}
                />
                {competitorUrl.trim() && (
                  <div className="absolute end-3 top-1/2 -translate-y-1/2 flex items-center gap-1.5">
                    {scanning ? (
                      <span className="flex items-center gap-1 text-body-sm text-info animate-pulse-soft">
                        <span className="h-1.5 w-1.5 rounded-full bg-info animate-pulse" />
                        {t("studio.scanning")}
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-body-sm text-warning font-semibold">
                        {t("studio.takedown")}
                      </span>
                    )}
                  </div>
                )}
              </div>
              {competitorUrl.trim() && (
                <div className="rounded-xl border border-warning/20 bg-warning-subtle px-3 py-2 text-body-sm text-warning">
                  {t("studio.skyscraperMode")}
                </div>
              )}

              {/* Source URLs */}
              <textarea
                placeholder={t("studio.sourceUrls")}
                className={`h-20 ${inputCls} resize-none`}
                value={sourceUrls}
                onChange={(e) => setSourceUrls(e.target.value)}
              />

              {/* Extra instructions */}
              <textarea
                placeholder={t("studio.extraInstructions")}
                className={`h-24 ${inputCls} resize-none`}
                value={extraInstructions}
                onChange={(e) => setExtraInstructions(e.target.value)}
              />

              <button
                type="submit"
                disabled={submitting || !selectedProjectId}
                className="w-full rounded-xl bg-accent px-4 py-2.5 text-body-md font-semibold text-ink-inverse transition-all duration-fast ease-apple hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {submitting ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="h-4 w-4 rounded-full border-2 border-ink-inverse/30 border-t-ink-inverse animate-spin" />
                    {t("studio.generating")}
                  </span>
                ) : t("studio.startGeneration")}
              </button>
            </form>
          </article>

          {/* Task Status side panel */}
          <article className="elevated-card p-5 flex flex-col gap-3">
            <h3 className="text-heading-sm text-ink">{t("studio.taskStatus")}</h3>
            {!taskStatus ? (
              <p className="text-body-md text-ink-tertiary">{t("studio.noActiveTask")}</p>
            ) : (
              <>
                <code className="block rounded-lg bg-surface-tertiary px-3 py-2 font-mono text-body-sm text-ink-secondary break-all">
                  {taskStatus.task_id}
                </code>
                <div className="flex items-center gap-2">
                  <span className={`rounded-full px-2.5 py-1 text-body-sm font-semibold border ${taskStatus.state === "SUCCESS"
                    ? "bg-success-subtle text-success border-success/20"
                    : taskStatus.state === "FAILURE"
                      ? "bg-danger-subtle text-danger border-danger/20"
                      : "bg-info-subtle text-info border-info/20 animate-pulse-soft"
                    }`}>{taskStatus.state}</span>
                  {!taskStatus.ready && (
                    <span className="text-body-sm text-ink-tertiary animate-pulse-soft">● live</span>
                  )}
                </div>
                {taskStatus.status && (
                  <p className="text-body-md text-ink-secondary">{taskStatus.status}</p>
                )}
                <button
                  type="button"
                  className="rounded-xl border border-border px-3 py-1.5 text-body-sm text-ink-secondary transition-colors duration-fast hover:bg-surface-tertiary"
                  onClick={() => void refreshTask(taskStatus.task_id, setTaskStatus, token)}
                >
                  {t("studio.refreshStatus")}
                </button>
              </>
            )}
          </article>
        </div>
      )}

      {/* ── Bulk Topic Queue Tab ── */}
      {activeTab === "bulk" && (
        <article className="glass-card p-5">
          <h3 className="text-heading-sm text-ink mb-4">{t("studio.bulkTab")}</h3>
          <form className="space-y-3" onSubmit={onSubmitBatch}>
            <div>
              <label className="block text-body-sm font-semibold text-ink-tertiary mb-1">{t("studio.bulkTopics")}</label>
              <textarea
                required
                placeholder={t("studio.bulkTopicsPlaceholder")}
                className={`h-40 ${inputCls} resize-none`}
                value={bulkTopics}
                onChange={(e) => setBulkTopics(e.target.value)}
              />
              <p className="mt-1 text-body-sm text-ink-tertiary">{t("studio.maxTopics")}</p>
            </div>
            <input placeholder={t("studio.sharedKeyword")} className={inputCls} value={bulkKeyword} onChange={(e) => setBulkKeyword(e.target.value)} />
            <select className={inputCls} value={bulkLanguage} onChange={(e) => setBulkLanguage(e.target.value as "fa" | "ar" | "en")}>
              <option value="fa">Persian — RTL</option>
              <option value="ar">Arabic — RTL</option>
              <option value="en">English — LTR</option>
            </select>
            <button
              type="submit"
              disabled={bulkSubmitting || !selectedProjectId}
              className="w-full rounded-xl bg-accent px-4 py-2.5 text-body-md font-semibold text-ink-inverse transition-all duration-fast ease-apple hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {bulkSubmitting ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="h-4 w-4 rounded-full border-2 border-ink-inverse/30 border-t-ink-inverse animate-spin" />
                  {t("studio.submittingBatch")}
                </span>
              ) : t("studio.submitBatch")}
            </button>
          </form>

          {/* Batch progress */}
          {batchStatus && (
            <div className="mt-5 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-heading-sm text-ink">{t("studio.batchProgress")}</h4>
                <span className={`rounded-full px-2.5 py-1 text-body-sm font-semibold border ${batchStatus.status === "completed" ? "bg-success-subtle text-success border-success/20" : "bg-info-subtle text-info border-info/20 animate-pulse-soft"
                  }`}>{batchStatus.completed}/{batchStatus.total}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-surface-tertiary">
                <div
                  className="h-full rounded-full bg-accent transition-all duration-slow ease-apple"
                  style={{ width: `${batchStatus.total > 0 ? (batchStatus.completed / batchStatus.total) * 100 : 0}%` }}
                />
              </div>
              {batchStatus.tasks && batchStatus.tasks.length > 0 && (
                <div className="space-y-1.5 max-h-60 overflow-y-auto">
                  {batchStatus.tasks.map((bt, idx) => (
                    <div key={idx} className="flex items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2">
                      <span className={`h-2 w-2 shrink-0 rounded-full ${bt.status.toUpperCase() === "SUCCESS" ? "bg-success"
                        : bt.status.toUpperCase() === "FAILURE" ? "bg-danger"
                          : "bg-info animate-pulse-soft"
                        }`} />
                      <span className="text-body-sm text-ink truncate flex-1">{bt.topic}</span>
                      <span className="text-body-sm text-ink-tertiary shrink-0">{bt.status}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </article>
      )}

      {/* ── Social Tab ── */}
      {activeTab === "social" && (
        <article className="glass-card p-5">
          <h3 className="text-heading-sm text-ink mb-4">{t("studio.socialTab")}</h3>
          {!taskStatus && (
            <p className="text-body-md text-ink-tertiary">{t("studio.socialStart")}</p>
          )}
          {taskStatus && !socialTaskId && (
            <div className="rounded-xl border border-info/20 bg-info-subtle px-4 py-3 text-body-md text-info">
              {t("studio.socialPending")}
            </div>
          )}
          {socialTaskId && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <code className="font-mono text-body-sm text-ink-secondary">{socialTaskId.slice(0, 16)}…</code>
                <span className={`rounded-full px-2.5 py-1 text-body-sm font-semibold border ${socialStatus?.state === "SUCCESS"
                  ? "bg-success-subtle text-success border-success/20"
                  : "bg-info-subtle text-info border-info/20 animate-pulse-soft"
                  }`}>{socialStatus?.state ?? "PENDING"}</span>
                <button
                  type="button"
                  className="ms-auto rounded-xl border border-border px-3 py-1.5 text-body-sm text-ink-secondary transition-colors duration-fast hover:bg-surface-tertiary"
                  onClick={() => void refreshTask(socialTaskId, setSocialStatus, token)}
                >
                  {t("common.refresh")}
                </button>
              </div>
              <SocialPostsView socialStatus={socialStatus} />
            </div>
          )}
        </article>
      )}

      {/* ── Schema Tab ── */}
      {activeTab === "schema" && (
        <article className="glass-card p-5">
          <h3 className="text-heading-sm text-ink mb-4">{t("studio.schemaTab")} — Auto JSON-LD</h3>
          {!articleId ? (
            <p className="text-body-md text-ink-tertiary">{t("studio.schemaWaiting")}</p>
          ) : !jsonld ? (
            <div className="space-y-2">
              <div className="skeleton h-4 w-3/4" /><div className="skeleton h-4 w-full" />
              <div className="skeleton h-4 w-5/6" /><div className="skeleton h-4 w-2/3" />
            </div>
          ) : (
            <div className="space-y-4">
              <div className="rounded-xl border border-success/20 bg-success-subtle px-3 py-2 text-body-sm text-success">
                {t("studio.schemaReady").replace("{articleId}", articleId)}
              </div>
              <pre className="max-h-72 overflow-auto rounded-xl bg-ink p-4 text-body-sm text-ink-inverse font-mono leading-relaxed">
                {JSON.stringify(jsonld, null, 2)}
              </pre>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  className="rounded-xl bg-accent px-4 py-2 text-body-sm font-semibold text-ink-inverse transition-colors duration-fast hover:bg-accent-hover"
                  onClick={() => void navigator.clipboard.writeText(JSON.stringify(jsonld, null, 2))}
                >
                  {t("studio.copyJsonLd")}
                </button>
                <button
                  type="button"
                  className="rounded-xl border border-border px-4 py-2 text-body-sm text-ink-secondary transition-colors duration-fast hover:bg-surface-tertiary"
                  onClick={() => void navigator.clipboard.writeText(exportHtml)}
                >
                  {t("studio.copyHtml")}
                </button>
              </div>
            </div>
          )}
        </article>
      )}
    </section>
  );
}

const PLATFORM_STYLES: Record<string, { bg: string; label: string; icon: string }> = {
  linkedin: { bg: "border-info/20 bg-info-subtle", label: "LinkedIn", icon: "in" },
  twitter: { bg: "border-ink/10 bg-surface-tertiary", label: "X / Twitter", icon: "𝕏" },
  instagram: { bg: "border-warning/20 bg-warning-subtle", label: "Instagram", icon: "◉" },
};

function SocialPostsView({ socialStatus }: { socialStatus: TaskStatusResponse | null }) {
  const posts = extractSocialPosts(socialStatus);
  if (!posts) {
    return (
      <div className="grid gap-3 md:grid-cols-3">
        {["linkedin", "twitter", "instagram"].map((p) => (
          <div key={p} className="rounded-xl border border-border bg-surface-secondary p-4">
            <div className="skeleton h-3 w-20 mb-3" />
            <div className="space-y-2">
              <div className="skeleton h-3 w-full" /><div className="skeleton h-3 w-5/6" /><div className="skeleton h-3 w-4/5" />
            </div>
          </div>
        ))}
      </div>
    );
  }
  return (
    <div className="grid gap-3 md:grid-cols-3">
      {Object.entries(posts).map(([platform, copy]) => {
        const style = PLATFORM_STYLES[platform.toLowerCase()] ?? { bg: "border-border bg-surface", label: platform, icon: "◆" };
        return (
          <div key={platform} className={`rounded-xl border p-4 ${style.bg}`}>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-body-sm font-bold text-ink-secondary">{style.icon}</span>
              <p className="text-body-sm font-semibold uppercase tracking-wider text-ink-secondary">{style.label}</p>
              <button
                type="button"
                className="ms-auto text-body-sm text-ink-tertiary hover:text-ink transition-colors"
                onClick={() => void navigator.clipboard.writeText(copy)}
              >
                Copy
              </button>
            </div>
            <p className="whitespace-pre-wrap text-body-md text-ink leading-relaxed">{copy}</p>
          </div>
        );
      })}
    </div>
  );
}

function extractSocialPosts(status: TaskStatusResponse | null): Record<string, string> | null {
  if (!status?.result) return null;
  const rawPosts = status.result.posts;
  if (!rawPosts || typeof rawPosts !== "object") return null;
  const postsRecord = rawPosts as Record<string, unknown>;
  const normalized: Record<string, string> = {};
  for (const [key, value] of Object.entries(postsRecord)) {
    if (typeof value === "string") {
      normalized[key] = value;
    }
  }
  return Object.keys(normalized).length > 0 ? normalized : null;
}

async function refreshTask(
  taskId: string,
  setter: (payload: TaskStatusResponse) => void,
  token: string
) {
  try {
    const payload = await apiRequest<TaskStatusResponse>(`/content/task/${taskId}`, { token });
    setter(payload);
  } catch { /* ignore refresh errors */ }
}

function extractError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.detail;
  }
  return "Unexpected error";
}
