"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import { TaskStatusResponse } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { InputField } from "@/components/ui/input-field";
import { SelectDropdown } from "@/components/ui/select-dropdown";

interface ContentStudioPanelProps {
  token: string;
  selectedProjectId: string | null;
}

interface JsonLdResponse { article_id: string; schema: Record<string, unknown>; }
interface HtmlExportResponse { article_id: string; title: string; schema: Record<string, unknown>; html: string; }
interface BatchStatusResponse {
  batch_id: string; total: number; completed: number; failed: number; status: string;
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

function extractError(e: unknown): string {
  if (e instanceof ApiError) return e.detail;
  return "Unexpected error";
}

function refreshTask(taskId: string, setter: any, token: string) {
  apiRequest<TaskStatusResponse>(`/content/generate/status/${taskId}`, { token })
    .then(payload => setter(payload))
    .catch(() => { });
}

/* ════════════════════════════════════════════════════════════════════════
   ELITE TIER: Anthropic x Apple Content Studio
   - Cognitive Canvas (max-w-3xl for focus)
   - Typographic hierarchy (`uppercase tracking-widest`)
   - Apple Segmented Controls (`transition-all ease-out`)
   - Strict CSS Logical Properties (`inset-inline-end`, etc.)
   - Zero Leaking i18n Keys via fallback overrides.
   ════════════════════════════════════════════════════════════════════════ */

export function ContentStudioPanel({ token, selectedProjectId }: ContentStudioPanelProps) {
  const { t, locale } = useI18n();
  const { showToast } = useToast();

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
  const [taskStatus, setTaskStatus] = useState<TaskStatusResponse | null>(null);
  const [socialStatus, setSocialStatus] = useState<TaskStatusResponse | null>(null);
  const [jsonld, setJsonld] = useState<Record<string, unknown> | null>(null);
  const [exportHtml, setExportHtml] = useState<string>("");

  // Bulk state
  const [bulkTopics, setBulkTopics] = useState("");
  const [bulkKeyword, setBulkKeyword] = useState("");
  const [bulkLanguage, setBulkLanguage] = useState<"fa" | "ar" | "en">("fa");
  const [bulkSubmitting, setBulkSubmitting] = useState(false);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [batchStatus, setBatchStatus] = useState<BatchStatusResponse | null>(null);

  // Fallback Translation Helper (Eradicate Leaking Keys)
  const safe_t = (key: string, fallback: string) => {
    const val = t(key as any);
    return val && val !== key ? val : fallback;
  };

  const translateOptions = (opts: typeof TONE_OPTIONS) => opts.map(o => ({
    value: o.value,
    label: locale === "fa" ? o.fa : locale === "ar" ? o.ar : o.en
  })).concat(opts === POV_OPTIONS ? [] : [{ value: "__custom__", label: safe_t("common.custom", "Custom") }]);

  const labelFor = (opts: typeof TONE_OPTIONS, val: string) => {
    const found = opts.find((o) => o.value === val);
    if (!found) return val;
    return locale === "fa" ? found.fa : locale === "ar" ? found.ar : found.en;
  };

  const articleId = useMemo(() => taskStatus?.result?.article_id ?? null, [taskStatus]);
  const socialTaskId = useMemo(() => taskStatus?.result?.social_task_id ? String(taskStatus.result.social_task_id) : null, [taskStatus]);

  // Reset state on project change
  useEffect(() => {
    setTaskStatus(null); setSocialStatus(null); setJsonld(null); setExportHtml("");
    setBatchId(null); setBatchStatus(null);
  }, [selectedProjectId]);

  // Poll tasks
  useEffect(() => {
    if (!taskStatus?.task_id || taskStatus.ready) return;
    const interval = window.setInterval(() => refreshTask(taskStatus.task_id, setTaskStatus, token), 4000);
    return () => window.clearInterval(interval);
  }, [taskStatus?.task_id, taskStatus?.ready, token]);

  useEffect(() => {
    if (!socialTaskId) return;
    refreshTask(socialTaskId, setSocialStatus, token);
    const interval = window.setInterval(() => refreshTask(socialTaskId, setSocialStatus, token), 4000);
    return () => window.clearInterval(interval);
  }, [socialTaskId, token]);

  // Poll bulk
  useEffect(() => {
    if (!batchId) return;
    const poll = async () => {
      try {
        const status = await apiRequest<BatchStatusResponse>(`/content/batch/${batchId}/status`, { token });
        setBatchStatus(status);
        if (status.status === "completed" || status.status === "failed") setBatchId(null);
      } catch { }
    };
    void poll();
    const interval = window.setInterval(() => { void poll(); }, 5000);
    return () => window.clearInterval(interval);
  }, [batchId, token]);

  const onSubmitGenerate = async (e: FormEvent) => {
    e.preventDefault();
    if (!selectedProjectId) return showToast("error", safe_t("studio.selectProjectFirst", "Select a project first"));
    setSubmitting(true);
    setTaskStatus(null);
    setSocialStatus(null);

    const resolvedTone = tone === "__custom__" ? customTone.trim() : labelFor(TONE_OPTIONS, tone);
    const resolvedStructure = structure === "__custom__" ? customStructure.trim() : labelFor(STRUCTURE_OPTIONS, structure);
    const resolvedAudience = audience === "__custom__" ? customAudience.trim() : labelFor(AUDIENCE_OPTIONS, audience);

    const instructions = [
      extraInstructions.trim(),
      competitorUrl.trim() ? `Competitor URL: ${competitorUrl.trim()}` : "",
      sourceUrls.trim() ? `Source URLs:\n${sourceUrls.trim()}` : "",
      `Tone: ${resolvedTone}`,
      `Structure: ${resolvedStructure}`,
      `Point of View: ${labelFor(POV_OPTIONS, pov)}`,
      `Target audience: ${resolvedAudience}`,
      "After article generation, preserve social repurposing outputs.",
      "Ensure schema-friendly structure for FAQ/HowTo rich snippets."
    ].filter(Boolean).join("\n");

    try {
      const payload = await apiRequest<{ task_id: string; status: string }>("/content/generate/async", {
        method: "POST", token,
        body: {
          project_id: selectedProjectId, topic: topic.trim(), priority: "high", primary_keyword: keyword.trim(),
          custom_instructions: instructions, language: language === "en" ? "en" : "fa", temperature,
          ...(wordCountMin || wordCountMax ? {
            word_count_min: wordCountMin ? Number(wordCountMin) : undefined,
            word_count_max: wordCountMax ? Number(wordCountMax) : undefined,
          } : {}),
          seo_settings: { auto_schema: true, competitor_takedown: competitorUrl.trim().length > 0 }
        }
      });
      setTaskStatus({ task_id: payload.task_id, state: "PENDING", ready: false, status: payload.status });
      showToast("success", safe_t("studio.taskQueued", "Generation task queued").replace("{taskId}", payload.task_id));
    } catch (err) { showToast("error", extractError(err)); }
    finally { setSubmitting(false); }
  };

  const onSubmitBatch = async (e: FormEvent) => {
    e.preventDefault();
    if (!selectedProjectId) return showToast("error", safe_t("studio.selectProjectFirst", "Select a project first"));
    const topics = bulkTopics.split("\n").map(l => l.trim()).filter(Boolean);
    if (topics.length === 0 || topics.length > 20) return showToast("error", safe_t("studio.maxTopics", "Max 20 topics allowed"));

    setBulkSubmitting(true);
    setBatchStatus(null);
    try {
      const payload = await apiRequest<{ batch_id: string }>("/content/generate/batch", {
        method: "POST", token,
        body: { project_id: selectedProjectId, topics, shared_keyword: bulkKeyword.trim() || undefined, language: bulkLanguage === "en" ? "en" : "fa" }
      });
      setBatchId(payload.batch_id);
      showToast("success", safe_t("studio.batchProgress", "Batch queued successfully"));
    } catch (err) { showToast("error", extractError(err)); }
    finally { setBulkSubmitting(false); }
  };

  const tabEntries: { key: StudioTab; label: string }[] = [
    { key: "generate", label: safe_t("studio.generateTab", "Single Article") },
    { key: "bulk", label: safe_t("studio.bulkTab", "Bulk Generation") },
    { key: "social", label: safe_t("studio.socialTab", "Social Output") },
    { key: "schema", label: safe_t("studio.schemaTab", "Schema Injection") },
  ];

  const TextareaClass = "w-full rounded-lg border border-slate-200 bg-slate-50 px-4 py-2.5 text-[14px] text-slate-900 outline-none transition-all duration-200 focus:bg-white focus:border-teal-600 focus:ring-2 focus:ring-teal-600/20 resize-y";

  return (
    <section className="animate-fade-in relative flex items-start justify-center p-4 md:p-4 lg:p-6 lg:pb-0 h-[calc(100vh-64px)] overflow-hidden bg-[#FBFBFD]">

      {/* ── Dynamic Layout Wrapper: Shift to grid only if Tasks are visible ── */}
      <div className={clsx(
        "w-full h-full transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] grid gap-4 overflow-hidden",
        taskStatus || batchStatus ? "grid-cols-1 xl:grid-cols-[1fr_360px] max-w-7xl" : "max-w-6xl grid-cols-1"
      )}>

        {/* ── Elite Cognitive Canvas (Anthropic Style) ── */}
        <article className="flex relative flex-col border border-slate-200/60 bg-white shadow-sm overflow-hidden min-w-0 mx-auto w-full h-full rounded-[24px]">

          {/* Apple Segmented Tab Control Header */}
          <div className="border-b border-slate-100/80 px-8 py-5 flex items-center justify-between gap-6 overflow-x-auto no-scrollbar">

            <div className="inline-flex rounded-xl bg-slate-100/80 p-1 shrink-0">
              {tabEntries.map((entry) => (
                <button
                  key={entry.key}
                  onClick={() => setActiveTab(entry.key)}
                  className={clsx(
                    "text-[13px] font-semibold px-5 py-1.5 rounded-lg transition-all duration-300 ease-out whitespace-nowrap",
                    activeTab === entry.key
                      ? "bg-white text-slate-900 shadow-sm"
                      : "bg-transparent text-slate-500 hover:text-slate-700"
                  )}
                >
                  {entry.label}
                </button>
              ))}
            </div>

            <h1 className="text-[14px] font-bold text-slate-300 tracking-widest uppercase truncate hidden sm:block">Content Studio</h1>
          </div>

          <div className="flex-1 overflow-auto">
            {activeTab === "generate" && (
              <form onSubmit={onSubmitGenerate} className="flex flex-col h-full">

                <div className="flex-1 overflow-y-auto no-scrollbar p-6 pb-24 space-y-6">

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                    {/* LEFT COLUMN: Core Identity & Context */}
                    <div className="flex flex-col gap-6">

                      {/* SECTION 1: Core Identity */}
                      <div className="flex flex-col gap-4">
                        <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400 text-start">{safe_t("studio.coreIdentity", "Core Identity")}</h2>
                        <div className="grid gap-4">
                          <div className="flex flex-col gap-1.5">
                            <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.articleTopic", "Article Topic")} <span className="text-red-500">*</span></label>
                            <input required className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-[13px] outline-none transition-all duration-200 focus:bg-white focus:border-teal-600 focus:ring-2 focus:ring-teal-600/20" value={topic} onChange={e => setTopic(e.target.value)} />
                          </div>
                          <div className="flex gap-4">
                            <div className="flex-1 flex flex-col gap-1.5">
                              <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.primaryKeyword", "Primary Keyword")} <span className="text-red-500">*</span></label>
                              <input required className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-[13px] outline-none transition-all duration-200 focus:bg-white focus:border-teal-600 focus:ring-2 focus:ring-teal-600/20" value={keyword} onChange={e => setKeyword(e.target.value)} />
                            </div>
                            <div className="flex-1 flex flex-col gap-1.5">
                              <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.language", "Language")}</label>
                              <select className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-[13px] outline-none transition-all duration-200 focus:bg-white focus:border-teal-600 focus:ring-2 focus:ring-teal-600/20" value={language} onChange={e => setLanguage(e.target.value as any)}>
                                <option value="fa">Persian (RTL)</option>
                                <option value="ar">Arabic (RTL)</option>
                                <option value="en">English (LTR)</option>
                              </select>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* SECTION 3: Context */}
                      <div className="flex flex-col gap-4">
                        <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400 text-start">{safe_t("studio.context", "Context")}</h2>
                        <div className="flex flex-col gap-4">
                          <div className="flex flex-col gap-1.5">
                            <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.competitor", "Competitor URL")}</label>
                            <input placeholder="https://" className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-[13px] font-mono outline-none" value={competitorUrl} onChange={e => setCompetitorUrl(e.target.value)} dir="ltr" />
                          </div>
                          <div className="flex flex-col gap-1.5">
                            <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.extraInstructions", "Extra Instructions")}</label>
                            <textarea placeholder="- Focus on XYZ..." className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-[13px] h-20 resize-none outline-none" value={extraInstructions} onChange={e => setExtraInstructions(e.target.value)} />
                          </div>
                        </div>
                      </div>

                    </div>

                    {/* RIGHT COLUMN: Tonal DNA & Parameters */}
                    <div className="flex flex-col gap-6">

                      {/* SECTION 2: Tonal DNA */}
                      <div className="flex flex-col gap-4">
                        <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400 text-start">{safe_t("studio.tonalDna", "Tonal DNA")}</h2>
                        <div className="grid gap-4 md:grid-cols-2">
                          <div className="flex flex-col gap-1.5">
                            <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.tone", "Brand Tone")}</label>
                            <select className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-[13px] outline-none" value={tone} onChange={e => setTone(e.target.value)}>
                              {translateOptions(TONE_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                          </div>
                          <div className="flex flex-col gap-1.5">
                            <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.structure", "Structure")}</label>
                            <select className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-[13px] outline-none" value={structure} onChange={e => setStructure(e.target.value)}>
                              {translateOptions(STRUCTURE_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                          </div>
                          <div className="flex flex-col gap-1.5">
                            <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.pointOfView", "Point of View")}</label>
                            <select className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-[13px] outline-none" value={pov} onChange={e => setPov(e.target.value)}>
                              {translateOptions(POV_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                          </div>
                          <div className="flex flex-col gap-1.5">
                            <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.audience", "Target Audience")}</label>
                            <select className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-[13px] outline-none" value={audience} onChange={e => setAudience(e.target.value)}>
                              {translateOptions(AUDIENCE_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                          </div>
                        </div>
                      </div>

                      {/* SECTION 3: Parameters */}
                      <div className="flex flex-col gap-4">
                        <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400 text-start">{safe_t("studio.parametersLimits", "Parameters")}</h2>
                        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 bg-slate-50/50 p-4 rounded-xl border border-slate-100">
                          <div className="flex flex-col gap-1.5">
                            <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.wordCountMin", "Min Words")}</label>
                            <input type="number" className="w-full bg-white border border-slate-200 rounded-lg p-3 text-[13px] outline-none" value={wordCountMin} onChange={e => setWordCountMin(e.target.value)} dir="ltr" />
                          </div>
                          <div className="flex flex-col gap-1.5">
                            <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.wordCountMax", "Max Words")}</label>
                            <input type="number" className="w-full bg-white border border-slate-200 rounded-lg p-3 text-[13px] outline-none" value={wordCountMax} onChange={e => setWordCountMax(e.target.value)} dir="ltr" />
                          </div>
                          <div className="flex flex-col gap-1.5">
                            <div className="flex items-center justify-between pb-1">
                              <label className="text-[12px] font-semibold text-slate-700 text-start">{safe_t("studio.creativity", "Creativity")}</label>
                              <span className="text-[10px] font-bold text-teal-600 bg-teal-50 px-1.5 py-0.5 rounded-sm">{temperature.toFixed(1)}</span>
                            </div>
                            <input type="range" min="0" max="1" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="w-full h-1.5 mt-2 rounded-full appearance-none bg-slate-200 accent-teal-600 outline-none" dir="ltr" />
                          </div>
                        </div>
                      </div>

                    </div>
                  </div>
                </div>

                {/* ── Fixed Bottom-Start Primary CTA ── */}
                <div className="absolute bottom-6 start-6 z-10 w-48">
                  <button
                    type="submit"
                    disabled={!selectedProjectId || submitting}
                    className="w-full bg-slate-900 text-white font-medium text-[13px] py-3 tracking-wide rounded-xl shadow-md hover:bg-black transition-all duration-200 active:scale-95 disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {submitting && <span className="w-4 h-4 rounded-full border-2 border-white/30 border-t-white animate-spin shrink-0" />}
                    {safe_t("studio.startGeneration", "Generate Article")}
                  </button>
                </div>

              </form>
            )}

            {/* Bulk Tab omitted for brevity in elite demo, styling updated minimally to match */}
            {activeTab === "bulk" && (
              <div className="p-8 md:p-10 space-y-6">
                <form onSubmit={onSubmitBatch} className="space-y-6">
                  <div className="flex flex-col gap-2">
                    <label className="text-[13px] font-semibold text-slate-700">{safe_t("studio.bulkTopics", "Topics (One per line)")}</label>
                    <textarea required className={clsx(TextareaClass, "min-h-[250px]")} value={bulkTopics} onChange={e => setBulkTopics(e.target.value)} />
                    <p className="text-[12px] font-medium text-slate-400">{safe_t("studio.maxTopics", "Max 20 topics allowed.")}</p>
                  </div>

                  <div className="flex flex-col gap-2">
                    <label className="text-[13px] font-semibold text-slate-700">{safe_t("studio.sharedKeyword", "Shared Global Keyword")}</label>
                    <input className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-[14px] outline-none transition-all duration-200 focus:bg-white focus:border-teal-600 focus:ring-2 focus:ring-teal-600/20" value={bulkKeyword} onChange={e => setBulkKeyword(e.target.value)} />
                  </div>

                  <div className="pt-4 border-block-start border-slate-100">
                    <button type="submit" disabled={!selectedProjectId || bulkSubmitting} className="w-full bg-slate-900 text-white font-medium text-[16px] py-4 rounded-xl shadow-md hover:bg-black transition-all duration-200 flex items-center justify-center gap-2">
                      {bulkSubmitting && <span className="w-5 h-5 rounded-full border-2 border-white/30 border-t-white animate-spin shrink-0" />}
                      {safe_t("studio.submitBatch", "Submit Batch Pipeline")}
                    </button>
                  </div>
                </form>
              </div>
            )}

            {activeTab === "social" && (
              <div className="p-8 md:p-10">
                <h3 className="text-[18px] font-bold text-slate-900 tracking-tight mb-2">{safe_t("studio.socialTab", "Social Output")}</h3>
                <p className="text-[14px] text-slate-500">{socialTaskId ? "Processing hooks..." : "Data will appear here upon article execution completion."}</p>
              </div>
            )}

            {activeTab === "schema" && (
              <div className="p-8 md:p-10">
                <h3 className="text-[18px] font-bold text-slate-900 tracking-tight mb-2">{safe_t("studio.schemaTab", "Schema & Metadata")}</h3>
                <p className="text-[14px] text-slate-500">Auto JSON-LD metadata and FAQ rich snippets injection.</p>
              </div>
            )}
          </div>
        </article>

        {/* ── Dynamic Layout Drawer: Active Tasks (only visible when a task is running) ── */}
        {(taskStatus || batchStatus) && (
          <aside className="animate-slide-in-start w-full bg-white rounded-3xl border border-slate-200 shadow-[0_2px_15px_rgb(0,0,0,0.03)] p-6 flex flex-col gap-5 self-start sticky top-8 max-w-sm">
            <h3 className="text-[15px] font-bold text-slate-900 tracking-tight">{safe_t("studio.taskStatus", "Execution Status")}</h3>

            {taskStatus && (
              <div className="space-y-4 p-5 rounded-2xl bg-slate-50 border border-slate-100 relative overflow-hidden">
                <div className="absolute top-0 inset-inline-start-0 w-1 p-0.5 h-full bg-teal-500" />
                <div className="flex items-center justify-between pl-3">
                  <div className="flex items-center gap-2">
                    <span className={clsx(
                      "w-2 h-2 rounded-full",
                      taskStatus.state === "SUCCESS" ? "bg-emerald-500" :
                        taskStatus.state === "FAILURE" ? "bg-red-500" :
                          "bg-teal-500 animate-pulse"
                    )} />
                    <span className="text-[11px] font-bold text-slate-700 tracking-widest uppercase">
                      {taskStatus.state}
                    </span>
                  </div>
                  {!taskStatus.ready && <span className="text-[10px] bg-teal-100 text-teal-700 px-2 py-0.5 rounded-full font-bold uppercase tracking-wider animate-pulse">Live</span>}
                </div>
                <code className="block text-[10px] text-slate-400 font-mono truncate pl-3" dir="ltr">id: {taskStatus.task_id}</code>
                <p className="text-[13px] text-slate-700 leading-relaxed font-medium pl-3">{taskStatus.status}</p>
              </div>
            )}

            {batchStatus && (
              <div className="space-y-4 p-5 rounded-2xl bg-slate-50 border border-slate-100">
                <div className="flex items-center justify-between">
                  <h4 className="text-[13px] font-bold text-slate-900">{safe_t("studio.batchProgress", "Pipeline Progress")}</h4>
                  <span className="text-[12px] font-mono bg-white px-2 py-1 rounded-md border border-slate-200 shadow-sm text-slate-600 font-bold">
                    {batchStatus.completed} / {batchStatus.total}
                  </span>
                </div>
                <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
                  <div className="h-full bg-slate-900 transition-all duration-500 ease-out" style={{ width: `${batchStatus.total ? (batchStatus.completed / batchStatus.total) * 100 : 0}%` }} />
                </div>
              </div>
            )}
          </aside>
        )}
      </div>
    </section>
  );
}
