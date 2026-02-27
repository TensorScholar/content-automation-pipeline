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
  const [customPov, setCustomPov] = useState("");
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
  })).concat([{ value: "__custom__", label: safe_t("common.custom", "سفارشی") }]); // Hardcoded fallback for now since common.custom might not exist in dictionary

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
      `Point of View: ${pov === "__custom__" ? customPov.trim() : labelFor(POV_OPTIONS, pov)}`,
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
    { key: "generate", label: safe_t("studio.tabGenerate", "Single Article") },
    { key: "bulk", label: safe_t("studio.tabBatch", "Bulk Generation") },
    { key: "social", label: safe_t("studio.tabSocial", "Social Output") },
    { key: "schema", label: safe_t("studio.tabSchema", "SEO & Schema") },
  ];

  const InputClass = clsx(
    "w-full bg-slate-50 border border-slate-200 focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 rounded-xl px-4 py-2.5 min-h-[42px] transition-all text-[14px] outline-none",
    locale === 'fa' || locale === 'ar' ? "font-normal" : ""
  );

  const LabelClass = clsx(
    "text-[13px] font-bold text-slate-700 max-w-max leading-none mb-1.5",
    locale === 'fa' || locale === 'ar' ? "font-medium" : ""
  );

  const HeaderClass = "text-base font-bold text-slate-800 mb-5 border-b border-slate-100 pb-3";

  return (
    <section className="animate-fade-in relative flex justify-center w-full min-h-[calc(100vh-64px)]">

      {/* ── Dynamic Layout Wrapper ── */}
      <div className={clsx(
        "w-full transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] grid gap-6",
        taskStatus || batchStatus ? "grid-cols-1 xl:grid-cols-[1fr_360px] max-w-7xl" : "grid-cols-1"
      )}>

        {/* ── LAYER 2: THE FIXED CANVAS (The "Page") ── */}
        <article className="flex flex-col h-[calc(100vh-48px)] my-6 mx-6 bg-slate-50/80 backdrop-blur-xl border border-slate-200/60 shadow-[0_4px_20px_rgba(0,0,0,0.03)] rounded-[2.5rem] overflow-hidden w-full relative">

          {/* Apple Segmented Tab Control Header (Fixed) */}
          <div className="flex items-center justify-between p-6 pb-4 border-b border-slate-200/50 bg-white/40 shrink-0 z-10 backdrop-blur-xl">

            <div className="inline-flex bg-slate-200/50 p-1 rounded-full backdrop-blur-md w-full sm:w-auto">
              {tabEntries.map((entry) => (
                <button
                  type="button"
                  key={entry.key}
                  onClick={() => setActiveTab(entry.key)}
                  className={clsx(
                    "px-6 py-2 rounded-full text-[13px] font-bold transition-all duration-300 whitespace-nowrap flex-1 sm:flex-none",
                    activeTab === entry.key
                      ? "bg-white text-slate-900 shadow-sm"
                      : "bg-transparent text-slate-500 hover:text-slate-700"
                  )}
                >
                  {entry.label}
                </button>
              ))}
            </div>

          </div>

          <div className="flex-1 overflow-y-auto w-full">
            {activeTab === "generate" && (
              <form onSubmit={onSubmitGenerate} className="flex flex-col min-h-full p-6 lg:p-8">

                <div className="flex flex-col max-w-5xl mx-auto w-full gap-6">
                  {/* ── LAYER 3: THE INTERNAL BOXES (The Form Cards) ── */}
                  <div className="grid grid-cols-12 gap-6">

                    {/* BOX 1: Core Identity */}
                    <div className="col-span-12 lg:col-span-6 bg-white rounded-2xl p-6 shadow-sm border border-slate-200/50 flex flex-col gap-4">
                      <h2 className={HeaderClass}>{safe_t("studio.coreIdentity", "Core Identity")}</h2>

                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{safe_t("studio.articleTopic", "Article Topic")} <span className="text-red-500">*</span></label>
                        <input required className={InputClass} value={topic} onChange={e => setTopic(e.target.value)} />
                      </div>

                      <div className="flex flex-col md:flex-row gap-4">
                        <div className="flex-1 flex flex-col gap-0">
                          <label className={LabelClass}>{safe_t("studio.primaryKeyword", "Primary Keyword")} <span className="text-red-500">*</span></label>
                          <input required className={InputClass} value={keyword} onChange={e => setKeyword(e.target.value)} />
                        </div>
                        <div className="flex-1 flex flex-col gap-0">
                          <label className={LabelClass}>{safe_t("studio.language", "Language")}</label>
                          <select className={InputClass} value={language} onChange={e => setLanguage(e.target.value as any)}>
                            <option value="fa">{safe_t("lang.fa", "فارسی")}</option>
                            <option value="ar">{safe_t("lang.ar", "العربية")}</option>
                            <option value="en">{safe_t("lang.en", "English")}</option>
                          </select>
                        </div>
                      </div>
                    </div>

                    {/* BOX 2: Tonal DNA */}
                    <div className="col-span-12 lg:col-span-6 bg-white rounded-2xl p-6 shadow-sm border border-slate-200/50 flex flex-col gap-4">
                      <h2 className={HeaderClass}>{safe_t("studio.tonalDna", "Tonal DNA")}</h2>

                      <div className="flex flex-col gap-4">
                        <div className="flex flex-col md:flex-row gap-4">
                          <div className="flex-1 flex flex-col gap-0 relative">
                            <label className={LabelClass}>{safe_t("studio.tone", "Brand Tone")}</label>
                            <select className={InputClass} value={tone} onChange={e => setTone(e.target.value)}>
                              {translateOptions(TONE_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                            {tone === "__custom__" && (
                              <div className="mt-3 animate-fade-in">
                                <input autoFocus placeholder="..." className={InputClass} value={customTone} onChange={e => setCustomTone(e.target.value)} />
                              </div>
                            )}
                          </div>

                          <div className="flex-1 flex flex-col gap-0 relative">
                            <label className={LabelClass}>{safe_t("studio.audience", "Target Audience")}</label>
                            <select className={InputClass} value={audience} onChange={e => setAudience(e.target.value)}>
                              {translateOptions(AUDIENCE_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                            {audience === "__custom__" && (
                              <div className="mt-3 animate-fade-in">
                                <input autoFocus placeholder="..." className={InputClass} value={customAudience} onChange={e => setCustomAudience(e.target.value)} />
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-col gap-4">
                        <div className="flex flex-col md:flex-row gap-4">
                          <div className="flex-1 flex flex-col gap-0 relative">
                            <label className={LabelClass}>{safe_t("studio.structure", "Structure")}</label>
                            <select className={InputClass} value={structure} onChange={e => setStructure(e.target.value)}>
                              {translateOptions(STRUCTURE_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                            {structure === "__custom__" && (
                              <div className="mt-3 animate-fade-in">
                                <input autoFocus placeholder="..." className={InputClass} value={customStructure} onChange={e => setCustomStructure(e.target.value)} />
                              </div>
                            )}
                          </div>

                          <div className="flex-1 flex flex-col gap-0 relative">
                            <label className={LabelClass}>{safe_t("studio.pointOfView", "Point of View")}</label>
                            <select className={InputClass} value={pov} onChange={e => setPov(e.target.value)}>
                              {translateOptions(POV_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                            {pov === "__custom__" && (
                              <div className="mt-3 animate-fade-in">
                                <input autoFocus placeholder="..." className={InputClass} value={customPov} onChange={e => setCustomPov(e.target.value)} />
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* BOX 3: Additional Context */}
                    <div className="col-span-12 bg-white rounded-2xl p-6 shadow-sm border border-slate-200/50 flex flex-col gap-4">
                      <h2 className={HeaderClass}>{safe_t("studio.context", "Additional Context")}</h2>
                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{safe_t("studio.reference", "Reference URL (Analysis)")}</label>
                        <input placeholder="https://" className={clsx(InputClass, "font-mono text-[13px]")} value={competitorUrl} onChange={e => setCompetitorUrl(e.target.value)} dir="ltr" />
                      </div>
                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{safe_t("studio.extraInstructions", "Extra Instructions")}</label>
                        <textarea rows={2} placeholder="- Focus on XYZ..." className={clsx(InputClass, "resize-none h-[74px]")} value={extraInstructions} onChange={e => setExtraInstructions(e.target.value)} />
                      </div>
                    </div>

                    {/* BOX 4: Parameters & Action */}
                    <div className="col-span-12 bg-white rounded-2xl p-6 shadow-sm border border-slate-200/50 flex flex-col gap-6">
                      <h2 className={HeaderClass}>{safe_t("studio.parametersLimits", "Parameters & Limits")}</h2>

                      <div className="flex flex-col gap-8">
                        {/* Sliders Area */}
                        <div className="flex flex-col gap-6">
                          <div className="flex gap-4 max-w-lg">
                            <div className="flex-1 flex flex-col gap-0">
                              <label className={LabelClass}>{safe_t("studio.wordCountMin", "Min Words")}</label>
                              <input type="number" className={InputClass} value={wordCountMin} onChange={e => setWordCountMin(e.target.value)} dir="ltr" />
                            </div>
                            <div className="flex-1 flex flex-col gap-0">
                              <label className={LabelClass}>{safe_t("studio.wordCountMax", "Max Words")}</label>
                              <input type="number" className={InputClass} value={wordCountMax} onChange={e => setWordCountMax(e.target.value)} dir="ltr" />
                            </div>
                          </div>

                          <div className="flex flex-col gap-1 w-full max-w-lg">
                            <label className={LabelClass}>{safe_t("studio.creativity", "Creativity")}</label>
                            <div className="flex items-center gap-4 mt-2">
                              <input type="range" min="0" max="1" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="flex-1 h-2 rounded-full appearance-none bg-slate-200 accent-emerald-600 outline-none" dir="ltr" />
                              <span className="text-[14px] font-black text-emerald-800 bg-emerald-100 px-3 py-1 rounded-lg shrink-0 w-12 text-center">{temperature.toFixed(1)}</span>
                            </div>
                            <p className="text-[12px] text-slate-500 mt-1 font-medium">0.1 - 0.4: {safe_t("studio.creativityConservative", "Conservative")} | 0.5 - 0.7: {safe_t("studio.creativityBalanced", "Balanced")} | 0.8 - 1.0: {safe_t("studio.creativityCreative", "Creative")}</p>
                          </div>
                        </div>

                        {/* Button Action Area */}
                        <div className="flex w-full justify-end mt-2 pt-6 border-t border-slate-100">
                          <button
                            type="submit"
                            disabled={!selectedProjectId || submitting}
                            className="w-full md:w-auto px-10 bg-emerald-800 text-white font-bold py-3.5 rounded-xl shadow-[0_8px_20px_rgba(4,71,49,0.3)] hover:-translate-y-1 hover:shadow-[0_12px_25px_rgba(4,71,49,0.4)] active:translate-y-0 transition-all duration-300 flex items-center justify-center gap-2 disabled:opacity-50 disabled:hover:translate-y-0 disabled:shadow-none"
                          >
                            {submitting && <span className="w-5 h-5 rounded-full border-[3px] border-white/30 border-t-white animate-spin shrink-0" />}
                            {safe_t("studio.generate", "Generate Article")}
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

              </form>
            )}

            {activeTab === "bulk" && (
              <div className="h-full flex flex-col pb-8 p-6 lg:p-8">
                <form onSubmit={onSubmitBatch} className="flex flex-col gap-6 max-w-4xl mx-auto w-full h-full">
                  <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200/50 flex flex-col gap-4">
                    <h2 className={HeaderClass}>{safe_t("studio.parametersLimits", "Parameters")}</h2>
                    <div className="flex flex-col md:flex-row gap-4">
                      <div className="flex-1 flex flex-col gap-0">
                        <label className={LabelClass}>{safe_t("studio.batchLanguage", "Language")}</label>
                        <select className={InputClass} value={bulkLanguage} onChange={e => setBulkLanguage(e.target.value as any)}>
                          <option value="fa">{safe_t("lang.fa", "فارسی")}</option>
                          <option value="ar">{safe_t("lang.ar", "العربية")}</option>
                          <option value="en">{safe_t("lang.en", "English")}</option>
                        </select>
                      </div>
                      <div className="flex-1 flex flex-col gap-0">
                        <label className={LabelClass}>{safe_t("studio.batchSharedKeyword", "Shared Keyword")}</label>
                        <input className={InputClass} value={bulkKeyword} onChange={e => setBulkKeyword(e.target.value)} />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200/50 flex flex-col flex-1">
                    <h2 className={HeaderClass}>{safe_t("studio.coreIdentity", "Core Identity")}</h2>
                    <div className="flex flex-col gap-0 flex-1 h-full min-h-[300px]">
                      <label className={LabelClass}>{safe_t("studio.batchTopics", "Topics (One per line)")}</label>
                      <textarea required className={clsx(InputClass, "resize-none h-full py-4")} placeholder={safe_t("studio.batchTopicsPlaceholder", "Topic 1\nTopic 2\nTopic 3")} value={bulkTopics} onChange={e => setBulkTopics(e.target.value)} />
                      <p className="text-[12px] font-bold text-slate-400 mt-2">{safe_t("studio.maxTopics", "Max 20 topics allowed.")}</p>
                    </div>
                  </div>

                  <div className="flex w-full justify-end mt-2 pb-8">
                    <button type="submit" disabled={!selectedProjectId || bulkSubmitting} className="w-full md:w-auto px-10 bg-emerald-800 text-white font-bold py-3.5 rounded-xl shadow-[0_8px_20px_rgba(4,71,49,0.3)] hover:-translate-y-1 hover:shadow-[0_12px_25px_rgba(4,71,49,0.4)] active:translate-y-0 transition-all duration-300 flex items-center justify-center gap-2 disabled:opacity-50 disabled:hover:translate-y-0 disabled:shadow-none">
                      {bulkSubmitting && <span className="w-5 h-5 rounded-full border-[3px] border-white/30 border-t-white animate-spin shrink-0" />}
                      {safe_t("studio.batchSubmit", "Submit Batch Pipeline")}
                    </button>
                  </div>
                </form>
              </div>
            )}

            {activeTab === "social" && (
              <div className="p-6 lg:p-8 max-w-5xl mx-auto w-full">
                <div className="flex flex-col items-center justify-center min-h-[300px] bg-white rounded-2xl border border-dashed border-slate-300 shadow-sm p-8 text-center">
                  <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                    <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" /></svg>
                  </div>
                  <h3 className="text-[16px] font-bold text-slate-800 mb-2">{safe_t("studio.tabSocial", "Social Media Management")}</h3>
                  <p className="text-[14px] font-medium text-slate-500 max-w-sm leading-relaxed">{socialTaskId ? "Processing social hooks..." : safe_t("studio.socialEmpty", "Extracted social media content will be ready for publication here after final article processing.")}</p>
                </div>
              </div>
            )}

            {activeTab === "schema" && (
              <div className="p-6 lg:p-8 max-w-5xl mx-auto w-full">
                <div className="flex flex-col items-center justify-center min-h-[300px] bg-white rounded-2xl border border-dashed border-slate-300 shadow-sm p-8 text-center">
                  <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                    <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>
                  </div>
                  <h3 className="text-[16px] font-bold text-slate-800 mb-2">{safe_t("studio.tabSchema", "SEO Optimization")}</h3>
                  <p className="text-[14px] font-medium text-slate-500 max-w-sm leading-relaxed">{safe_t("studio.schemaEmpty", "SEO micro-transactions (FAQ) and structured metadata (JSON-LD) will be injected here for optimal indexing post-generation.")}</p>
                </div>
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
