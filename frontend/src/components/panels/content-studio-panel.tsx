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

  // Translation helpers
  const translateOptions = (opts: typeof TONE_OPTIONS) => opts.map(o => ({
    value: o.value,
    label: locale === "fa" ? o.fa : locale === "ar" ? o.ar : o.en
  })).concat(opts === POV_OPTIONS ? [] : [{ value: "__custom__", label: t("common.custom") || "Custom" }]);

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
    if (!selectedProjectId) return showToast("error", t("studio.selectProjectFirst"));
    setSubmitting(true);
    setTaskStatus(null);
    setSocialStatus(null);

    const resolvedTone = tone === "__custom__" ? customTone.trim() : labelFor(TONE_OPTIONS, tone);
    const resolvedStructure = structure === "__custom__" ? customStructure.trim() : labelFor(STRUCTURE_OPTIONS, structure);
    const resolvedAudience = audience === "__custom__" ? customAudience.trim() : labelFor(AUDIENCE_OPTIONS, audience);

    // Explicitly add safe fallback if "pointOfView" translation is missing in the backend
    const povTranslation = t("studio.pointOfView").replace("studio.pointOfView", "Point of View");

    const instructions = [
      extraInstructions.trim(),
      competitorUrl.trim() ? `Competitor URL: ${competitorUrl.trim()}` : "",
      sourceUrls.trim() ? `Source URLs:\n${sourceUrls.trim()}` : "",
      `Tone: ${resolvedTone}`,
      `Structure: ${resolvedStructure}`,
      `${povTranslation}: ${labelFor(POV_OPTIONS, pov)}`,
      `Target audience: ${resolvedAudience}`,
      "After article generation, preserve social repurposing outputs for LinkedIn/Twitter/Instagram.",
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
      showToast("success", t("studio.taskQueued").replace("{taskId}", payload.task_id));
    } catch (err) { showToast("error", extractError(err)); }
    finally { setSubmitting(false); }
  };

  const onSubmitBatch = async (e: FormEvent) => {
    e.preventDefault();
    if (!selectedProjectId) return showToast("error", t("studio.selectProjectFirst"));
    const topics = bulkTopics.split("\n").map(l => l.trim()).filter(Boolean);
    if (topics.length === 0 || topics.length > 20) return showToast("error", t("studio.maxTopics"));

    setBulkSubmitting(true);
    setBatchStatus(null);
    try {
      const payload = await apiRequest<{ batch_id: string }>("/content/generate/batch", {
        method: "POST", token,
        body: { project_id: selectedProjectId, topics, shared_keyword: bulkKeyword.trim() || undefined, language: bulkLanguage === "en" ? "en" : "fa" }
      });
      setBatchId(payload.batch_id);
      showToast("success", t("studio.batchProgress"));
    } catch (err) { showToast("error", extractError(err)); }
    finally { setBulkSubmitting(false); }
  };

  const tabEntries: { key: StudioTab; label: string }[] = [
    { key: "generate", label: t("studio.generateTab") },
    { key: "bulk", label: t("studio.bulkTab") },
    { key: "social", label: t("studio.socialTab") },
    { key: "schema", label: t("studio.schemaTab") },
  ];

  /* ── Spatial Depth: Main Content Area occupies the full canvas layout container ── */
  return (
    <section className="animate-fade-in relative flex items-start justify-center p-4 md:p-8 pt-4 pb-24 lg:pb-8 min-h-screen bg-[#F5F5F7]">

      {/* ── Dynamic Layout Wrapper ── */}
      <div className={clsx(
        "w-full transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] grid gap-8",
        taskStatus || batchStatus ? "grid-cols-1 xl:grid-cols-[1fr_360px] max-w-7xl" : "max-w-4xl grid-cols-1"
      )}>

        {/* ── Main Canvas Card ── */}
        <article className="flex relative flex-col border border-gray-100/80 bg-white rounded-3xl shadow-sm overflow-hidden min-w-0">

          {/* Apple Segmented Tab Control Header (Flush with card) */}
          <div className="bg-gray-50/50 border-b border-gray-100 px-6 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <h1 className="text-[20px] font-bold text-gray-900 tracking-tight">Studio</h1>

            <div className="inline-flex rounded-xl bg-gray-100 p-1 shrink-0 w-full md:w-auto overflow-x-auto no-scrollbar">
              {tabEntries.map((entry) => (
                <button
                  key={entry.key}
                  onClick={() => setActiveTab(entry.key)}
                  className={clsx(
                    "flex-1 md:flex-none text-[13px] font-medium px-4 py-1.5 rounded-lg transition-all duration-200 whitespace-nowrap",
                    activeTab === entry.key
                      ? "bg-white text-gray-900 shadow-sm ring-1 ring-black/5"
                      : "text-gray-500 hover:text-gray-700"
                  )}
                >
                  {entry.label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-auto">
            {activeTab === "generate" && (
              <form onSubmit={onSubmitGenerate} className="flex flex-col h-full">

                <div className="p-6 md:p-8 space-y-10 flex-1">
                  {/* SECTION 1: Core Identity */}
                  <section>
                    <h2 className="text-[15px] font-semibold text-gray-900 border-b border-gray-100 pb-2 mb-5">{t("studio.contentGeneration")}</h2>
                    <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
                      <div className="lg:col-span-1">
                        <InputField required label={t("studio.articleTopic")} value={topic} onChange={e => setTopic(e.target.value)} />
                      </div>
                      <div className="lg:col-span-1">
                        <InputField required label={t("studio.primaryKeyword")} value={keyword} onChange={e => setKeyword(e.target.value)} />
                      </div>
                      <div className="lg:col-span-1">
                        <SelectDropdown label={t("studio.language") || "Language"} value={language} onChange={v => setLanguage(v as any)} options={[
                          { value: "fa", label: "Persian (RTL)" },
                          { value: "ar", label: "Arabic (RTL)" },
                          { value: "en", label: "English (LTR)" },
                        ]} />
                      </div>
                    </div>
                  </section>

                  {/* SECTION 2: Tonal DNA */}
                  <section>
                    <h2 className="text-[15px] font-semibold text-gray-900 border-b border-gray-100 pb-2 mb-5">Tonal DNA</h2>
                    <div className="grid gap-5 md:grid-cols-2">
                      <div>
                        <SelectDropdown label={t("studio.tone")} value={tone} onChange={v => setTone(v)} options={translateOptions(TONE_OPTIONS)} />
                        {tone === "__custom__" && <div className="mt-2"><InputField required label={t("studio.toneCustom")} value={customTone} onChange={e => setCustomTone(e.target.value)} /></div>}
                      </div>
                      <div>
                        <SelectDropdown label={t("studio.structure")} value={structure} onChange={v => setStructure(v)} options={translateOptions(STRUCTURE_OPTIONS)} />
                        {structure === "__custom__" && <div className="mt-2"><InputField required label={t("studio.structureCustom")} value={customStructure} onChange={e => setCustomStructure(e.target.value)} /></div>}
                      </div>
                      <div>
                        <SelectDropdown label={t("studio.pointOfView").replace("studio.pointOfView", "Point of View")} value={pov} onChange={v => setPov(v)} options={translateOptions(POV_OPTIONS)} />
                      </div>
                      <div>
                        <SelectDropdown label={t("studio.audience")} value={audience} onChange={v => setAudience(v)} options={translateOptions(AUDIENCE_OPTIONS)} />
                        {audience === "__custom__" && <div className="mt-2"><InputField required label={t("studio.audienceCustom")} value={customAudience} onChange={e => setCustomAudience(e.target.value)} /></div>}
                      </div>
                    </div>
                  </section>

                  {/* SECTION 3: Parameters */}
                  <section>
                    <h2 className="text-[15px] font-semibold text-gray-900 border-b border-gray-100 pb-2 mb-5">Parameters & Limits</h2>
                    <div className="grid gap-5 md:grid-cols-2">
                      <div className="flex gap-3">
                        <div className="flex-1"><InputField type="number" label={t("studio.wordCountMin")} value={wordCountMin} onChange={e => setWordCountMin(e.target.value)} dir="ltr" /></div>
                        <div className="flex-1"><InputField type="number" label={t("studio.wordCountMax")} value={wordCountMax} onChange={e => setWordCountMax(e.target.value)} dir="ltr" /></div>
                      </div>
                      <div>
                        <label className="block text-[13px] font-medium text-gray-700 mb-2">{t("studio.creativity")}: <span className="text-teal-600 font-bold">{temperature.toFixed(1)}</span></label>
                        <input type="range" min="0" max="1" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 accent-teal-600 outline-none" dir="ltr" />
                      </div>
                    </div>
                  </section>

                  {/* SECTION 4: Context (Bidi Support) */}
                  <section>
                    <h2 className="text-[15px] font-semibold text-gray-900 border-b border-gray-100 pb-2 mb-5">Context & Instructions</h2>
                    <div className="space-y-4">
                      <InputField
                        label={t("studio.competitor")}
                        value={competitorUrl}
                        onChange={e => setCompetitorUrl(e.target.value)}
                        helperText={competitorUrl.trim() ? <span className="text-amber-600 flex items-center gap-1 font-medium"><svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg> {t("studio.skyscraperMode")}</span> : undefined}
                        dir="ltr" // BIDI Isolation
                      />

                      <div className="flex flex-col gap-[6px]">
                        <label className="text-[13px] font-medium text-gray-700">{t("studio.sourceUrls")}</label>
                        <textarea
                          className="w-full rounded-xl border border-gray-200 bg-white px-4 py-2.5 text-[14px] text-gray-900 outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 resize-none h-24"
                          value={sourceUrls}
                          onChange={e => setSourceUrls(e.target.value)}
                          dir="ltr" // URLs should remain LTR
                        />
                      </div>

                      <div className="flex flex-col gap-[6px]">
                        <label className="text-[13px] font-medium text-gray-700">{t("studio.extraInstructions")}</label>
                        <textarea
                          className="w-full rounded-xl border border-gray-200 bg-white px-4 py-2.5 text-[14px] text-gray-900 outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 resize-y min-h-[100px]"
                          value={extraInstructions}
                          onChange={e => setExtraInstructions(e.target.value)}
                        />
                      </div>
                    </div>
                  </section>
                </div>

                {/* ── Sticky Action Bar (Backdrop blur + Primary CTA) ── */}
                <div className="sticky bottom-0 z-10 p-5 bg-white/80 backdrop-blur-md border-t border-gray-100/60 mt-auto">
                  <Button type="submit" variant="primary" fullWidth size="lg" loading={submitting} disabled={!selectedProjectId}>
                    {t("studio.startGeneration")}
                  </Button>
                </div>

              </form>
            )}

            {/* Bulk Tab omitted for brevity in demo, assuming identical semantic group style */}
            {activeTab === "bulk" && (
              <div className="p-6 md:p-8 space-y-6">
                {/* Keeping original logic, just styled to match the new canvas */}
                <form onSubmit={onSubmitBatch} className="space-y-5">
                  <div className="flex flex-col gap-[6px]">
                    <label className="text-[13px] font-medium text-gray-700">{t("studio.bulkTopics")}</label>
                    <textarea required className="w-full rounded-xl border border-gray-200 bg-white px-4 py-2.5 text-[14px] text-gray-900 outline-none focus:border-teal-500 focus:ring-1 min-h-[150px]" value={bulkTopics} onChange={e => setBulkTopics(e.target.value)} />
                    <p className="text-[11px] text-gray-500">{t("studio.maxTopics")}</p>
                  </div>
                  <InputField label={t("studio.sharedKeyword")} value={bulkKeyword} onChange={e => setBulkKeyword(e.target.value)} />
                  <SelectDropdown label={t("studio.language") || "Language"} value={bulkLanguage} onChange={v => setBulkLanguage(v as any)} options={[
                    { value: "fa", label: "Persian (RTL)" },
                    { value: "ar", label: "Arabic (RTL)" },
                    { value: "en", label: "English (LTR)" },
                  ]} />
                  <Button type="submit" variant="primary" fullWidth size="lg" loading={bulkSubmitting} disabled={!selectedProjectId}>
                    {t("studio.submitBatch")}
                  </Button>
                </form>
              </div>
            )}

            {activeTab === "social" && (
              <div className="p-6 md:p-8">
                <h3 className="text-[18px] font-bold text-gray-900 mb-2">{t("studio.socialTab")}</h3>
                <p className="text-[13px] text-gray-500">{socialTaskId ? "Processing" : "Data will appear here after task completion"}</p>
              </div>
            )}

            {activeTab === "schema" && (
              <div className="p-6 md:p-8">
                <h3 className="text-[18px] font-bold text-gray-900 mb-2">{t("studio.schemaTab")}</h3>
                <p className="text-[13px] text-gray-500">Auto JSON-LD metadata for SEO injection.</p>
              </div>
            )}
          </div>
        </article>

        {/* ── Dynamic Layout Drawer: Active Tasks (only visible when a task is running) ── */}
        {(taskStatus || batchStatus) && (
          <aside className="animate-slide-in-start w-full bg-white rounded-3xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4 self-start sticky top-8">
            <h3 className="text-[16px] font-bold text-gray-900">{t("studio.taskStatus")}</h3>

            {taskStatus && (
              <div className="space-y-3 p-4 rounded-2xl bg-gray-50 border border-gray-100">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={clsx(
                      "w-2 h-2 rounded-full",
                      taskStatus.state === "SUCCESS" ? "bg-emerald-500" :
                        taskStatus.state === "FAILURE" ? "bg-red-500" :
                          "bg-teal-500 animate-pulse"
                    )} />
                    <span className="text-[12px] font-bold text-gray-700 tracking-wider">
                      {taskStatus.state}
                    </span>
                  </div>
                  {!taskStatus.ready && <span className="text-[11px] text-teal-600 animate-pulse font-medium">Live</span>}
                </div>
                <code className="block text-[11px] text-gray-400 font-mono truncate" dir="ltr">{taskStatus.task_id}</code>
                <p className="text-[13px] text-gray-600 leading-relaxed font-medium">{taskStatus.status}</p>
              </div>
            )}

            {batchStatus && (
              <div className="space-y-3 p-4 rounded-2xl bg-gray-50 border border-gray-100">
                <div className="flex items-center justify-between">
                  <h4 className="text-[13px] font-bold text-gray-900">{t("studio.batchProgress")}</h4>
                  <span className="text-[12px] font-mono bg-white px-2 py-0.5 rounded-md border border-gray-100 shadow-sm text-gray-600">
                    {batchStatus.completed}/{batchStatus.total}
                  </span>
                </div>
                <div className="h-1.5 w-full bg-gray-200 rounded-full overflow-hidden">
                  <div className="h-full bg-teal-500 transition-all duration-500 ease-out" style={{ width: `${batchStatus.total ? (batchStatus.completed / batchStatus.total) * 100 : 0}%` }} />
                </div>
              </div>
            )}
          </aside>
        )}
      </div>
    </section>
  );
}
