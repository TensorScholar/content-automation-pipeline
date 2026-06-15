"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import { LlmOptionsResponse, ProjectReadiness, TaskStatusResponse } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";

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
interface SocialPost {
  platform: string;
  content: string;
}

type StudioTab = "generate" | "bulk" | "social" | "schema";
type GenerationLanguage = "fa" | "en";

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

const READINESS_COPY = {
  en: {
    ready: "Project is ready for generation.",
    warning: "Project has readiness warnings. Generation is still available.",
    blocked: "Generation is blocked until runtime dependencies are restored.",
    publishingBlocked: "Publishing is blocked. Generation is still available.",
    checking: "Checking project readiness...",
  },
  fa: {
    ready: "پروژه برای تولید محتوا آماده است.",
    warning: "پروژه هشدار آماده‌سازی دارد، اما تولید محتوا فعال است.",
    blocked: "تا بازیابی وابستگی‌های اجرایی، تولید محتوا مسدود است.",
    publishingBlocked: "انتشار مسدود است، اما تولید محتوا فعال است.",
    checking: "در حال بررسی آمادگی پروژه...",
  },
  ar: {
    ready: "المشروع جاهز لإنشاء المحتوى.",
    warning: "توجد تحذيرات جاهزية، لكن الإنشاء متاح.",
    blocked: "إنشاء المحتوى محظور حتى استعادة الاعتماديات التشغيلية.",
    publishingBlocked: "النشر محظور، لكن إنشاء المحتوى متاح.",
    checking: "جارٍ فحص جاهزية المشروع...",
  },
};

const MODEL_COPY = {
  en: {
    label: "AI model",
    loading: "Checking model access...",
    unavailable: "No configured AI model is available. Ask a manager to add a key.",
    active: "Active",
    recommended: "Recommended",
    warning: "Provider quota or credits may be exhausted. Try another configured model.",
  },
  fa: {
    label: "مدل هوش مصنوعی",
    loading: "در حال بررسی دسترسی مدل...",
    unavailable: "مدل هوش مصنوعی پیکربندی‌شده‌ای در دسترس نیست. از مدیر بخواهید کلید API اضافه کند.",
    active: "فعال",
    recommended: "پیشنهادی",
    warning: "ممکن است سهمیه یا اعتبار ارائه‌دهنده تمام شده باشد. یک مدل پیکربندی‌شده دیگر را امتحان کنید.",
  },
  ar: {
    label: "نموذج الذكاء الاصطناعي",
    loading: "جارٍ فحص الوصول إلى النموذج...",
    unavailable: "لا يوجد نموذج ذكاء اصطناعي مهيأ. اطلب من المدير إضافة مفتاح API.",
    active: "نشط",
    recommended: "موصى به",
    warning: "قد تكون الحصة أو الرصيد لدى المزود قد نفدت. جرّب نموذجاً مهيأ آخر.",
  },
};

function extractError(e: unknown): string {
  if (e instanceof ApiError) return e.detail;
  return "Unexpected error";
}

function refreshTask(taskId: string, token: string, signal?: AbortSignal) {
  return apiRequest<TaskStatusResponse>(`/content/task/${taskId}`, { token, signal });
}

function buildWordCountPayload(minRaw: string, maxRaw: string): {
  error?: string;
  payload: { word_count_range?: string; target_word_count?: number };
} {
  const min = minRaw.trim() ? Number(minRaw) : undefined;
  const max = maxRaw.trim() ? Number(maxRaw) : undefined;

  if ((min !== undefined && (!Number.isFinite(min) || min < 1)) || (max !== undefined && (!Number.isFinite(max) || max < 1))) {
    return { error: "Word count must be a positive number.", payload: {} };
  }
  if (min !== undefined && max !== undefined && min > max) {
    return { error: "Minimum word count cannot exceed maximum word count.", payload: {} };
  }
  if (min !== undefined && max !== undefined) {
    return { payload: { word_count_range: `${Math.round(min)}-${Math.round(max)}` } };
  }
  if (max !== undefined) {
    return { payload: { target_word_count: Math.round(max) } };
  }
  if (min !== undefined) {
    return { payload: { target_word_count: Math.round(min) } };
  }
  return { payload: {} };
}

function getSocialPosts(status: TaskStatusResponse | null): SocialPost[] {
  const posts = status?.result?.posts;
  if (typeof posts !== "object" || posts === null || Array.isArray(posts)) {
    return [];
  }

  return Object.entries(posts)
    .filter((entry): entry is [string, string] => typeof entry[1] === "string" && entry[1].trim().length > 0)
    .map(([platform, content]) => ({ platform, content }));
}

function downloadTextFile(content: string, filename: string, type = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function platformLabel(platform: string) {
  if (platform === "twitter") return "X / Twitter";
  return platform.charAt(0).toUpperCase() + platform.slice(1);
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
  const [language, setLanguage] = useState<GenerationLanguage>("fa");
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
  const [schemaLoading, setSchemaLoading] = useState(false);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [schemaTitle, setSchemaTitle] = useState("");

  // Bulk state
  const [bulkTopics, setBulkTopics] = useState("");
  const [bulkKeyword, setBulkKeyword] = useState("");
  const [bulkLanguage, setBulkLanguage] = useState<GenerationLanguage>("fa");
  const [bulkSubmitting, setBulkSubmitting] = useState(false);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [batchStatus, setBatchStatus] = useState<BatchStatusResponse | null>(null);
  const [readiness, setReadiness] = useState<ProjectReadiness | null>(null);
  const [readinessLoading, setReadinessLoading] = useState(false);
  const [llmOptions, setLlmOptions] = useState<LlmOptionsResponse | null>(null);
  const [llmOptionsLoading, setLlmOptionsLoading] = useState(false);
  const [modelOverride, setModelOverride] = useState("");

  // Fallback Translation Helper (Eradicate Leaking Keys)
  const safe_t = (key: string, fallback: string) => {
    const val = t(key as any);
    return val && val !== key ? val : fallback;
  };
  const customFallback = locale === "fa" ? "سفارشی" : locale === "ar" ? "مخصص" : "Custom";
  const customValuePlaceholder = locale === "fa" ? "مقدار دلخواه را وارد کنید" : locale === "ar" ? "أدخل قيمة مخصصة" : "Enter custom value";
  const contextPlaceholder = locale === "fa" ? "- روی مورد X تمرکز شود..." : locale === "ar" ? "- ركّز على العنصر X..." : "- Focus on topic X...";
  const sourcePlaceholder = locale === "fa" ? "https://example.com/page-1\nhttps://example.com/page-2" : locale === "ar" ? "https://example.com/page-1\nhttps://example.com/page-2" : "https://example.com/page-1\nhttps://example.com/page-2";
  const socialLoadingText = locale === "fa" ? "در حال پردازش خروجی‌های شبکه اجتماعی..." : locale === "ar" ? "جاري معالجة مخرجات الشبكات الاجتماعية..." : "Processing social outputs...";
  const socialReadyText = locale === "fa" ? "خروجی‌های آماده انتشار" : locale === "ar" ? "مخرجات جاهزة للنشر" : "Ready-to-publish outputs";
  const socialMissingText = locale === "fa" ? "برای ساخت خروجی‌های شبکه اجتماعی، ابتدا یک مقاله تولید کنید." : locale === "ar" ? "أنشئ مقالة أولاً لإعداد مخرجات الشبكات الاجتماعية." : "Generate an article first to prepare social outputs.";
  const schemaMissingText = locale === "fa" ? "برای تولید JSON-LD و خروجی HTML، ابتدا یک مقاله موفق تولید کنید." : locale === "ar" ? "أنشئ مقالة ناجحة أولاً لتوليد JSON-LD و HTML." : "Generate a successful article first to produce JSON-LD and HTML export.";
  const schemaReadyText = locale === "fa" ? "متادیتای ساختاریافته آماده است" : locale === "ar" ? "البيانات المنظمة جاهزة" : "Structured metadata is ready";
  const loadingSchemaText = locale === "fa" ? "در حال آماده‌سازی JSON-LD و خروجی HTML..." : locale === "ar" ? "جاري إعداد JSON-LD و HTML..." : "Preparing JSON-LD and HTML export...";
  const modelCopy = MODEL_COPY[locale];

  const translateOptions = (opts: typeof TONE_OPTIONS) => opts.map(o => ({
    value: o.value,
    label: locale === "fa" ? o.fa : locale === "ar" ? o.ar : o.en
  })).concat([{ value: "__custom__", label: safe_t("common.custom", customFallback) }]);

  const labelFor = (opts: typeof TONE_OPTIONS, val: string) => {
    const found = opts.find((o) => o.value === val);
    if (!found) return val;
    return locale === "fa" ? found.fa : locale === "ar" ? found.ar : found.en;
  };

  const articleId = useMemo(() => taskStatus?.result?.article_id ?? null, [taskStatus]);
  const socialTaskId = useMemo(() => taskStatus?.result?.social_task_id ? String(taskStatus.result.social_task_id) : null, [taskStatus]);
  const socialPosts = useMemo(() => getSocialPosts(socialStatus), [socialStatus]);
  const schemaJson = useMemo(() => jsonld ? JSON.stringify(jsonld, null, 2) : "", [jsonld]);

  const copyToClipboard = useCallback(async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      showToast("success", t("common.copied"));
    } catch {
      showToast("error", t("common.unexpectedError"));
    }
  }, [showToast, t]);

  // Reset state on project change
  useEffect(() => {
    setTaskStatus(null); setSocialStatus(null); setJsonld(null); setExportHtml("");
    setSchemaError(null); setSchemaLoading(false); setSchemaTitle("");
    setBatchId(null); setBatchStatus(null);
  }, [selectedProjectId]);

  useEffect(() => {
    if (!selectedProjectId) {
      setReadiness(null);
      setReadinessLoading(false);
      return;
    }

    const controller = new AbortController();
    setReadinessLoading(true);
    apiRequest<ProjectReadiness>(`/projects/${selectedProjectId}/readiness`, {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => {
        if (!controller.signal.aborted) setReadiness(payload);
      })
      .catch(() => {
        if (!controller.signal.aborted) setReadiness(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setReadinessLoading(false);
      });

    return () => controller.abort();
  }, [selectedProjectId, token]);

  useEffect(() => {
    const controller = new AbortController();
    setLlmOptionsLoading(true);
    apiRequest<LlmOptionsResponse>("/system/llm/options", {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => {
        if (controller.signal.aborted) return;
        setLlmOptions(payload);
        const activeAvailable = payload.selectable_models.some((option) => option.model === payload.active_model);
        const nextModel = activeAvailable
          ? payload.active_model
          : payload.selectable_models[0]?.model ?? "";
        setModelOverride((current) => current || nextModel);
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setLlmOptions(null);
          setModelOverride("");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLlmOptionsLoading(false);
      });

    return () => controller.abort();
  }, [token]);

  // Poll tasks without overlapping backend requests.
  // FIX: Prevent memory leak by tracking polling state
  useEffect(() => {
    if (!taskStatus?.task_id || taskStatus.ready) return;
    const controller = new AbortController();
    const taskId = taskStatus.task_id;
    let mounted = true;
    let isPolling = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const schedule = () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      timeoutId = setTimeout(() => { void poll(); }, 4000);
    };

    const poll = async () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      isPolling = true;

      try {
        const payload = await refreshTask(taskId, token, controller.signal);
        if (!mounted || controller.signal.aborted) return;
        setTaskStatus(payload);
        isPolling = false;
        if (!payload.ready && mounted && !controller.signal.aborted) {
          schedule();
        }
      } catch {
        if (!mounted || controller.signal.aborted) return;
        isPolling = false;
        if (mounted && !controller.signal.aborted) {
          schedule();
        }
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
  }, [taskStatus?.task_id, taskStatus?.ready, token]);

  useEffect(() => {
    if (!socialTaskId) return;
    const controller = new AbortController();
    let mounted = true;
    let isPolling = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const schedule = () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      timeoutId = setTimeout(() => { void poll(); }, 4000);
    };

    const poll = async () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      isPolling = true;

      try {
        const payload = await refreshTask(socialTaskId, token, controller.signal);
        if (!mounted || controller.signal.aborted) return;
        setSocialStatus(payload);
        isPolling = false;
        if (!payload.ready && mounted && !controller.signal.aborted) {
          schedule();
        }
      } catch {
        if (!mounted || controller.signal.aborted) return;
        isPolling = false;
        if (mounted && !controller.signal.aborted) {
          schedule();
        }
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
  }, [socialTaskId, token]);

  useEffect(() => {
    if (!articleId || taskStatus?.state !== "SUCCESS") {
      setJsonld(null);
      setExportHtml("");
      setSchemaError(null);
      setSchemaLoading(false);
      setSchemaTitle("");
      return;
    }

    const controller = new AbortController();
    setSchemaLoading(true);
    setSchemaError(null);

    const loadSchemaAssets = async () => {
      const [jsonResult, htmlResult] = await Promise.allSettled([
        apiRequest<JsonLdResponse>(`/content/${articleId}/schema/jsonld`, {
          token,
          signal: controller.signal,
          timeoutMs: 15000,
        }),
        apiRequest<HtmlExportResponse>(`/content/${articleId}/export/html`, {
          token,
          signal: controller.signal,
          timeoutMs: 15000,
        }),
      ]);

      if (controller.signal.aborted) return;

      const nextJson = jsonResult.status === "fulfilled" ? jsonResult.value.schema : null;
      const nextHtml = htmlResult.status === "fulfilled" ? htmlResult.value.html : "";
      setJsonld(nextJson);
      setExportHtml(nextHtml);
      setSchemaTitle(htmlResult.status === "fulfilled" ? htmlResult.value.title : "");

      if (!nextJson && !nextHtml) {
        const error = jsonResult.status === "rejected" ? jsonResult.reason : htmlResult.status === "rejected" ? htmlResult.reason : null;
        setSchemaError(extractError(error));
      }
    };

    void loadSchemaAssets().finally(() => {
      if (!controller.signal.aborted) setSchemaLoading(false);
    });

    return () => controller.abort();
  }, [articleId, taskStatus?.state, token]);

  // Poll bulk without overlapping backend requests.
  // FIX: Prevent memory leak by tracking polling state
  useEffect(() => {
    if (!batchId) return;
    const controller = new AbortController();
    let mounted = true;
    let isPolling = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const schedule = () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      timeoutId = setTimeout(() => { void poll(); }, 5000);
    };

    const poll = async () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      isPolling = true;

      try {
        const status = await apiRequest<BatchStatusResponse>(`/content/batch/${batchId}/status`, {
          token,
          signal: controller.signal,
          timeoutMs: 10000,
        });
        if (!mounted || controller.signal.aborted) return;
        setBatchStatus(status);
        isPolling = false;
        if (status.status === "completed" || status.status === "failed") {
          setBatchId(null);
          return;
        }
        if (mounted && !controller.signal.aborted) {
          schedule();
        }
      } catch {
        if (!mounted || controller.signal.aborted) return;
        isPolling = false;
        if (mounted && !controller.signal.aborted) {
          schedule();
        }
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
  }, [batchId, token]);

  const onSubmitGenerate = async (e: FormEvent) => {
    e.preventDefault();
    if (!selectedProjectId) return showToast("error", safe_t("studio.selectProjectFirst", "Select a project first"));
    if (readiness && !readiness.can_generate) return showToast("error", READINESS_COPY[locale].blocked);
    setSubmitting(true);
    setTaskStatus(null);
    setSocialStatus(null);
    setJsonld(null);
    setExportHtml("");
    setSchemaError(null);
    setSchemaTitle("");

    const resolvedTone = tone === "__custom__" ? customTone.trim() : labelFor(TONE_OPTIONS, tone);
    const resolvedStructure = structure === "__custom__" ? customStructure.trim() : labelFor(STRUCTURE_OPTIONS, structure);
    const resolvedAudience = audience === "__custom__" ? customAudience.trim() : labelFor(AUDIENCE_OPTIONS, audience);
    const wordCount = buildWordCountPayload(wordCountMin, wordCountMax);

    if (wordCount.error) {
      setSubmitting(false);
      return showToast("error", wordCount.error);
    }

    const instructions = [
      language === "en" ? "Output language must be English." : "Output language must be Persian (Farsi).",
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
          additional_instructions: instructions, language, temperature,
          model_override: modelOverride || undefined,
          ...wordCount.payload,
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
    if (readiness && !readiness.can_generate) return showToast("error", READINESS_COPY[locale].blocked);
    const topics = bulkTopics.split("\n").map(l => l.trim()).filter(Boolean);
    if (topics.length === 0 || topics.length > 20) return showToast("error", safe_t("studio.maxTopics", "Max 20 topics allowed"));

    setBulkSubmitting(true);
    setBatchStatus(null);
    try {
      const bulkInstructions = [
        bulkKeyword.trim() ? `Primary keyword for all topics: ${bulkKeyword.trim()}` : "",
        bulkLanguage === "en"
            ? "Output language must be English."
            : "Output language must be Persian (Farsi).",
      ].filter(Boolean).join("\n");

      const payload = await apiRequest<{ batch_id: string }>("/content/generate/batch", {
        method: "POST", token,
        body: {
          project_id: selectedProjectId,
          topics,
          priority: "high",
          custom_instructions: bulkInstructions || undefined,
          model_override: modelOverride || undefined,
          language: bulkLanguage,
        }
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
    "w-full min-h-[36px] rounded-xl border border-black/5 bg-slate-50 px-3 py-2 text-[14px] font-medium text-slate-900 outline-none transition-colors duration-150 placeholder:text-slate-400 focus:border-emerald-500 focus:bg-white focus:ring-2 focus:ring-emerald-500/20 dark:border-white/10 dark:bg-surface-alt dark:text-gray-100 dark:placeholder:text-gray-400 dark:focus:bg-surface",
    locale === 'fa' || locale === 'ar' ? "" : ""
  );

  const LabelClass = "mb-1.5 max-w-max text-[12px] font-medium leading-none text-slate-500 dark:text-gray-300";

  const HeaderClass = "mb-5 border-b border-black/5 pb-3 text-[13px] font-semibold text-slate-500 dark:border-white/10 dark:text-gray-300";
  const readinessState = readinessLoading
    ? "checking"
    : readiness && !readiness.can_generate
      ? "blocked"
      : readiness && (readiness.status === "warning" || readiness.status === "blocked")
        ? "warning"
        : "ready";
  const generationBlocked = !!readiness && !readiness.can_generate;
  const publishingBlocked = !!readiness && readiness.can_generate && !readiness.can_publish;
  const readinessMessage =
    readinessState === "checking"
      ? READINESS_COPY[locale].checking
      : readinessState === "blocked"
        ? READINESS_COPY[locale].blocked
        : publishingBlocked
          ? READINESS_COPY[locale].publishingBlocked
          : readinessState === "warning"
          ? READINESS_COPY[locale].warning
          : READINESS_COPY[locale].ready;
  const readinessDetail = readiness?.blocking_items[0]?.message ?? readiness?.warnings[0]?.message ?? null;
  const selectableModels = llmOptions?.selectable_models ?? [];
  const selectedModelOption = selectableModels.find((option) => option.model === modelOverride);
  const llmUnavailable = !llmOptionsLoading && selectableModels.length === 0;
  const llmUserMessage = llmOptions?.user_message && llmOptions.user_message !== "AI generation is available."
    ? llmOptions.user_message
    : null;
  const llmWarning = llmOptions?.warnings[0] ?? llmUserMessage ?? (llmUnavailable ? modelCopy.unavailable : null);
  const selectedActiveProviderUnavailable =
    !!llmUserMessage && selectedModelOption?.provider === llmOptions?.active_provider;
  const selectedModelUnavailable = llmUnavailable || selectedActiveProviderUnavailable;

  return (
    <section className="animate-fade-in relative flex h-full min-h-0 min-w-0 w-full justify-center overflow-hidden">

      {/* ── Dynamic Layout Wrapper ── */}
      <div className={clsx(
        "grid h-full min-h-0 min-w-0 w-full gap-4 transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)]",
        taskStatus || batchStatus ? "grid-cols-1 xl:grid-cols-[minmax(0,1fr)_360px] max-w-7xl" : "grid-cols-1"
      )}>

        {/* ── LAYER 2: THE FIXED CANVAS (The "Page") ── */}
        <article className="relative m-4 flex h-[calc(100%-2rem)] min-h-0 min-w-0 flex-col overflow-hidden rounded-xl border border-black/5 bg-white dark:border-white/10 dark:bg-surface">

          {/* Apple Segmented Tab Control Header (Fixed) */}
          <div className="z-10 flex min-w-0 shrink-0 items-center justify-between border-b border-black/5 bg-gray-50 p-4 dark:border-white/10 dark:bg-surface-alt">

            <div className="inline-flex max-w-full flex-wrap rounded-md bg-slate-200 p-1 dark:bg-white/10 sm:w-auto">
              {tabEntries.map((entry) => (
                <button
                  type="button"
                  key={entry.key}
                  onClick={() => setActiveTab(entry.key)}
                  className={clsx(
                    "flex-1 rounded-md px-3 py-1.5 text-center text-[13px] font-medium leading-5 transition-colors duration-150 sm:flex-none",
                    activeTab === entry.key
                      ? "bg-white text-slate-900 dark:bg-white/15 dark:text-gray-100"
                      : "bg-transparent text-slate-500 dark:text-gray-400 hover:text-slate-700 dark:hover:text-gray-200"
                  )}
                >
                  {entry.label}
                </button>
              ))}
            </div>

          </div>

          <div className="min-h-0 w-full flex-1 overflow-x-hidden overflow-y-auto">
            {selectedProjectId && (activeTab === "generate" || activeTab === "bulk") && (
              <div className="px-4 pt-4 lg:px-5">
                <div className={clsx(
                  "mx-auto flex w-full max-w-5xl items-start justify-between gap-3 rounded-xl border px-4 py-3 text-[13px] shadow-sm",
                  readinessState === "blocked"
                    ? "border-red-200 bg-red-50 text-red-900 dark:border-red-400/20 dark:bg-red-500/10 dark:text-red-100"
                    : readinessState === "warning"
                      ? "border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-400/20 dark:bg-amber-500/10 dark:text-amber-100"
                      : "border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-400/20 dark:bg-emerald-500/10 dark:text-emerald-100"
                )}>
                  <div className="flex min-w-0 items-start gap-3">
                    <span className={clsx(
                      "mt-1 h-2 w-2 shrink-0 rounded-full",
                      readinessState === "checking" ? "animate-pulse bg-slate-400" :
                        readinessState === "blocked" ? "bg-red-500" :
                          readinessState === "warning" ? "bg-amber-500" : "bg-emerald-500"
                    )} />
                    <div className="min-w-0">
                      <p className="font-semibold leading-5">{readinessMessage}</p>
                      {readinessDetail && <p className="mt-0.5 truncate text-[12px] opacity-75">{readinessDetail}</p>}
                    </div>
                  </div>
                  {readiness && (
                    <span className="shrink-0 rounded-full bg-white/70 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-normal text-slate-700 dark:bg-white/10 dark:text-white/80">
                      {readiness.status}
                    </span>
                  )}
                </div>
              </div>
            )}

            {activeTab === "generate" && (
              <form onSubmit={onSubmitGenerate} className="flex min-h-full min-w-0 flex-col p-4 lg:p-5">

                <div className="mx-auto flex min-w-0 w-full max-w-5xl flex-col gap-4">
                  {/* ── LAYER 3: THE INTERNAL BOXES (The Form Cards) ── */}
                  <div className="grid min-w-0 grid-cols-12 gap-4">

                    {/* BOX 1: Core Identity */}
                    <div className="col-span-12 flex flex-col gap-4 rounded-xl border border-black/5 bg-white p-5 dark:border-white/10 dark:bg-surface-alt lg:col-span-6">
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
                          <select className={InputClass} value={language} onChange={e => setLanguage(e.target.value as GenerationLanguage)}>
                            <option value="fa">{safe_t("lang.fa", "فارسی")}</option>
                            <option value="en">{safe_t("lang.en", "English")}</option>
                          </select>
                        </div>
                      </div>
                    </div>

                    {/* BOX 2: Tonal DNA */}
                    <div className="col-span-12 flex flex-col gap-4 rounded-xl border border-black/5 bg-white p-5 dark:border-white/10 dark:bg-surface-alt lg:col-span-6">
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
                                <input autoFocus placeholder={customValuePlaceholder} className={InputClass} value={customTone} onChange={e => setCustomTone(e.target.value)} />
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
                                <input autoFocus placeholder={customValuePlaceholder} className={InputClass} value={customAudience} onChange={e => setCustomAudience(e.target.value)} />
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
                                <input autoFocus placeholder={customValuePlaceholder} className={InputClass} value={customStructure} onChange={e => setCustomStructure(e.target.value)} />
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
                                <input autoFocus placeholder={customValuePlaceholder} className={InputClass} value={customPov} onChange={e => setCustomPov(e.target.value)} />
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* BOX 3: Additional Context */}
                    <div className="col-span-12 flex flex-col gap-4 rounded-xl border border-black/5 bg-white p-5 dark:border-white/10 dark:bg-surface-alt">
                      <h2 className={HeaderClass}>{safe_t("studio.context", "Additional Context")}</h2>
                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{safe_t("studio.reference", "Reference URL (Analysis)")}</label>
                        <input placeholder="https://" className={clsx(InputClass, "font-mono text-[13px]")} value={competitorUrl} onChange={e => setCompetitorUrl(e.target.value)} dir="ltr" />
                      </div>
                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{safe_t("studio.extraInstructions", "Extra Instructions")}</label>
                        <textarea rows={2} placeholder={contextPlaceholder} className={clsx(InputClass, "h-[74px] resize-none")} value={extraInstructions} onChange={e => setExtraInstructions(e.target.value)} />
                      </div>
                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{safe_t("studio.sourceUrls", "Source URLs")}</label>
                        <textarea rows={2} placeholder={sourcePlaceholder} className={clsx(InputClass, "h-[74px] resize-none font-mono text-[13px]")} value={sourceUrls} onChange={e => setSourceUrls(e.target.value)} dir="ltr" />
                      </div>
                    </div>

                    {/* BOX 4: Parameters & Action */}
                    <div className="col-span-12 flex flex-col gap-5 rounded-xl border border-black/5 bg-white p-5 dark:border-white/10 dark:bg-surface-alt">
                      <h2 className={HeaderClass}>{safe_t("studio.parametersLimits", "Parameters & Limits")}</h2>

                      <div className="flex flex-col gap-8">
                        {/* Sliders Area */}
                        <div className="flex flex-col gap-6">
                          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(220px,auto)]">
                            <div className="flex flex-col gap-0">
                              <label className={LabelClass}>{modelCopy.label}</label>
                              <select
                                className={InputClass}
                                value={modelOverride}
                                onChange={(e) => setModelOverride(e.target.value)}
                                disabled={llmOptionsLoading || selectableModels.length === 0}
                              >
                                {llmOptionsLoading ? (
                                  <option value="">{modelCopy.loading}</option>
                                ) : selectableModels.length > 0 ? (
                                  selectableModels.map((option) => (
                                    <option key={`${option.provider}:${option.model}`} value={option.model}>
                                      {option.label} · {option.provider}
                                    </option>
                                  ))
                                ) : (
                                  <option value="">{modelCopy.unavailable}</option>
                                )}
                              </select>
                            </div>
                            <div className="flex min-h-[36px] items-center rounded-xl border border-black/5 bg-slate-50 px-3 py-2 text-[12px] font-medium text-slate-600 dark:border-white/10 dark:bg-white/[0.04] dark:text-gray-300">
                              {selectedModelOption
                                ? `${selectedModelOption.recommended ? modelCopy.recommended : modelCopy.active} · ${selectedModelOption.model}`
                                : llmOptionsLoading
                                  ? modelCopy.loading
                                  : modelCopy.unavailable}
                            </div>
                          </div>

                          {llmWarning && (
                            <div className="rounded-xl border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-[12px] font-medium leading-5 text-amber-800 dark:text-amber-200">
                              {llmWarning}
                            </div>
                          )}

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
                              <input type="range" min="0" max="1" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="flex-1 h-2 rounded-full appearance-none bg-slate-200 dark:bg-white/10 accent-emerald-600 outline-none" dir="ltr" />
                              <span className="w-12 shrink-0 rounded-lg bg-emerald-100 px-3 py-1 text-center text-[14px] font-semibold text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-200">{temperature.toFixed(1)}</span>
                            </div>
                            <p className="text-[12px] text-slate-500 dark:text-gray-400 mt-1 font-medium">0.1 - 0.4: {safe_t("studio.creativityConservative", "Conservative")} | 0.5 - 0.7: {safe_t("studio.creativityBalanced", "Balanced")} | 0.8 - 1.0: {safe_t("studio.creativityCreative", "Creative")}</p>
                          </div>
                        </div>

                        {/* Button Action Area */}
                        <div className="flex w-full justify-end mt-2 pt-6 border-t border-black/5 dark:border-white/10">
                          <button
                            type="submit"
                            disabled={!selectedProjectId || submitting || generationBlocked || selectedModelUnavailable}
                            className="flex w-full items-center justify-center gap-2 rounded-xl bg-brand px-6 py-2.5 font-semibold text-white transition-colors duration-150 hover:bg-brand-hover active:bg-brand-hover disabled:opacity-50 md:w-auto"
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
                <form onSubmit={onSubmitBatch} className="mx-auto flex h-full w-full max-w-4xl flex-col gap-4">
                  <div className="flex flex-col gap-4 rounded-xl border border-black/5 bg-white p-5 dark:border-white/10 dark:bg-surface-alt">
                    <h2 className={HeaderClass}>{safe_t("studio.parametersLimits", "Parameters")}</h2>
                    <div className="flex flex-col md:flex-row gap-4">
                      <div className="flex-1 flex flex-col gap-0">
                        <label className={LabelClass}>{safe_t("studio.batchLanguage", "Language")}</label>
                        <select className={InputClass} value={bulkLanguage} onChange={e => setBulkLanguage(e.target.value as GenerationLanguage)}>
                          <option value="fa">{safe_t("lang.fa", "فارسی")}</option>
                          <option value="en">{safe_t("lang.en", "English")}</option>
                        </select>
                      </div>
                      <div className="flex-1 flex flex-col gap-0">
                        <label className={LabelClass}>{safe_t("studio.batchSharedKeyword", "Shared Keyword")}</label>
                        <input className={InputClass} value={bulkKeyword} onChange={e => setBulkKeyword(e.target.value)} />
                      </div>
                    </div>
                    <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(220px,auto)]">
                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{modelCopy.label}</label>
                        <select
                          className={InputClass}
                          value={modelOverride}
                          onChange={(e) => setModelOverride(e.target.value)}
                          disabled={llmOptionsLoading || selectableModels.length === 0}
                        >
                          {llmOptionsLoading ? (
                            <option value="">{modelCopy.loading}</option>
                          ) : selectableModels.length > 0 ? (
                            selectableModels.map((option) => (
                              <option key={`bulk:${option.provider}:${option.model}`} value={option.model}>
                                {option.label} · {option.provider}
                              </option>
                            ))
                          ) : (
                            <option value="">{modelCopy.unavailable}</option>
                          )}
                        </select>
                      </div>
                      <div className="flex min-h-[36px] items-center rounded-xl border border-black/5 bg-slate-50 px-3 py-2 text-[12px] font-medium text-slate-600 dark:border-white/10 dark:bg-white/[0.04] dark:text-gray-300">
                        {selectedModelOption
                          ? `${selectedModelOption.recommended ? modelCopy.recommended : modelCopy.active} · ${selectedModelOption.model}`
                          : llmOptionsLoading
                            ? modelCopy.loading
                            : modelCopy.unavailable}
                      </div>
                    </div>
                    {llmWarning && (
                      <div className="rounded-xl border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-[12px] font-medium leading-5 text-amber-800 dark:text-amber-200">
                        {llmWarning}
                      </div>
                    )}
                  </div>

                  <div className="flex flex-1 flex-col rounded-xl border border-black/5 bg-white p-5 dark:border-white/10 dark:bg-surface-alt">
                    <h2 className={HeaderClass}>{safe_t("studio.coreIdentity", "Core Identity")}</h2>
                    <div className="flex flex-col gap-0 flex-1 h-full min-h-[300px]">
                      <label className={LabelClass}>{safe_t("studio.batchTopics", "Topics (One per line)")}</label>
                      <textarea required className={clsx(InputClass, "resize-none h-full py-4")} placeholder={safe_t("studio.batchTopicsPlaceholder", "Topic 1\nTopic 2\nTopic 3")} value={bulkTopics} onChange={e => setBulkTopics(e.target.value)} />
                      <p className="text-[12px] font-bold text-slate-400 dark:text-gray-300 mt-2">{safe_t("studio.maxTopics", "Max 20 topics allowed.")}</p>
                    </div>
                  </div>

                  <div className="flex w-full justify-end mt-2 pb-8">
                    <button type="submit" disabled={!selectedProjectId || bulkSubmitting || generationBlocked || selectedModelUnavailable} className="flex w-full items-center justify-center gap-2 rounded-xl bg-brand px-6 py-2.5 font-semibold text-white transition-colors duration-150 hover:bg-brand-hover active:bg-brand-hover disabled:opacity-50 md:w-auto">
                      {bulkSubmitting && <span className="w-5 h-5 rounded-full border-[3px] border-white/30 border-t-white animate-spin shrink-0" />}
                      {safe_t("studio.batchSubmit", "Submit Batch Pipeline")}
                    </button>
                  </div>
                </form>
              </div>
            )}

            {activeTab === "social" && (
              <div className="p-6 lg:p-8 max-w-5xl mx-auto w-full">
                {socialPosts.length > 0 ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-black/5 pb-4 dark:border-white/10">
                      <div>
                        <h3 className="text-[18px] font-semibold text-slate-900 dark:text-gray-100">{safe_t("studio.tabSocial", "Social Media Management")}</h3>
                        <p className="mt-1 text-[13px] font-medium text-slate-500 dark:text-gray-300">{socialReadyText}</p>
                      </div>
                      <span className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-[12px] font-semibold text-emerald-700 dark:text-emerald-300">
                        {socialStatus?.state ?? "SUCCESS"}
                      </span>
                    </div>

                    <div className="grid gap-4 lg:grid-cols-3">
                      {socialPosts.map((post) => (
                        <article key={post.platform} className="flex min-h-[260px] flex-col rounded-xl border border-black/5 bg-white p-4 dark:border-white/10 dark:bg-surface-alt">
                          <div className="mb-3 flex items-center justify-between gap-3">
                            <h4 className="text-[13px] font-semibold text-slate-900 dark:text-gray-100">{platformLabel(post.platform)}</h4>
                            <Button variant="ghost" size="sm" onClick={() => void copyToClipboard(post.content)}>
                              {t("common.copy")}
                            </Button>
                          </div>
                          <textarea
                            readOnly
                            value={post.content}
                            className="min-h-0 flex-1 resize-none rounded-lg border border-black/5 bg-slate-50 p-3 text-[13px] leading-6 text-slate-700 outline-none dark:border-white/10 dark:bg-white/[0.04] dark:text-gray-200"
                            dir="auto"
                          />
                        </article>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex min-h-[300px] flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center dark:border-white/10 dark:bg-surface-alt">
                    <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-slate-50 dark:bg-white/10">
                      <svg className="w-8 h-8 text-slate-400 dark:text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" /></svg>
                    </div>
                    <h3 className="text-[16px] font-bold text-slate-800 dark:text-gray-100 mb-2">{safe_t("studio.tabSocial", "Social Media Management")}</h3>
                    <p className="max-w-sm text-[14px] font-medium leading-relaxed text-slate-500 dark:text-gray-300">
                      {socialTaskId
                        ? socialStatus?.state === "FAILURE"
                          ? extractError(socialStatus.error ?? socialStatus.last_error)
                          : socialLoadingText
                        : articleId
                          ? safe_t("studio.socialEmpty", "Extracted social media content will be ready for publication here after final article processing.")
                          : socialMissingText}
                    </p>
                  </div>
                )}
              </div>
            )}

            {activeTab === "schema" && (
              <div className="p-6 lg:p-8 max-w-5xl mx-auto w-full">
                {articleId ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-black/5 pb-4 dark:border-white/10">
                      <div>
                        <h3 className="text-[18px] font-semibold text-slate-900 dark:text-gray-100">{safe_t("studio.tabSchema", "SEO Optimization")}</h3>
                        <p className="mt-1 text-[13px] font-medium text-slate-500 dark:text-gray-300">
                          {schemaLoading ? loadingSchemaText : schemaReadyText}
                        </p>
                      </div>
                      {schemaTitle && (
                        <span className="max-w-xs truncate rounded-lg border border-black/5 bg-white px-3 py-1.5 text-[12px] font-medium text-slate-600 dark:border-white/10 dark:bg-white/[0.06] dark:text-gray-300" dir="auto">
                          {schemaTitle}
                        </span>
                      )}
                    </div>

                    {schemaError && (
                      <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-[13px] font-medium text-red-700 dark:text-red-300" role="alert">
                        {schemaError}
                      </div>
                    )}

                    <div className="grid gap-4 xl:grid-cols-2">
                      <article className="flex min-h-[420px] flex-col rounded-xl border border-black/5 bg-white dark:border-white/10 dark:bg-surface-alt">
                        <div className="flex items-center justify-between gap-3 border-b border-black/5 px-4 py-3 dark:border-white/10">
                          <h4 className="text-[13px] font-semibold text-slate-900 dark:text-gray-100">JSON-LD</h4>
                          <div className="flex items-center gap-2">
                            <Button variant="ghost" size="sm" disabled={!schemaJson} onClick={() => void copyToClipboard(schemaJson)}>
                              {t("common.copy")}
                            </Button>
                            <Button variant="outlined" size="sm" disabled={!schemaJson || !articleId} onClick={() => downloadTextFile(schemaJson, `${articleId}-jsonld.json`, "application/ld+json;charset=utf-8")}>
                              {t("common.download")}
                            </Button>
                          </div>
                        </div>
                        <pre className="min-h-0 flex-1 overflow-auto p-4 text-[12px] leading-5 text-slate-700 dark:text-gray-200" dir="ltr">
                          {schemaLoading && !schemaJson ? loadingSchemaText : schemaJson || safe_t("studio.schemaEmpty", "SEO micro-transactions (FAQ) and structured metadata (JSON-LD) will be injected here for optimal indexing post-generation.")}
                        </pre>
                      </article>

                      <article className="flex min-h-[420px] flex-col rounded-xl border border-black/5 bg-white dark:border-white/10 dark:bg-surface-alt">
                        <div className="flex items-center justify-between gap-3 border-b border-black/5 px-4 py-3 dark:border-white/10">
                          <h4 className="text-[13px] font-semibold text-slate-900 dark:text-gray-100">HTML</h4>
                          <div className="flex items-center gap-2">
                            <Button variant="ghost" size="sm" disabled={!exportHtml} onClick={() => void copyToClipboard(exportHtml)}>
                              {t("common.copy")}
                            </Button>
                            <Button variant="outlined" size="sm" disabled={!exportHtml || !articleId} onClick={() => downloadTextFile(exportHtml, `${articleId}.html`, "text/html;charset=utf-8")}>
                              {t("common.download")}
                            </Button>
                          </div>
                        </div>
                        <pre className="min-h-0 flex-1 overflow-auto p-4 text-[12px] leading-5 text-slate-700 dark:text-gray-200" dir="ltr">
                          {schemaLoading && !exportHtml ? loadingSchemaText : exportHtml || safe_t("studio.schemaEmpty", "SEO micro-transactions (FAQ) and structured metadata (JSON-LD) will be injected here for optimal indexing post-generation.")}
                        </pre>
                      </article>
                    </div>
                  </div>
                ) : (
                  <div className="flex min-h-[300px] flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center dark:border-white/10 dark:bg-surface-alt">
                    <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-slate-50 dark:bg-white/10">
                      <svg className="w-8 h-8 text-slate-400 dark:text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>
                    </div>
                    <h3 className="text-[16px] font-bold text-slate-800 dark:text-gray-100 mb-2">{safe_t("studio.tabSchema", "SEO Optimization")}</h3>
                    <p className="text-[14px] font-medium text-slate-500 dark:text-gray-400 max-w-sm leading-relaxed">{schemaMissingText}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </article>

        {/* ── Dynamic Layout Drawer: Active Tasks (only visible when a task is running) ── */}
        {(taskStatus || batchStatus) && (
          <aside className="animate-slide-in-start sticky top-4 flex w-full max-w-sm flex-col gap-4 self-start rounded-xl border border-black/5 bg-white p-5 dark:border-white/10 dark:bg-surface">
            <h3 className="text-[13px] font-semibold text-slate-400 dark:text-gray-300">{safe_t("studio.taskStatus", "Execution Status")}</h3>

            {taskStatus && (
              <div className="relative space-y-4 overflow-hidden rounded-xl border border-black/5 bg-slate-50 p-4 dark:border-white/10 dark:bg-surface-alt">
                <div className="absolute top-0 start-0 w-1 p-0.5 h-full bg-teal-500" />
                <div className="flex items-center justify-between ps-3">
                  <div className="flex items-center gap-2">
                    <span className={clsx(
                      "w-2 h-2 rounded-full",
                      taskStatus.state === "SUCCESS" ? "bg-emerald-500" :
                        taskStatus.state === "FAILURE" ? "bg-red-500" :
                          "bg-teal-500 animate-pulse"
                    )} />
                    <span className="text-[11px] font-semibold text-slate-600 dark:text-gray-300 uppercase">
                      {taskStatus.state}
                    </span>
                  </div>
                  {!taskStatus.ready && <span className="text-[10px] bg-teal-100 dark:bg-teal-500/15 text-teal-700 dark:text-teal-200 px-2 py-0.5 rounded-full font-semibold uppercase animate-pulse">Live</span>}
                </div>
                <code className="block text-[10px] text-slate-400 dark:text-gray-300 font-mono truncate ps-3" dir="ltr">id: {taskStatus.task_id}</code>
                <p className="text-[13px] text-slate-700 dark:text-gray-300 leading-relaxed font-medium ps-3">{taskStatus.status}</p>
                {taskStatus.error && (
                  <p className="rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-[12px] font-medium leading-5 text-red-700 dark:text-red-300">
                    {taskStatus.error}
                  </p>
                )}
                {taskStatus.manager_error_detail && (
                  <p className="rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-[11px] leading-5 text-amber-800 dark:text-amber-200">
                    {taskStatus.manager_error_detail}
                  </p>
                )}
              </div>
            )}

            {batchStatus && (
              <div className="space-y-4 rounded-xl border border-black/5 bg-slate-50 p-4 dark:border-white/10 dark:bg-surface-alt">
                <div className="flex items-center justify-between">
                  <h4 className="text-[13px] font-bold text-slate-900 dark:text-gray-100">{safe_t("studio.batchProgress", "Pipeline Progress")}</h4>
                  <span className="text-[12px] font-mono bg-white dark:bg-white/10 px-2 py-1 rounded-md border border-slate-200 dark:border-white/10 shadow-sm text-slate-600 dark:text-gray-300 font-bold">
                    {batchStatus.completed} / {batchStatus.total}
                  </span>
                </div>
                <div className="h-1.5 w-full bg-slate-200 dark:bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full bg-slate-900/70 dark:bg-emerald-400/80 transition-all duration-500 ease-out" style={{ width: `${batchStatus.total ? (batchStatus.completed / batchStatus.total) * 100 : 0}%` }} />
                </div>
              </div>
            )}
          </aside>
        )}
      </div>
    </section>
  );
}
