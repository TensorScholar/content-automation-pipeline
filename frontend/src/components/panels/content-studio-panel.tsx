"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import { formatModelDisplayName } from "@/lib/model-display";
import { LlmOptionsResponse, ProjectReadiness, TaskStatusResponse } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import type { MessageKey } from "@/i18n/types";
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

type ReadinessItem = ProjectReadiness["blocking_items"][number];

function isWordPressReadinessItem(item: ReadinessItem): boolean {
  const stableId = item.id.trim().toLowerCase();
  if (/(^|[._:-])wordpress($|[._:-])/.test(stableId)) return true;

  return /\bwordpress\b|وردپرس|ووردبريس/i.test(item.message);
}

function isContentRulesReadinessItem(item: ReadinessItem): boolean {
  const stableId = item.id.trim().toLowerCase();
  if (/(content[._:-]?rules?|rulebook|editorial[._:-]?rules?)/.test(stableId)) return true;

  return /content rulebook|content rules|قوانین محتوا|قاعده.*محتوا|قواعد المحتوى/i.test(item.message);
}

type GenerationBlockerKind = "contentRules" | "worker" | "aiProvider" | "budget" | "projectProfile" | "system";

function generationBlockerKind(items: ReadinessItem[]): GenerationBlockerKind {
  const text = items.map((item) => `${item.id} ${item.label} ${item.message} ${item.remediation ?? ""}`.toLowerCase()).join(" ");
  if (items.some(isContentRulesReadinessItem)) return "contentRules";
  if (/worker|celery|queue|صف پردازش|عامل معالجة/.test(text)) return "worker";
  if (/ai provider|llm|model|api key|ارائه‌دهنده هوش|مدل|مزود الذكاء/.test(text)) return "aiProvider";
  if (/budget|cost limit|بودجه|ميزانية/.test(text)) return "budget";
  if (/project profile|project identity|مشخصات پروژه|ملف المشروع/.test(text)) return "projectProfile";
  return "system";
}

type StudioTab = "generate" | "bulk" | "social" | "schema";
type GenerationLanguage = "fa" | "ar" | "en";

const OUTPUT_LANGUAGE_INSTRUCTION: Record<GenerationLanguage, string> = {
  fa: "Output language must be Persian (Farsi).",
  ar: "Output language must be Arabic.",
  en: "Output language must be English.",
};

const EXECUTION_COPY = {
  en: { success: "Success", failed: "Failed", running: "Running", pending: "Pending", live: "Live", taskStatus: "Execution Status", completed: "Generation completed successfully", processing: "Generation is being processed", queued: "Generation request queued.", failureSummary: "Content generation did not pass the release checks.", releaseGate: "Release gate", wordCount: "Words", wordRange: "Allowed", headings: "Headings", paragraphs: "Paragraphs", findings: "Findings", technicalDetails: "Technical details", wordCountBounds: "Article word counts must be between 800 and 3500." },
  fa: { success: "موفق", failed: "ناموفق", running: "در حال اجرا", pending: "در انتظار", live: "زنده", taskStatus: "وضعیت اجرا", completed: "پردازش با موفقیت کامل شد", processing: "پردازش در حال انجام است", queued: "درخواست تولید در صف قرار گرفت.", failureSummary: "تولید محتوا از بررسی‌های نهایی عبور نکرد.", releaseGate: "دروازه کیفیت", wordCount: "تعداد کلمات", wordRange: "بازه مجاز", headings: "تیترها", paragraphs: "پاراگراف‌ها", findings: "یافته‌ها", technicalDetails: "جزئیات فنی", wordCountBounds: "تعداد کلمات مقاله باید بین ۸۰۰ تا ۳۵۰۰ باشد." },
  ar: { success: "ناجح", failed: "فشل", running: "قيد التشغيل", pending: "قيد الانتظار", live: "مباشر", taskStatus: "حالة التنفيذ", completed: "اكتمل طلب الإنشاء بنجاح", processing: "طلب الإنشاء قيد المعالجة", queued: "تمت إضافة طلب الإنشاء إلى قائمة الانتظار.", failureSummary: "لم يجتز إنشاء المحتوى فحوص الإصدار النهائية.", releaseGate: "بوابة الجودة", wordCount: "الكلمات", wordRange: "النطاق المسموح", headings: "العناوين", paragraphs: "الفقرات", findings: "النتائج", technicalDetails: "التفاصيل التقنية", wordCountBounds: "يجب أن يكون عدد كلمات المقال بين 800 و3500." },
} as const;

const ARTICLE_WORD_COUNT_MIN = 800;
const ARTICLE_WORD_COUNT_MAX = 3500;

function localizeExecutionState(state: string, locale: keyof typeof EXECUTION_COPY) {
  const normalized = state.trim().toUpperCase();
  if (normalized === "SUCCESS") return EXECUTION_COPY[locale].success;
  if (normalized === "FAILURE" || normalized === "FAILED") return EXECUTION_COPY[locale].failed;
  if (normalized === "PENDING") return EXECUTION_COPY[locale].pending;
  if (normalized === "QUEUED") return EXECUTION_COPY[locale].queued;
  return EXECUTION_COPY[locale].running;
}

function localizeExecutionMessage(message: string, locale: keyof typeof EXECUTION_COPY) {
  const normalized = message.trim().toLowerCase().replace(/[.!]+$/, "");
  if (normalized === "task completed successfully") return EXECUTION_COPY[locale].completed;
  if (normalized === "task is being processed") return EXECUTION_COPY[locale].processing;
  if (normalized === "queued" || normalized === "task is queued") return EXECUTION_COPY[locale].queued;
  if (normalized === "task failed") return EXECUTION_COPY[locale].failureSummary;
  return message;
}

function formatExecutionNumber(value: number | undefined, locale: keyof typeof EXECUTION_COPY) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  return new Intl.NumberFormat(localeName).format(value);
}

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
    blocked: "Generation is temporarily unavailable. Review project readiness for the exact cause.",
    contentRulesBlocked: "Configure the project content rules before generating an article.",
    workerBlocked: "Generation needs an active processing worker.",
    aiBlocked: "Generation needs an available AI provider and model.",
    budgetBlocked: "Generation is paused because the configured AI budget is unavailable or exhausted.",
    profileBlocked: "Complete the required project profile before generating an article.",
    contentRulesHint: "Open Content Rules in the project settings.",
    workerHint: "Open Monitoring and check the processing queue and worker status.",
    workerStatus: "Processing queue unavailable",
    additionalSetupItems: "Additional generation setup items also require attention.",
    aiHint: "Check AI provider credentials and model availability.",
    llmTimeout: "The AI model test connection did not finish in time.",
    budgetHint: "Review the project or global AI budget.",
    profileHint: "Complete the required project fields.",
    systemHint: "Open project readiness for details.",
    publishingBlocked: "WordPress is not connected. You can still generate content, but publishing to WordPress is unavailable until the connection is configured.",
    checking: "Checking project readiness...",
    unavailable: "Project readiness could not be verified. Generation is paused until a successful recheck.",
    readyStatus: "Ready",
    warningStatus: "Review warnings",
    blockedStatus: "Generation blocked",
    setupStatus: "Generation setup needed",
    publishingStatus: "Publishing setup needed",
    checkingStatus: "Checking",
    unavailableStatus: "Readiness unavailable",
    unavailableHint: "Check the API connection, then recheck project readiness.",
    refresh: "Recheck",
  },
  fa: {
    ready: "پروژه برای تولید محتوا آماده است.",
    warning: "پروژه هشدار آماده‌سازی دارد، اما تولید محتوا فعال است.",
    blocked: "تولید محتوا موقتاً در دسترس نیست. علت دقیق را در بررسی آماده‌سازی پروژه ببینید.",
    contentRulesBlocked: "پیش از تولید مقاله، قوانین محتوای پروژه را تکمیل کنید.",
    workerBlocked: "برای تولید محتوا، حداقل یک پردازشگر فعال لازم است.",
    aiBlocked: "برای تولید محتوا، ارائه‌دهنده و مدل هوش مصنوعی باید در دسترس باشند.",
    budgetBlocked: "تولید محتوا به‌دلیل نبودن یا پایان بودجه هوش مصنوعی متوقف شده است.",
    profileBlocked: "پیش از تولید مقاله، مشخصات الزامی پروژه را تکمیل کنید.",
    contentRulesHint: "قوانین محتوا را در تنظیمات پروژه باز کنید.",
    workerHint: "به مانیتورینگ بروید و وضعیت صف پردازش و پردازشگرها را بررسی کنید.",
    workerStatus: "صف پردازش غیرفعال است",
    additionalSetupItems: "موارد دیگری از تنظیمات تولید نیز نیازمند بررسی است.",
    aiHint: "اعتبارنامه ارائه‌دهنده و دسترسی مدل را بررسی کنید.",
    llmTimeout: "اتصال آزمایشی به مدل هوش مصنوعی در زمان مجاز کامل نشد.",
    budgetHint: "بودجه هوش مصنوعی پروژه یا سامانه را بررسی کنید.",
    profileHint: "فیلدهای الزامی پروژه را تکمیل کنید.",
    systemHint: "برای جزئیات، بررسی آماده‌سازی پروژه را باز کنید.",
    publishingBlocked: "وردپرس متصل نیست. همچنان می‌توانید محتوا تولید کنید، اما انتشار در وردپرس تا زمان تکمیل اتصال در دسترس نیست.",
    checking: "در حال بررسی آمادگی پروژه...",
    unavailable: "آمادگی پروژه تأیید نشد. تولید محتوا تا بررسی موفق دوباره متوقف است.",
    readyStatus: "آماده",
    warningStatus: "نیازمند بررسی",
    blockedStatus: "تولید مسدود است",
    setupStatus: "تنظیم تولید لازم است",
    publishingStatus: "تنظیم انتشار لازم است",
    checkingStatus: "در حال بررسی",
    unavailableStatus: "آمادگی در دسترس نیست",
    unavailableHint: "اتصال API را بررسی کنید و سپس آمادگی پروژه را دوباره بسنجید.",
    refresh: "بررسی دوباره",
  },
  ar: {
    ready: "المشروع جاهز لإنشاء المحتوى.",
    warning: "توجد تحذيرات جاهزية، لكن الإنشاء متاح.",
    blocked: "إنشاء المحتوى غير متاح مؤقتاً. راجع فحص جاهزية المشروع لمعرفة السبب الدقيق.",
    contentRulesBlocked: "أكمل إعداد قواعد محتوى المشروع قبل إنشاء المقال.",
    workerBlocked: "يتطلب إنشاء المحتوى عامل معالجة نشطاً واحداً على الأقل.",
    aiBlocked: "يتطلب إنشاء المحتوى مزود ذكاء اصطناعي ونموذجاً متاحين.",
    budgetBlocked: "تم إيقاف إنشاء المحتوى لأن ميزانية الذكاء الاصطناعي غير متاحة أو مستنفدة.",
    profileBlocked: "أكمل حقول ملف المشروع المطلوبة قبل إنشاء المقال.",
    contentRulesHint: "افتح قواعد المحتوى من إعدادات المشروع.",
    workerHint: "انتقل إلى المراقبة وتحقق من حالة قائمة المعالجة والعاملين.",
    workerStatus: "قائمة المعالجة غير متاحة",
    additionalSetupItems: "توجد عناصر إعداد إضافية لإنشاء المحتوى تحتاج إلى المراجعة.",
    aiHint: "تحقق من بيانات اعتماد المزود وتوفر النموذج.",
    llmTimeout: "لم يكتمل اختبار الاتصال بنموذج الذكاء الاصطناعي ضمن المهلة.",
    budgetHint: "راجع ميزانية الذكاء الاصطناعي للمشروع أو النظام.",
    profileHint: "أكمل حقول المشروع المطلوبة.",
    systemHint: "افتح فحص جاهزية المشروع للتفاصيل.",
    publishingBlocked: "ووردبريس غير متصل. لا يزال بإمكانك إنشاء المحتوى، لكن النشر إلى ووردبريس غير متاح حتى يتم إعداد الاتصال.",
    checking: "جارٍ فحص جاهزية المشروع...",
    unavailable: "تعذر التحقق من جاهزية المشروع. تم إيقاف الإنشاء حتى ينجح الفحص مجدداً.",
    readyStatus: "جاهز",
    warningStatus: "يحتاج إلى مراجعة",
    blockedStatus: "إنشاء المحتوى محظور",
    setupStatus: "يلزم إعداد الإنشاء",
    publishingStatus: "يلزم إعداد النشر",
    checkingStatus: "جارٍ الفحص",
    unavailableStatus: "الجاهزية غير متاحة",
    unavailableHint: "تحقق من اتصال API ثم أعد فحص جاهزية المشروع.",
    refresh: "إعادة الفحص",
  },
};

function localizeTechnicalMessage(message: string, locale: keyof typeof READINESS_COPY) {
  const normalized = message.trim().toLowerCase();
  if (normalized.includes("llm ping timed out") || normalized.includes("ai provider ping timed out")) {
    return READINESS_COPY[locale].llmTimeout;
  }
  return localizeExecutionMessage(message, locale);
}

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
  error?: "bounds" | "range";
  payload: { word_count_range?: string; target_word_count?: number };
} {
  const min = minRaw.trim() ? Number(minRaw) : undefined;
  const max = maxRaw.trim() ? Number(maxRaw) : undefined;

  if (
    min === undefined || max === undefined ||
    !Number.isInteger(min) || !Number.isInteger(max) ||
    min < ARTICLE_WORD_COUNT_MIN || max > ARTICLE_WORD_COUNT_MAX
  ) {
    return { error: "bounds", payload: {} };
  }
  if (min !== undefined && max !== undefined && min > max) {
    return { error: "range", payload: {} };
  }
  return { payload: { word_count_range: `${min}-${max}` } };
}

function sanitizeArticleWordCountInput(value: string): string {
  return value.replace(/[^\d]/g, "");
}

function clampArticleWordCountInput(value: string): string {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return "";
  return String(Math.min(ARTICLE_WORD_COUNT_MAX, Math.max(ARTICLE_WORD_COUNT_MIN, Math.round(numericValue))));
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

/* Content creation workspace. Generation logic and outputs are preserved; advanced controls use progressive disclosure. */

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

  const [wordCountMin, setWordCountMin] = useState(String(ARTICLE_WORD_COUNT_MIN));
  const [wordCountMax, setWordCountMax] = useState("1000");
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
  const [readinessError, setReadinessError] = useState(false);
  const readinessRequestRef = useRef<AbortController | null>(null);
  const [llmOptions, setLlmOptions] = useState<LlmOptionsResponse | null>(null);
  const [llmOptionsLoading, setLlmOptionsLoading] = useState(false);
  const [modelOverride, setModelOverride] = useState("");

  // Translation keys are statically typed; provider-level English fallback is canonical.
  const tr = (key: MessageKey, _fallback?: string) => t(key);
  const customFallback = locale === "fa" ? "سفارشی" : locale === "ar" ? "مخصص" : "Custom";
  const customValuePlaceholder = locale === "fa" ? "مقدار دلخواه را وارد کنید" : locale === "ar" ? "أدخل قيمة مخصصة" : "Enter custom value";
  const contextPlaceholder = locale === "fa" ? "- روی مورد X تمرکز شود..." : locale === "ar" ? "- ركّز على العنصر X..." : "- Focus on topic X...";
  const sourcePlaceholder = locale === "fa" ? "https://example.com/page-1\nhttps://example.com/page-2" : locale === "ar" ? "https://example.com/page-1\nhttps://example.com/page-2" : "https://example.com/page-1\nhttps://example.com/page-2";
  const socialLoadingText = locale === "fa" ? "در حال آماده‌سازی پیش‌نویس‌های شبکه اجتماعی..." : locale === "ar" ? "جارٍ إعداد مسودات الشبكات الاجتماعية..." : "Preparing social drafts...";
  const socialReadyText = locale === "fa" ? "پیش‌نویس‌های آماده برای کپی؛ بدون اتصال حساب، زمان‌بندی، انتشار یا تحلیل." : locale === "ar" ? "مسودات جاهزة للنسخ؛ بدون ربط حساب أو جدولة أو نشر أو تحليلات." : "Drafts ready to copy; no account connection, scheduling, publishing, or analytics.";
  const schemaReadyText = locale === "fa" ? "داده ساخت‌یافته و خروجی HTML آماده است؛ این بخش ممیزی یا ویرایشگر کامل سئو نیست." : locale === "ar" ? "البيانات المنظمة وتصدير HTML جاهزان؛ هذه ليست أداة تدقيق أو تحرير كاملة لتحسين البحث." : "Structured data and HTML export are ready; this is not a full SEO audit or editor.";
  const socialTabLabel = locale === "fa" ? "پیش‌نویس‌های شبکه‌های اجتماعی" : locale === "ar" ? "مسودات الشبكات الاجتماعية" : "Social Drafts";
  const schemaTabLabel = locale === "fa" ? "خروجی و ابزارهای سئو" : locale === "ar" ? "أدوات وتصدير تحسين البحث" : "SEO Export Tools";
  const socialEmptyText = locale === "fa"
    ? "پیش‌نویس‌ها فقط پس از تولید موفق مقاله در همین نشست استودیو نمایش داده می‌شوند. اتصال حساب، زمان‌بندی، انتشار و تحلیل در این بخش وجود ندارد."
    : locale === "ar"
      ? "تظهر المسودات فقط بعد إنشاء مقالة ناجحة في جلسة الاستوديو الحالية. لا يتوفر ربط حساب أو جدولة أو نشر أو تحليلات هنا."
      : "Drafts appear only after a successful article generated in this Studio session. This view has no account connection, scheduling, publishing, or analytics.";
  const schemaEmptyText = locale === "fa"
    ? "این ابزارها فقط پس از تولید موفق مقاله در همین نشست استودیو، داده ساخت‌یافته و خروجی HTML را نشان می‌دهند. این بخش ممیزی یا ویرایشگر کامل سئو نیست."
    : locale === "ar"
      ? "تعرض هذه الأدوات البيانات المنظمة وتصدير HTML فقط بعد إنشاء مقالة ناجحة في جلسة الاستوديو الحالية. ليست أداة تدقيق أو تحرير كاملة لتحسين البحث."
      : "These tools show structured data and HTML export only after a successful article generated in this Studio session. This is not a full SEO audit or editor.";
  const loadingSchemaText = locale === "fa" ? "در حال آماده‌سازی JSON-LD و خروجی HTML..." : locale === "ar" ? "جاري إعداد JSON-LD و HTML..." : "Preparing JSON-LD and HTML export...";
  const modelCopy = MODEL_COPY[locale];

  const translateOptions = (opts: typeof TONE_OPTIONS) => opts.map(o => ({
    value: o.value,
    label: locale === "fa" ? o.fa : locale === "ar" ? o.ar : o.en
  })).concat([{ value: "__custom__", label: tr("common.custom", customFallback) }]);

  const labelFor = (opts: typeof TONE_OPTIONS, val: string) => {
    const found = opts.find((o) => o.value === val);
    if (!found) return val;
    return locale === "fa" ? found.fa : locale === "ar" ? found.ar : found.en;
  };

  const articleId = useMemo(() => taskStatus?.result?.article_id ?? null, [taskStatus]);
  const executionState = taskStatus?.state.trim().toUpperCase();
  const executionDiagnostics = taskStatus?.quality_diagnostics;
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

  const loadReadiness = useCallback(async () => {
    readinessRequestRef.current?.abort();

    if (!selectedProjectId) {
      readinessRequestRef.current = null;
      setReadiness(null);
      setReadinessLoading(false);
      setReadinessError(false);
      return;
    }

    const controller = new AbortController();
    readinessRequestRef.current = controller;
    setReadinessLoading(true);
    setReadinessError(false);
    try {
      const payload = await apiRequest<ProjectReadiness>(`/projects/${selectedProjectId}/readiness`, {
        token,
        signal: controller.signal,
        timeoutMs: 10000,
      });
      if (controller.signal.aborted) return;
      setReadiness(payload);
      setReadinessError(false);
    } catch {
      if (controller.signal.aborted) return;
      setReadiness(null);
      setReadinessError(true);
    } finally {
      if (controller.signal.aborted) return;
      if (readinessRequestRef.current === controller) readinessRequestRef.current = null;
      setReadinessLoading(false);
    }
  }, [selectedProjectId, token]);

  // Reset state on project change
  useEffect(() => {
    setTaskStatus(null); setSocialStatus(null); setJsonld(null); setExportHtml("");
    setSchemaError(null); setSchemaLoading(false); setSchemaTitle("");
    setBatchId(null); setBatchStatus(null);
  }, [selectedProjectId]);

  useEffect(() => {
    void loadReadiness();
    return () => readinessRequestRef.current?.abort();
  }, [loadReadiness]);

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
    if (!selectedProjectId) return showToast("error", tr("studio.selectProjectFirst", "Select a project first"));
    if (generationBlocked) return showToast("error", readinessMessage);
    setSubmitting(true);
    setTaskStatus(null);
    setSocialStatus(null);
    setJsonld(null);
    setExportHtml("");
    setSchemaError(null);
    setSchemaTitle("");

    // Stable machine values for structured top-level request fields.
    // Uses the option `value` key (locale-independent), not the display label.
    const resolvedTone = tone === "__custom__" ? customTone.trim() : tone;
    const resolvedStructure = structure === "__custom__" ? customStructure.trim() : structure;
    const resolvedAudience = audience === "__custom__" ? customAudience.trim() : audience;
    const wordCount = buildWordCountPayload(wordCountMin, wordCountMax);

    if (wordCount.error) {
      setSubmitting(false);
      return showToast(
        "error",
        wordCount.error === "bounds"
          ? EXECUTION_COPY[locale].wordCountBounds
          : tr("studio.wordCountInvalidRange", "Minimum word count cannot exceed maximum word count."),
      );
    }

    // additional_instructions carries only free-form context and point-of-view.
    // Tone, structure, and audience are transmitted as structured top-level fields.
    const instructions = [
      OUTPUT_LANGUAGE_INSTRUCTION[language],
      extraInstructions.trim(),
      competitorUrl.trim() ? `Competitor URL: ${competitorUrl.trim()}` : "",
      sourceUrls.trim() ? `Source URLs:\n${sourceUrls.trim()}` : "",
      `Point of View: ${pov === "__custom__" ? customPov.trim() : pov}`,
      "After article generation, preserve social repurposing outputs.",
      "Ensure schema-friendly structure for FAQ/HowTo rich snippets."
    ].filter(Boolean).join("\n");

    try {
      const payload = await apiRequest<{ task_id: string; status: string }>("/content/generate/async", {
        method: "POST", token,
        body: {
          project_id: selectedProjectId, topic: topic.trim(), priority: "high", primary_keyword: keyword.trim(),
          additional_instructions: instructions, language, temperature,
          tone: resolvedTone || undefined,
          target_audience: resolvedAudience || undefined,
          content_structure: resolvedStructure || undefined,
          model_override: modelOverride || undefined,
          ...wordCount.payload,
          seo_settings: { auto_schema: true, competitor_takedown: competitorUrl.trim().length > 0 }
        }
      });
      setTaskStatus({ task_id: payload.task_id, state: "PENDING", ready: false, status: payload.status });
      showToast("success", tr("studio.taskQueued", EXECUTION_COPY[locale].queued));
    } catch (err) { showToast("error", extractError(err)); }
    finally { setSubmitting(false); }
  };

  const onSubmitBatch = async (e: FormEvent) => {
    e.preventDefault();
    if (!selectedProjectId) return showToast("error", tr("studio.selectProjectFirst", "Select a project first"));
    if (generationBlocked) return showToast("error", readinessMessage);
    const topics = bulkTopics.split("\n").map(l => l.trim()).filter(Boolean);
    if (topics.length === 0 || topics.length > 20) return showToast("error", tr("studio.maxTopics", "Max 20 topics allowed"));

    setBulkSubmitting(true);
    setBatchStatus(null);
    try {
      const bulkInstructions = [
        bulkKeyword.trim() ? `Primary keyword for all topics: ${bulkKeyword.trim()}` : "",
        OUTPUT_LANGUAGE_INSTRUCTION[bulkLanguage],
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
      showToast("success", tr("toast.batchSubmitted", "Batch request submitted"));
    } catch (err) { showToast("error", extractError(err)); }
    finally { setBulkSubmitting(false); }
  };

  const tabEntries: { key: StudioTab; label: string }[] = [
    { key: "generate", label: tr("studio.tabGenerate", "Single Article") },
    { key: "bulk", label: tr("studio.tabBatch", "Bulk Generation") },
    { key: "social", label: socialTabLabel },
    { key: "schema", label: schemaTabLabel },
  ];

  const creationTabs = tabEntries.filter((entry) => entry.key === "generate" || entry.key === "bulk");
  const outputTabs = tabEntries.filter((entry) => entry.key === "social" || entry.key === "schema");

  const InputClass = "smx-input";
  const LabelClass = "mb-1.5 max-w-max text-sm font-medium leading-5 text-ink-muted";
  const HeaderClass = "mb-4 text-base font-semibold tracking-[-0.01em] text-ink";
   const wordpressBlockingItems = readiness?.blocking_items.filter(isWordPressReadinessItem) ?? [];
   const generationBlockingItems = readiness?.blocking_items.filter((item) => !isWordPressReadinessItem(item)) ?? [];
   const blockerKind = generationBlockerKind(generationBlockingItems);
   const wordpressOnlyBlocking = !!readiness
     && !readiness.can_generate
     && readiness.blocking_items.length > 0
     && wordpressBlockingItems.length === readiness.blocking_items.length;
   const readinessState = readinessLoading
     ? "checking"
     : readinessError
       ? "unavailable"
     : readiness && !readiness.can_generate && !wordpressOnlyBlocking
       ? "blocked"
       : (readiness && !readiness.can_publish) || wordpressOnlyBlocking
         ? "warning"
         : readiness && (readiness.status === "warning" || readiness.status === "blocked")
           ? "warning"
           : "ready";
   const generationBlocked = readinessLoading
     || readinessError
     || (!!readiness && !readiness.can_generate && !wordpressOnlyBlocking);
   const publishingBlocked = (!!readiness && readiness.can_generate && !readiness.can_publish) || wordpressOnlyBlocking;
   const wordpressPublishingBlocked = publishingBlocked && wordpressBlockingItems.length > 0;
   const readinessMessage =
     readinessState === "checking"
       ? READINESS_COPY[locale].checking
       : readinessState === "unavailable"
         ? READINESS_COPY[locale].unavailable
       : readinessState === "blocked"
         ? blockerKind === "contentRules"
           ? READINESS_COPY[locale].contentRulesBlocked
           : blockerKind === "worker"
             ? READINESS_COPY[locale].workerBlocked
             : blockerKind === "aiProvider"
               ? READINESS_COPY[locale].aiBlocked
               : blockerKind === "budget"
                 ? READINESS_COPY[locale].budgetBlocked
                 : blockerKind === "projectProfile"
                   ? READINESS_COPY[locale].profileBlocked
                   : READINESS_COPY[locale].blocked
         : wordpressPublishingBlocked
           ? READINESS_COPY[locale].publishingBlocked
           : publishingBlocked
             ? READINESS_COPY[locale].warning
           : readinessState === "warning"
           ? READINESS_COPY[locale].warning
           : READINESS_COPY[locale].ready;
   const readinessStatusLabel = readinessState === "checking"
     ? READINESS_COPY[locale].checkingStatus
     : readinessState === "unavailable"
       ? READINESS_COPY[locale].unavailableStatus
     : readinessState === "blocked"
       ? blockerKind === "worker"
         ? READINESS_COPY[locale].workerStatus
         : blockerKind === "system"
           ? READINESS_COPY[locale].blockedStatus
           : READINESS_COPY[locale].setupStatus
       : wordpressPublishingBlocked
         ? READINESS_COPY[locale].publishingStatus
         : readinessState === "warning"
           ? READINESS_COPY[locale].warningStatus
           : READINESS_COPY[locale].readyStatus;
   const readinessHint = readinessState === "unavailable"
     ? READINESS_COPY[locale].unavailableHint
     : readinessState === "blocked"
     ? blockerKind === "contentRules"
       ? READINESS_COPY[locale].contentRulesHint
       : blockerKind === "worker"
         ? READINESS_COPY[locale].workerHint
         : blockerKind === "aiProvider"
           ? READINESS_COPY[locale].aiHint
           : blockerKind === "budget"
             ? READINESS_COPY[locale].budgetHint
             : blockerKind === "projectProfile"
               ? READINESS_COPY[locale].profileHint
               : READINESS_COPY[locale].systemHint
     : null;
  const selectableModels = llmOptions?.selectable_models ?? [];
  const selectedModelOption = selectableModels.find((option) => option.model === modelOverride);
  const llmUnavailable = !llmOptionsLoading && selectableModels.length === 0;
  const llmUserMessage = llmOptions?.user_message && llmOptions.user_message !== "AI generation is available."
    ? llmOptions.user_message
    : null;
  const llmWarning = llmOptions?.warnings[0] ?? llmUserMessage ?? (llmUnavailable ? modelCopy.unavailable : null);
  const localizedLlmWarning = llmWarning ? localizeTechnicalMessage(llmWarning, locale) : null;
  const selectedActiveProviderUnavailable =
    !!llmUserMessage && selectedModelOption?.provider === llmOptions?.active_provider;
  const selectedModelUnavailable = llmUnavailable || selectedActiveProviderUnavailable;

  return (
    <section className="smx-page !max-w-none relative flex h-full min-h-0 min-w-0 justify-center overflow-hidden !pb-0">

      {/* ── Dynamic Layout Wrapper ── */}
      <div className={clsx(
        "grid h-full min-h-0 min-w-0 w-full gap-0",
        taskStatus || batchStatus ? "grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px]" : "grid-cols-1"
      )}>

        {/* Creation canvas */}
        <article className="relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-surface">

          {/* Creation modes are primary; generated outputs are a distinct workspace group. */}
          <div className="z-10 flex min-w-0 shrink-0 flex-wrap items-center justify-between gap-3 border-b border-line bg-transparent px-4 py-2.5 lg:px-5">
            <div className="flex min-w-0 items-center gap-1" aria-label={tr("studio.coreIdentity", "Creation")}>
              {creationTabs.map((entry) => (
                <button
                  type="button"
                  key={entry.key}
                  onClick={() => setActiveTab(entry.key)}
                  aria-pressed={activeTab === entry.key}
                  className={clsx(
                    "relative min-h-8 rounded-md px-3 text-sm font-medium transition-colors duration-fast",
                    activeTab === entry.key
                      ? "bg-brand/[0.075] text-ink"
                      : "text-ink-muted hover:bg-ink/[0.035] hover:text-ink",
                  )}
                >
                  {entry.label}
                </button>
              ))}
            </div>

            <div className="flex min-w-0 items-center gap-1 border-s border-line ps-2" aria-label={schemaTabLabel}>
              {outputTabs.map((entry) => (
                <button
                  type="button"
                  key={entry.key}
                  onClick={() => setActiveTab(entry.key)}
                  aria-pressed={activeTab === entry.key}
                  className={clsx(
                    "inline-flex min-h-8 items-center gap-2 rounded-md px-2.5 text-xs font-medium transition-colors duration-fast",
                    activeTab === entry.key
                      ? "bg-ink/[0.055] text-ink"
                      : "text-ink-muted hover:bg-ink/[0.035] hover:text-ink",
                  )}
                >
                  <span className={clsx("h-1.5 w-1.5 rounded-full", articleId ? "bg-success" : "bg-ink-muted/40")} aria-hidden />
                  {entry.label}
                </button>
              ))}
            </div>
          </div>

          <div className="min-h-0 w-full flex-1 overflow-x-hidden overflow-y-auto">
            {selectedProjectId && (activeTab === "generate" || activeTab === "bulk") && (
              <div className="px-4 pt-4 lg:px-5">
                <div className={clsx(
                  "mx-auto flex w-full max-w-5xl flex-col gap-3 border-s-2 px-4 py-2.5 text-sm sm:flex-row sm:items-start sm:justify-between",
                  readinessState === "blocked" || readinessState === "unavailable"
                    ? "border-s-error text-error"
                    : readinessState === "warning"
                      ? "border-s-warning text-warning"
                      : "border-s-success text-success"
                )}>
                  <div className="flex min-w-0 w-full items-start gap-3 sm:w-auto sm:flex-1">
                    <span className={clsx(
                      "mt-1 h-2 w-2 shrink-0 rounded-full",
                      readinessState === "checking" ? "animate-pulse bg-ink-muted/35" :
                        readinessState === "blocked" || readinessState === "unavailable" ? "bg-danger" :
                          readinessState === "warning" ? "bg-warning" : "bg-success"
                    )} />
                    <div className="min-w-0">
                      <p className="font-semibold leading-5">{readinessMessage}</p>
                      {readinessHint && <p className="mt-1 text-xs leading-5 opacity-80">{readinessHint}</p>}
                      {generationBlocked && generationBlockingItems.length > 1 && (
                        <p className="mt-1 text-xs leading-5 opacity-70">{READINESS_COPY[locale].additionalSetupItems}</p>
                      )}
                    </div>
                  </div>
                  <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto sm:shrink-0 sm:justify-end">
                    {(readiness || readinessError) && (
                      <span className="text-xs font-medium text-ink-muted">
                        {readinessStatusLabel}
                      </span>
                    )}
                    <button
                      type="button"
                      disabled={readinessLoading}
                      onClick={() => void loadReadiness()}
                      className="rounded-md border border-line bg-surface px-2.5 py-1 text-xs font-semibold text-ink-secondary transition-colors hover:bg-ink/[0.035] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {READINESS_COPY[locale].refresh}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "generate" && (
              <form onSubmit={onSubmitGenerate} className="flex min-h-full min-w-0 flex-col p-4 lg:p-5">

                <div className="mx-auto flex min-w-0 w-full max-w-5xl flex-col gap-4">
                  {/* Generation fields */}
                  <div className="grid min-w-0 grid-cols-12 gap-4">

                    {/* Core fields */}
                    <div className="col-span-12 flex flex-col gap-4 border-t border-line pt-5 lg:col-span-6">
                      <h2 className={HeaderClass}>{tr("studio.coreIdentity", "Core Identity")}</h2>

                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{tr("studio.articleTopic", "Article Topic")} <span className="text-danger">*</span></label>
                        <input required aria-label={tr("studio.articleTopic", "Article Topic")} className={InputClass} value={topic} onChange={e => setTopic(e.target.value)} />
                      </div>

                      <div className="flex flex-col md:flex-row gap-4">
                        <div className="flex-1 flex flex-col gap-0">
                          <label className={LabelClass}>{tr("studio.primaryKeyword", "Primary Keyword")} <span className="text-danger">*</span></label>
                          <input required aria-label={tr("studio.primaryKeyword", "Primary Keyword")} className={InputClass} value={keyword} onChange={e => setKeyword(e.target.value)} />
                        </div>
                        <div className="flex-1 flex flex-col gap-0">
                          <label className={LabelClass}>{tr("studio.language", "Language")}</label>
                          <select aria-label={tr("studio.language", "Language")} className={InputClass} value={language} onChange={e => setLanguage(e.target.value as GenerationLanguage)}>
                            <option value="fa">{tr("lang.fa", "فارسی")}</option>
                            <option value="ar">{tr("lang.ar", "العربية")}</option>
                            <option value="en">{tr("lang.en", "English")}</option>
                          </select>
                        </div>
                      </div>
                    </div>

                    {/* Writing preferences */}
                    <details className="group col-span-12 border-t border-line pt-4 [&>*:not(summary)]:mt-4 lg:col-span-6">
                      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 py-1 text-base font-semibold text-ink">
                        <span>{tr("studio.tonalDna", "Writing preferences")}</span>
                        <svg className="h-4 w-4 shrink-0 text-ink-tertiary transition-transform duration-fast group-open:rotate-180" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="m6 8 4 4 4-4" strokeLinecap="round" strokeLinejoin="round" /></svg>
                      </summary>

                      <div className="flex flex-col gap-4">
                        <div className="flex flex-col md:flex-row gap-4">
                          <div className="flex-1 flex flex-col gap-0 relative">
                            <label className={LabelClass}>{tr("studio.tone", "Brand Tone")}</label>
                            <select aria-label={tr("studio.tone", "Brand Tone")} className={InputClass} value={tone} onChange={e => setTone(e.target.value)}>
                              {translateOptions(TONE_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                            {tone === "__custom__" && (
                              <div className="mt-3 animate-fade-in">
                                <input autoFocus aria-label={`${tr("studio.tone", "Brand Tone")} · ${customValuePlaceholder}`} placeholder={customValuePlaceholder} className={InputClass} value={customTone} onChange={e => setCustomTone(e.target.value)} />
                              </div>
                            )}
                          </div>

                          <div className="flex-1 flex flex-col gap-0 relative">
                            <label className={LabelClass}>{tr("studio.audience", "Target Audience")}</label>
                            <select aria-label={tr("studio.audience", "Target Audience")} className={InputClass} value={audience} onChange={e => setAudience(e.target.value)}>
                              {translateOptions(AUDIENCE_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                            {audience === "__custom__" && (
                              <div className="mt-3 animate-fade-in">
                                <input autoFocus aria-label={`${tr("studio.audience", "Target Audience")} · ${customValuePlaceholder}`} placeholder={customValuePlaceholder} className={InputClass} value={customAudience} onChange={e => setCustomAudience(e.target.value)} />
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-col gap-4">
                        <div className="flex flex-col md:flex-row gap-4">
                          <div className="flex-1 flex flex-col gap-0 relative">
                            <label className={LabelClass}>{tr("studio.structure", "Structure")}</label>
                            <select aria-label={tr("studio.structure", "Structure")} className={InputClass} value={structure} onChange={e => setStructure(e.target.value)}>
                              {translateOptions(STRUCTURE_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                            {structure === "__custom__" && (
                              <div className="mt-3 animate-fade-in">
                                <input autoFocus aria-label={`${tr("studio.structure", "Structure")} · ${customValuePlaceholder}`} placeholder={customValuePlaceholder} className={InputClass} value={customStructure} onChange={e => setCustomStructure(e.target.value)} />
                              </div>
                            )}
                          </div>

                          <div className="flex-1 flex flex-col gap-0 relative">
                            <label className={LabelClass}>{tr("studio.pointOfView", "Point of View")}</label>
                            <select aria-label={tr("studio.pointOfView", "Point of View")} className={InputClass} value={pov} onChange={e => setPov(e.target.value)}>
                              {translateOptions(POV_OPTIONS).map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                            </select>
                            {pov === "__custom__" && (
                              <div className="mt-3 animate-fade-in">
                                <input autoFocus aria-label={`${tr("studio.pointOfView", "Point of View")} · ${customValuePlaceholder}`} placeholder={customValuePlaceholder} className={InputClass} value={customPov} onChange={e => setCustomPov(e.target.value)} />
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </details>

                    {/* Sources and context */}
                    <details className="group col-span-12 border-t border-line pt-4 [&>*:not(summary)]:mt-4">
                      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 py-1 text-base font-semibold text-ink">
                        <span>{tr("studio.context", "Sources & context")}</span>
                        <svg className="h-4 w-4 shrink-0 text-ink-tertiary transition-transform duration-fast group-open:rotate-180" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="m6 8 4 4 4-4" strokeLinecap="round" strokeLinejoin="round" /></svg>
                      </summary>
                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{tr("studio.reference", "Reference URL (Analysis)")}</label>
                        <input aria-label={tr("studio.reference", "Reference URL (Analysis)")} placeholder="https://" className={clsx(InputClass, "font-mono text-sm")} value={competitorUrl} onChange={e => setCompetitorUrl(e.target.value)} dir="ltr" />
                      </div>
                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{tr("studio.extraInstructions", "Extra Instructions")}</label>
                        <textarea aria-label={tr("studio.extraInstructions", "Extra Instructions")} rows={2} placeholder={contextPlaceholder} className={clsx(InputClass, "h-[74px] resize-none")} value={extraInstructions} onChange={e => setExtraInstructions(e.target.value)} />
                      </div>
                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{tr("studio.sourceUrls", "Source URLs")}</label>
                        <textarea aria-label={tr("studio.sourceUrls", "Source URLs")} rows={2} placeholder={sourcePlaceholder} className={clsx(InputClass, "h-[74px] resize-none font-mono text-sm")} value={sourceUrls} onChange={e => setSourceUrls(e.target.value)} dir="ltr" />
                      </div>
                    </details>

                    {/* Model and advanced parameters */}
                    <details className="group col-span-12 border-t border-line pt-4 [&>*:not(summary)]:mt-4">
                      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 py-1 text-base font-semibold text-ink">
                        <span>{tr("studio.parametersLimits", "Model & advanced")}</span>
                        <svg className="h-4 w-4 shrink-0 text-ink-tertiary transition-transform duration-fast group-open:rotate-180" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="m6 8 4 4 4-4" strokeLinecap="round" strokeLinejoin="round" /></svg>
                      </summary>

                      <div className="flex flex-col gap-8">
                        {/* Sliders Area */}
                        <div className="flex flex-col gap-6">
                          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(220px,auto)]">
                            <div className="flex flex-col gap-0">
                              <label className={LabelClass}>{modelCopy.label}</label>
                              <select
                                aria-label={modelCopy.label}
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
                                      {formatModelDisplayName(option.model)}
                                    </option>
                                  ))
                                ) : (
                                  <option value="">{modelCopy.unavailable}</option>
                                )}
                              </select>
                            </div>
                            <div className="flex min-h-[36px] items-center border-s border-line px-3 py-2 text-xs font-medium text-ink-muted">
                              {selectedModelOption
                                ? `${selectedModelOption.recommended ? modelCopy.recommended : modelCopy.active} · ${formatModelDisplayName(selectedModelOption.model)}`
                                : llmOptionsLoading
                                  ? modelCopy.loading
                                  : modelCopy.unavailable}
                            </div>
                          </div>

                          {localizedLlmWarning && (
                            <div className="border-s-2 border-s-warning px-3 py-2 text-xs font-medium leading-5 text-warning">
                              {localizedLlmWarning}
                            </div>
                          )}

                          <div className="flex gap-4 max-w-lg">
                            <div className="flex-1 flex flex-col gap-0">
                              <label className={LabelClass}>{tr("studio.wordCountMin", "Min Words")}</label>
                              <input aria-label={tr("studio.wordCountMin", "Min Words")} type="number" min={ARTICLE_WORD_COUNT_MIN} max={ARTICLE_WORD_COUNT_MAX} step={1} className={InputClass} value={wordCountMin} onChange={e => setWordCountMin(sanitizeArticleWordCountInput(e.target.value))} onBlur={() => setWordCountMin((value) => clampArticleWordCountInput(value))} dir="ltr" />
                            </div>
                            <div className="flex-1 flex flex-col gap-0">
                              <label className={LabelClass}>{tr("studio.wordCountMax", "Max Words")}</label>
                              <input aria-label={tr("studio.wordCountMax", "Max Words")} type="number" min={ARTICLE_WORD_COUNT_MIN} max={ARTICLE_WORD_COUNT_MAX} step={1} className={InputClass} value={wordCountMax} onChange={e => setWordCountMax(sanitizeArticleWordCountInput(e.target.value))} onBlur={() => setWordCountMax((value) => clampArticleWordCountInput(value))} dir="ltr" />
                            </div>
                          </div>

                          <div className="flex flex-col gap-1 w-full max-w-lg">
                            <label className={LabelClass}>{tr("studio.creativity", "Creativity")}</label>
                            <div className="flex items-center gap-4 mt-2">
                              <input aria-label={tr("studio.creativity", "Creativity")} type="range" min="0" max="1" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="h-1.5 flex-1 appearance-none rounded-full bg-ink/10 accent-brand outline-none" dir="ltr" />
                              <span className="w-12 shrink-0 text-center text-base font-semibold text-brand">{temperature.toFixed(1)}</span>
                            </div>
                            <p className="text-xs text-ink-muted mt-1 font-medium">0.1 - 0.4: {tr("studio.creativityConservative", "Conservative")} | 0.5 - 0.7: {tr("studio.creativityBalanced", "Balanced")} | 0.8 - 1.0: {tr("studio.creativityCreative", "Creative")}</p>
                          </div>
                        </div>

                      </div>
                    </details>

                    <div className="col-span-12 flex justify-end border-t border-line pt-5">
                      <button
                        type="submit"
                        disabled={!selectedProjectId || submitting || generationBlocked || selectedModelUnavailable}
                        className="flex min-h-[40px] w-full items-center justify-center gap-2 rounded-md bg-brand px-5 py-2 text-sm font-semibold text-white transition-colors duration-fast hover:bg-brand-hover disabled:cursor-not-allowed disabled:opacity-50 md:w-auto"
                      >
                        {submitting && <span className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-white/30 border-t-white" />}
                        {tr("studio.generate", "Generate Article")}
                      </button>
                    </div>
                  </div>
                </div>

              </form>
            )}

            {activeTab === "bulk" && (
              <div className="h-full flex flex-col pb-8 p-6 lg:p-8">
                <form onSubmit={onSubmitBatch} className="mx-auto flex h-full w-full max-w-4xl flex-col gap-4">
                  <div className="flex flex-col gap-4 border-t border-line pt-5">
                    <h2 className={HeaderClass}>{tr("studio.parametersLimits", "Parameters")}</h2>
                    <div className="flex flex-col md:flex-row gap-4">
                      <div className="flex-1 flex flex-col gap-0">
                        <label className={LabelClass}>{tr("studio.batchLanguage", "Language")}</label>
                        <select aria-label={tr("studio.batchLanguage", "Language")} className={InputClass} value={bulkLanguage} onChange={e => setBulkLanguage(e.target.value as GenerationLanguage)}>
                          <option value="fa">{tr("lang.fa", "فارسی")}</option>
                          <option value="ar">{tr("lang.ar", "العربية")}</option>
                          <option value="en">{tr("lang.en", "English")}</option>
                        </select>
                      </div>
                      <div className="flex-1 flex flex-col gap-0">
                        <label className={LabelClass}>{tr("studio.batchSharedKeyword", "Shared Keyword")}</label>
                        <input aria-label={tr("studio.batchSharedKeyword", "Shared Keyword")} className={InputClass} value={bulkKeyword} onChange={e => setBulkKeyword(e.target.value)} />
                      </div>
                    </div>
                    <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(220px,auto)]">
                      <div className="flex flex-col gap-0">
                        <label className={LabelClass}>{modelCopy.label}</label>
                        <select
                          aria-label={modelCopy.label}
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
                                {formatModelDisplayName(option.model)}
                              </option>
                            ))
                          ) : (
                            <option value="">{modelCopy.unavailable}</option>
                          )}
                        </select>
                      </div>
                      <div className="flex min-h-[36px] items-center border-s border-line px-3 py-2 text-xs font-medium text-ink-muted">
                        {selectedModelOption
                          ? `${selectedModelOption.recommended ? modelCopy.recommended : modelCopy.active} · ${formatModelDisplayName(selectedModelOption.model)}`
                          : llmOptionsLoading
                            ? modelCopy.loading
                            : modelCopy.unavailable}
                      </div>
                    </div>
                    {localizedLlmWarning && (
                      <div className="border-s-2 border-s-warning px-3 py-2 text-xs font-medium leading-5 text-warning">
                        {localizedLlmWarning}
                      </div>
                    )}
                  </div>

                  <div className="flex flex-1 flex-col border-t border-line pt-5">
                    <h2 className={HeaderClass}>{tr("studio.coreIdentity", "Core Identity")}</h2>
                    <div className="flex flex-col gap-0 flex-1 h-full min-h-[300px]">
                      <label className={LabelClass}>{tr("studio.batchTopics", "Topics (One per line)")}</label>
                      <textarea required aria-label={tr("studio.batchTopics", "Topics (One per line)")} className={clsx(InputClass, "resize-none h-full py-4")} placeholder={tr("studio.batchTopicsPlaceholder", "Topic 1\nTopic 2\nTopic 3")} value={bulkTopics} onChange={e => setBulkTopics(e.target.value)} />
                      <p className="text-xs font-bold text-ink-muted mt-2">{tr("studio.maxTopics", "Max 20 topics allowed.")}</p>
                    </div>
                  </div>

                  <div className="flex w-full justify-end mt-2 pb-8">
                    <button type="submit" disabled={!selectedProjectId || bulkSubmitting || generationBlocked || selectedModelUnavailable} className="flex min-h-[40px] w-full items-center justify-center gap-2 rounded-md bg-brand px-5 py-2 text-sm font-semibold text-white transition-colors duration-fast hover:bg-brand-hover disabled:cursor-not-allowed disabled:opacity-50 md:w-auto">
                      {bulkSubmitting && <span className="w-5 h-5 rounded-full border-[3px] border-white/30 border-t-white animate-spin shrink-0" />}
                      {tr("studio.batchSubmit", "Submit Batch Pipeline")}
                    </button>
                  </div>
                </form>
              </div>
            )}

            {activeTab === "social" && (
              <div className="p-6 lg:p-8 max-w-5xl mx-auto w-full">
                {socialPosts.length > 0 ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-line pb-4">
                      <div>
                        <h3 className="text-xl font-semibold text-ink">{socialTabLabel}</h3>
                        <p className="mt-1 text-sm font-medium text-ink-muted">{socialReadyText}</p>
                      </div>
                      <span className="rounded-lg border border-success/20 bg-success/10 px-3 py-1.5 text-xs font-semibold text-success">
                        {localizeExecutionState(socialStatus?.state ?? "SUCCESS", locale)}
                      </span>
                    </div>

                    <div className="grid gap-4 lg:grid-cols-3">
                      {socialPosts.map((post) => (
                        <article key={post.platform} className="flex min-h-[260px] flex-col rounded-xl border border-line bg-surface p-4">
                          <div className="mb-3 flex items-center justify-between gap-3">
                            <h4 className="text-sm font-semibold text-ink">{platformLabel(post.platform)}</h4>
                            <Button variant="ghost" size="sm" onClick={() => void copyToClipboard(post.content)}>
                              {t("common.copy")}
                            </Button>
                          </div>
                          <textarea
                            readOnly
                            aria-label={platformLabel(post.platform)}
                            value={post.content}
                            className="min-h-0 flex-1 resize-none rounded-lg border border-line bg-surface-alt p-3 text-sm leading-6 text-ink outline-none"
                            dir="auto"
                          />
                        </article>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex min-h-[300px] flex-col items-center justify-center p-8 text-center">
                    <div className="mb-4 flex h-10 w-10 items-center justify-center text-ink-tertiary">
                      <svg className="h-5 w-5 text-current" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" /></svg>
                    </div>
                    <h3 className="text-lg font-bold text-ink mb-2">{socialTabLabel}</h3>
                    <p className="max-w-sm text-base font-medium leading-relaxed text-ink-muted">
                      {socialTaskId
                        ? socialStatus?.state === "FAILURE"
                          ? extractError(socialStatus.error ?? socialStatus.last_error)
                          : socialLoadingText
                        : socialEmptyText}
                    </p>
                  </div>
                )}
              </div>
            )}

            {activeTab === "schema" && (
              <div className="p-6 lg:p-8 max-w-5xl mx-auto w-full">
                {articleId ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-line pb-4">
                      <div>
                        <h3 className="text-xl font-semibold text-ink">{schemaTabLabel}</h3>
                        <p className="mt-1 text-sm font-medium text-ink-muted">
                          {schemaLoading ? loadingSchemaText : schemaReadyText}
                        </p>
                      </div>
                      {schemaTitle && (
                        <span className="max-w-xs truncate rounded-lg border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink-secondary" dir="auto">
                          {schemaTitle}
                        </span>
                      )}
                    </div>

                    {schemaError && (
                      <div className="rounded-xl border border-danger/20 bg-danger/10 px-4 py-3 text-sm font-medium text-danger" role="alert">
                        {schemaError}
                      </div>
                    )}

                    <div className="grid gap-4 xl:grid-cols-2">
                      <article className="flex min-h-[420px] flex-col rounded-xl border border-line bg-surface">
                        <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
                          <h4 className="text-sm font-semibold text-ink">JSON-LD</h4>
                          <div className="flex items-center gap-2">
                            <Button variant="ghost" size="sm" disabled={!schemaJson} onClick={() => void copyToClipboard(schemaJson)}>
                              {t("common.copy")}
                            </Button>
                            <Button variant="outlined" size="sm" disabled={!schemaJson || !articleId} onClick={() => downloadTextFile(schemaJson, `${articleId}-jsonld.json`, "application/ld+json;charset=utf-8")}>
                              {t("common.download")}
                            </Button>
                          </div>
                        </div>
                        <pre className="min-h-0 flex-1 overflow-auto p-4 text-xs leading-5 text-ink-secondary" dir="ltr">
                          {schemaLoading && !schemaJson ? loadingSchemaText : schemaJson || schemaEmptyText}
                        </pre>
                      </article>

                      <article className="flex min-h-[420px] flex-col rounded-xl border border-line bg-surface">
                        <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
                          <h4 className="text-sm font-semibold text-ink">HTML</h4>
                          <div className="flex items-center gap-2">
                            <Button variant="ghost" size="sm" disabled={!exportHtml} onClick={() => void copyToClipboard(exportHtml)}>
                              {t("common.copy")}
                            </Button>
                            <Button variant="outlined" size="sm" disabled={!exportHtml || !articleId} onClick={() => downloadTextFile(exportHtml, `${articleId}.html`, "text/html;charset=utf-8")}>
                              {t("common.download")}
                            </Button>
                          </div>
                        </div>
                        <pre className="min-h-0 flex-1 overflow-auto p-4 text-xs leading-5 text-ink-secondary" dir="ltr">
                          {schemaLoading && !exportHtml ? loadingSchemaText : exportHtml || schemaEmptyText}
                        </pre>
                      </article>
                    </div>
                  </div>
                ) : (
                  <div className="flex min-h-[300px] flex-col items-center justify-center p-8 text-center">
                    <div className="mb-4 flex h-10 w-10 items-center justify-center text-ink-tertiary">
                      <svg className="h-5 w-5 text-current" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>
                    </div>
                    <h3 className="text-lg font-bold text-ink mb-2">{schemaTabLabel}</h3>
                    <p className="text-base font-medium text-ink-muted max-w-sm leading-relaxed">
                      {schemaEmptyText}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </article>

        {/* Active execution inspector */}
        {(taskStatus || batchStatus) && (
          <aside className="sticky top-0 flex h-full w-full max-w-none flex-col gap-4 self-start overflow-y-auto border-s border-line bg-[rgb(var(--bg-secondary))] p-5">
            <h3 className="text-base font-semibold text-ink">{EXECUTION_COPY[locale].taskStatus}</h3>

            {taskStatus && (
              <div className="relative space-y-3 overflow-hidden border-t border-line pt-4">
                <div className={clsx(
                  "absolute top-0 start-0 h-full w-1 p-0.5",
                  executionState === "FAILURE" ? "bg-danger" : executionState === "SUCCESS" ? "bg-success" : "bg-brand",
                )} />
                <div className="flex items-center justify-between ps-3">
                  <div className="flex items-center gap-2">
                    <span className={clsx(
                      "w-2 h-2 rounded-full",
                      executionState === "SUCCESS" ? "bg-success" :
                        executionState === "FAILURE" ? "bg-danger" :
                          "bg-brand animate-pulse"
                    )} />
                    <span className="text-xs font-semibold text-ink-secondary uppercase">
                      {localizeExecutionState(taskStatus.state, locale)}
                    </span>
                  </div>
                  {!taskStatus.ready && <span className="text-xs font-medium text-brand">{EXECUTION_COPY[locale].live}</span>}
                </div>
                <code className="block text-xs text-ink-muted font-mono truncate ps-3" dir="ltr">#{taskStatus.task_id.slice(0, 8)}</code>
                {taskStatus.status && (
                  <p className="text-sm text-ink-secondary leading-relaxed font-medium ps-3">
                    {localizeExecutionMessage(taskStatus.status, locale)}
                  </p>
                )}
                {executionState === "FAILURE" && executionDiagnostics && (
                  <section className="ms-3 border-t border-danger/20 pt-3">
                    <h4 className="text-xs font-semibold text-danger">
                      {EXECUTION_COPY[locale].releaseGate}
                    </h4>
                    <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-2">
                      <div>
                        <dt className="text-xs text-ink-muted">{EXECUTION_COPY[locale].wordCount}</dt>
                        <dd className="mt-0.5 text-xs font-semibold text-ink">{formatExecutionNumber(executionDiagnostics.actual_word_count, locale)}</dd>
                      </div>
                      <div>
                        <dt className="text-xs text-ink-muted">{EXECUTION_COPY[locale].wordRange}</dt>
                        <dd className="mt-0.5 text-xs font-semibold text-ink">
                          {formatExecutionNumber(executionDiagnostics.min_word_count, locale)}–{formatExecutionNumber(executionDiagnostics.max_word_count, locale)}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-xs text-ink-muted">{EXECUTION_COPY[locale].headings}</dt>
                        <dd className="mt-0.5 text-xs font-semibold text-ink">{formatExecutionNumber(executionDiagnostics.headings_count, locale)}</dd>
                      </div>
                      <div>
                        <dt className="text-xs text-ink-muted">{EXECUTION_COPY[locale].paragraphs}</dt>
                        <dd className="mt-0.5 text-xs font-semibold text-ink">{formatExecutionNumber(executionDiagnostics.paragraphs_count, locale)}</dd>
                      </div>
                    </dl>
                    {executionDiagnostics.findings?.length ? (
                      <p className="mt-2 text-xs font-medium text-danger">
                        {EXECUTION_COPY[locale].findings}: {formatExecutionNumber(executionDiagnostics.findings.length, locale)}
                      </p>
                    ) : null}
                  </section>
                )}
                {(
                  (typeof taskStatus.error === "string" && !taskStatus.error.includes("[object Object]"))
                  || (typeof taskStatus.manager_error_detail === "string" && !taskStatus.manager_error_detail.includes("[object Object]"))
                ) && (
                  <details className="border-s-2 border-s-error px-3 py-2 text-xs leading-5 text-error">
                    <summary className="cursor-pointer font-semibold">{EXECUTION_COPY[locale].technicalDetails}</summary>
                    {typeof taskStatus.error === "string" && !taskStatus.error.includes("[object Object]") && (
                      <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap font-mono" dir="ltr">{taskStatus.error}</pre>
                    )}
                    {typeof taskStatus.manager_error_detail === "string" && !taskStatus.manager_error_detail.includes("[object Object]") && (
                      <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap border-t border-danger/20 pt-2 font-mono" dir="ltr">{taskStatus.manager_error_detail}</pre>
                    )}
                  </details>
                )}
              </div>
            )}

            {batchStatus && (
              <div className="space-y-4 border-t border-line pt-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-bold text-ink">{tr("studio.batchProgress", "Pipeline Progress")}</h4>
                  <span className="font-mono text-xs font-semibold text-ink-muted">
                    {batchStatus.completed} / {batchStatus.total}
                  </span>
                </div>
                <div className="h-1.5 w-full bg-surface-tertiary rounded-full overflow-hidden">
                  <div className="h-full bg-ink/70 bg-success/80 transition-all duration-500 ease-out" style={{ width: `${batchStatus.total ? (batchStatus.completed / batchStatus.total) * 100 : 0}%` }} />
                </div>
              </div>
            )}
          </aside>
        )}
      </div>
    </section>
  );
}
