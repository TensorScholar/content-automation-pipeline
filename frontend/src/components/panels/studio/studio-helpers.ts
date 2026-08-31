"use client";

import { ApiError, apiRequest } from "@/lib/api";
import { ProjectReadiness, TaskStatusResponse } from "@/types/models";

interface SocialPost {
  platform: string;
  content: string;
}

export type ReadinessItem = ProjectReadiness["blocking_items"][number];

export function isWordPressReadinessItem(item: ReadinessItem): boolean {
  const stableId = item.id.trim().toLowerCase();
  if (/(^|[._:-])wordpress($|[._:-])/.test(stableId)) return true;

  return /\bwordpress\b|وردپرس|ووردبريس/i.test(item.message);
}

export function isContentRulesReadinessItem(item: ReadinessItem): boolean {
  const stableId = item.id.trim().toLowerCase();
  if (/(content[._:-]?rules?|rulebook|editorial[._:-]?rules?)/.test(stableId)) return true;

  return /content rulebook|content rules|قوانین محتوا|قاعده.*محتوا|قواعد المحتوى/i.test(item.message);
}

export type GenerationBlockerKind = "contentRules" | "worker" | "aiProvider" | "budget" | "projectProfile" | "system";

export function generationBlockerKind(items: ReadinessItem[]): GenerationBlockerKind {
  const text = items.map((item) => `${item.id} ${item.label} ${item.message} ${item.remediation ?? ""}`.toLowerCase()).join(" ");
  if (items.some(isContentRulesReadinessItem)) return "contentRules";
  if (/worker|celery|queue|صف پردازش|عامل معالجة/.test(text)) return "worker";
  if (/ai provider|llm|model|api key|ارائه‌دهنده هوش|مدل|مزود الذكاء/.test(text)) return "aiProvider";
  if (/budget|cost limit|بودجه|ميزانية/.test(text)) return "budget";
  if (/project profile|project identity|مشخصات پروژه|ملف المشروع/.test(text)) return "projectProfile";
  return "system";
}

export type StudioTab = "generate" | "bulk" | "social" | "schema";
export type GenerationLanguage = "fa" | "ar" | "en";

export const OUTPUT_LANGUAGE_INSTRUCTION: Record<GenerationLanguage, string> = {
  fa: "Output language must be Persian (Farsi).",
  ar: "Output language must be Arabic.",
  en: "Output language must be English.",
};

export const EXECUTION_COPY = {
  en: { success: "Success", failed: "Failed", running: "Running", pending: "Pending", live: "Live", taskStatus: "Execution Status", completed: "Generation completed successfully", processing: "Generation is being processed", queued: "Generation request queued.", failureSummary: "Content generation did not pass the release checks.", releaseGate: "Release gate", wordCount: "Words", wordRange: "Allowed", headings: "Headings", paragraphs: "Paragraphs", findings: "Findings", technicalDetails: "Technical details", wordCountBounds: "Article word counts must be between 800 and 3500." },
  fa: { success: "موفق", failed: "ناموفق", running: "در حال اجرا", pending: "در انتظار", live: "زنده", taskStatus: "وضعیت اجرا", completed: "پردازش با موفقیت کامل شد", processing: "پردازش در حال انجام است", queued: "درخواست تولید در صف قرار گرفت.", failureSummary: "تولید محتوا از بررسی‌های نهایی عبور نکرد.", releaseGate: "دروازه کیفیت", wordCount: "تعداد کلمات", wordRange: "بازه مجاز", headings: "تیترها", paragraphs: "پاراگراف‌ها", findings: "یافته‌ها", technicalDetails: "جزئیات فنی", wordCountBounds: "تعداد کلمات مقاله باید بین ۸۰۰ تا ۳۵۰۰ باشد." },
  ar: { success: "ناجح", failed: "فشل", running: "قيد التشغيل", pending: "قيد الانتظار", live: "مباشر", taskStatus: "حالة التنفيذ", completed: "اكتمل طلب الإنشاء بنجاح", processing: "طلب الإنشاء قيد المعالجة", queued: "تمت إضافة طلب الإنشاء إلى قائمة الانتظار.", failureSummary: "لم يجتز إنشاء المحتوى فحوص الإصدار النهائية.", releaseGate: "بوابة الجودة", wordCount: "الكلمات", wordRange: "النطاق المسموح", headings: "العناوين", paragraphs: "الفقرات", findings: "النتائج", technicalDetails: "التفاصيل التقنية", wordCountBounds: "يجب أن يكون عدد كلمات المقال بين 800 و3500." },
} as const;

export const ARTICLE_WORD_COUNT_MIN = 800;
export const ARTICLE_WORD_COUNT_MAX = 3500;

export function localizeExecutionState(state: string, locale: keyof typeof EXECUTION_COPY) {
  const normalized = state.trim().toUpperCase();
  if (normalized === "SUCCESS") return EXECUTION_COPY[locale].success;
  if (normalized === "FAILURE" || normalized === "FAILED") return EXECUTION_COPY[locale].failed;
  if (normalized === "PENDING") return EXECUTION_COPY[locale].pending;
  if (normalized === "QUEUED") return EXECUTION_COPY[locale].queued;
  return EXECUTION_COPY[locale].running;
}

export function localizeExecutionMessage(message: string, locale: keyof typeof EXECUTION_COPY) {
  const normalized = message.trim().toLowerCase().replace(/[.!]+$/, "");
  if (normalized === "task completed successfully") return EXECUTION_COPY[locale].completed;
  if (normalized === "task is being processed") return EXECUTION_COPY[locale].processing;
  if (normalized === "queued" || normalized === "task is queued") return EXECUTION_COPY[locale].queued;
  if (normalized === "task failed") return EXECUTION_COPY[locale].failureSummary;
  return message;
}

export function formatExecutionNumber(value: number | undefined, locale: keyof typeof EXECUTION_COPY) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  return new Intl.NumberFormat(localeName).format(value);
}

export const TONE_OPTIONS = [
  { value: "professional", fa: "حرفه‌ای", ar: "مهني", en: "Professional" },
  { value: "friendly", fa: "دوستانه", ar: "ودي", en: "Friendly" },
  { value: "formal", fa: "رسمی", ar: "رسمي", en: "Formal" },
  { value: "persuasive", fa: "متقاعدکننده", ar: "مقنع", en: "Persuasive" },
  { value: "educational", fa: "آموزشی", ar: "تعليمي", en: "Educational" },
];
export const STRUCTURE_OPTIONS = [
  { value: "standard", fa: "استاندارد", ar: "قياسي", en: "Standard" },
  { value: "listicle", fa: "فهرستی", ar: "قائمة", en: "Listicle" },
  { value: "howto", fa: "آموزشی (How-to)", ar: "كيفية (How-to)", en: "How-to Guide" },
  { value: "comparison", fa: "مقایسه‌ای", ar: "مقارنة", en: "Comparison" },
  { value: "pillar", fa: "ستونی (Pillar)", ar: "ركيزة (Pillar)", en: "Pillar Page" },
];
export const POV_OPTIONS = [
  { value: "first_person", fa: "اول شخص", ar: "ضمير المتكلم", en: "First person" },
  { value: "second_person", fa: "دوم شخص", ar: "ضمير المخاطب", en: "Second person" },
  { value: "third_person", fa: "سوم شخص", ar: "ضمير الغائب", en: "Third person" },
];
export const AUDIENCE_OPTIONS = [
  { value: "general", fa: "عمومی", ar: "عام", en: "General" },
  { value: "technical", fa: "فنی و تخصصی", ar: "تقني ومتخصص", en: "Technical" },
  { value: "beginner", fa: "مبتدی", ar: "مبتدئ", en: "Beginner" },
  { value: "business", fa: "مدیران و کسب‌وکار", ar: "رجال الأعمال", en: "Business professionals" },
];

export const READINESS_COPY = {
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

export function localizeTechnicalMessage(message: string, locale: keyof typeof READINESS_COPY) {
  const normalized = message.trim().toLowerCase();
  if (normalized.includes("llm ping timed out") || normalized.includes("ai provider ping timed out")) {
    return READINESS_COPY[locale].llmTimeout;
  }
  return localizeExecutionMessage(message, locale);
}

export const MODEL_COPY = {
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

export function extractError(e: unknown): string {
  if (e instanceof ApiError) return e.detail;
  return "Unexpected error";
}

export function refreshTask(taskId: string, token: string, signal?: AbortSignal) {
  return apiRequest<TaskStatusResponse>(`/content/task/${taskId}`, { token, signal });
}

export function buildWordCountPayload(minRaw: string, maxRaw: string): {
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

export function sanitizeArticleWordCountInput(value: string): string {
  return value.replace(/[^\d]/g, "");
}

export function clampArticleWordCountInput(value: string): string {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return "";
  return String(Math.min(ARTICLE_WORD_COUNT_MAX, Math.max(ARTICLE_WORD_COUNT_MIN, Math.round(numericValue))));
}

export function getSocialPosts(status: TaskStatusResponse | null): SocialPost[] {
  const posts = status?.result?.posts;
  if (typeof posts !== "object" || posts === null || Array.isArray(posts)) {
    return [];
  }

  return Object.entries(posts)
    .filter((entry): entry is [string, string] => typeof entry[1] === "string" && entry[1].trim().length > 0)
    .map(([platform, content]) => ({ platform, content }));
}

export function downloadTextFile(content: string, filename: string, type = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function platformLabel(platform: string) {
  if (platform === "twitter") return "X / Twitter";
  return platform.charAt(0).toUpperCase() + platform.slice(1);
}