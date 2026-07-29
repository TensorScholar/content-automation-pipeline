"use client";

import { FormEvent, useEffect, useMemo, useState, useRef } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import {
  PerformanceImportResponse,
  PerformanceOpportunity,
  PerformanceSnapshot,
  Project,
  ProjectPerformanceFeedback,
  ProjectReadiness,
} from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { useToast } from "@/components/ui/toast";
import { InputField } from "@/components/ui/input-field";
import { SelectDropdown } from "@/components/ui/select-dropdown";
import type { SelectOption } from "@/components/ui/select-dropdown";

/* ═══════════════════════════════════════════════════════════════
   Spec: Screen 3 — Projects Page (Apple/Linear SaaS UI Tier)
   Architecture: 100vh Master-Detail split layout (NO SCROLL)
   - Left (30%): macOS-style Sidebar Project List
   - Right (70%): Tabbed Editor (Zero layout shift, Spatial Forms)
   ═══════════════════════════════════════════════════════════════ */

interface ProjectsPanelProps {
  token: string;
  projects: Project[];
  selectedProjectId: string | null;
  canManageProjects: boolean;
  onSelectProject: (projectId: string | null) => void;
  onProjectsRefresh: () => Promise<void>;
}

interface RulebookResponse { content?: string; }

type ProjectTab = "general" | "wordpress" | "rules" | "readiness" | "performance";

const VERTICAL_OPTIONS = [
  { value: "tech", fa: "فناوری و نرم‌افزار", ar: "التكنولوجيا والبرمجيات", en: "Technology and Software" },
  { value: "health", fa: "سلامت و پزشکی", ar: "الصحة والطب", en: "Health and Medical" },
  { value: "ecommerce", fa: "فروشگاه و تجارت", ar: "المتاجر والتجارة", en: "E-Commerce" },
  { value: "education", fa: "آموزش و یادگیری", ar: "التعليم والتعلم", en: "Education and Learning" },
  { value: "finance", fa: "مالی و اقتصادی", ar: "المالية والاقتصاد", en: "Finance and Economy" },
  { value: "marketing", fa: "بازاریابی دیجیتال", ar: "التسويق الرقمي", en: "Digital Marketing" },
];

const READINESS_COPY = {
  en: {
    tab: "Flight Check",
    title: "Project Flight Check",
    subtitle: "Operational readiness for generation and publishing.",
    loading: "Checking project readiness...",
    refresh: "Run check",
    ready: "Ready",
    warning: "Needs review",
    blocked: "Blocked",
    canGenerate: "Generation",
    canPublish: "Publishing",
    available: "Available",
    unavailable: "Unavailable",
    blockers: "Blocking items",
    warnings: "Warnings",
    allChecks: "All checks",
    actions: "Manager actions",
    noActions: "No action required.",
    lastChecked: "Last checked",
    failed: "Readiness check failed.",
    openRulebook: "Open content rules",
    openWordPress: "Configure WordPress",
  },
  fa: {
    tab: "بررسی آماده‌سازی",
    title: "بررسی آماده‌سازی پروژه",
    subtitle: "آمادگی عملیاتی برای تولید و انتشار محتوا.",
    loading: "در حال بررسی آمادگی پروژه...",
    refresh: "اجرای بررسی",
    ready: "آماده",
    warning: "نیازمند بررسی",
    blocked: "مسدود",
    canGenerate: "تولید محتوا",
    canPublish: "انتشار",
    available: "فعال",
    unavailable: "غیرفعال",
    blockers: "موارد مسدودکننده",
    warnings: "هشدارها",
    allChecks: "همه بررسی‌ها",
    actions: "اقدام‌های مدیر",
    noActions: "اقدامی لازم نیست.",
    lastChecked: "آخرین بررسی",
    failed: "بررسی آمادگی ناموفق بود.",
    openRulebook: "باز کردن قوانین محتوا",
    openWordPress: "تنظیم اتصال وردپرس",
  },
  ar: {
    tab: "فحص الجاهزية",
    title: "فحص جاهزية المشروع",
    subtitle: "الجاهزية التشغيلية للإنشاء والنشر.",
    loading: "جارٍ فحص جاهزية المشروع...",
    refresh: "تشغيل الفحص",
    ready: "جاهز",
    warning: "يحتاج مراجعة",
    blocked: "محظور",
    canGenerate: "إنشاء المحتوى",
    canPublish: "النشر",
    available: "متاح",
    unavailable: "غير متاح",
    blockers: "العناصر المانعة",
    warnings: "التحذيرات",
    allChecks: "كل الفحوصات",
    actions: "إجراءات المدير",
    noActions: "لا يلزم إجراء.",
    lastChecked: "آخر فحص",
    failed: "فشل فحص الجاهزية.",
    openRulebook: "فتح قواعد المحتوى",
    openWordPress: "إعداد اتصال ووردبريس",
  },
};

type ReadinessLocale = keyof typeof READINESS_COPY;

const READINESS_ITEM_COPY = {
  en: {
    labels: {
      projectProfile: "Project profile",
      contentRules: "Content rules",
      wordpressPublishing: "WordPress publishing",
      aiProvider: "AI provider",
      redis: "Redis broker/cache",
      workerQueue: "Worker queue",
      dailyBudget: "Daily budget",
      recentFailures: "Recent failures",
    },
    projectConfigured: "Project identity is configured.",
    noRulebook: "No content rulebook is configured.",
    addRules: "Add brand voice, SEO, and editorial rules for more consistent output.",
    wordpressMissing: "WordPress is not connected. Content generation remains available, but WordPress publishing is unavailable.",
    configureWordPress: "Complete the WordPress connection before publishing.",
    providerConfigured: "AI provider credentials are configured.",
    localProvider: "Configured provider: Local LLM.",
    redisReachable: "Redis is reachable.",
    activeWorkers: (count: string) => `${count} worker${count === "1" ? "" : "s"} are available.`,
    noWorkers: "No active workers were detected.",
    startWorker: "Start at least one processing worker before generation.",
    budgetWithinLimit: "Daily AI cost is within the configured limit.",
    llmTimeout: "The AI provider health check timed out.",
    noRecentFailures: "No recent project task failures detected.",
    domainMissing: "Project domain is missing.",
    addDomain: "Add a domain to improve analysis, SEO context, and publishing readiness.",
    rulebookConfigured: "Content rulebook is configured.",
  },
  fa: {
    labels: {
      projectProfile: "مشخصات پروژه",
      contentRules: "قوانین محتوا",
      wordpressPublishing: "انتشار در وردپرس",
      aiProvider: "ارائه‌دهنده هوش مصنوعی",
      redis: "کارگزار و حافظه موقت Redis",
      workerQueue: "صف پردازش",
      dailyBudget: "بودجه روزانه",
      recentFailures: "خطاهای اخیر",
    },
    projectConfigured: "هویت پروژه تنظیم شده است.",
    noRulebook: "هنوز قوانین محتوایی برای پروژه تنظیم نشده است.",
    addRules: "برای خروجی منسجم‌تر، لحن برند و قواعد سئو و ویرایشی را اضافه کنید.",
    wordpressMissing: "وردپرس متصل نیست. تولید محتوا فعال می‌ماند، اما انتشار در وردپرس در دسترس نیست.",
    configureWordPress: "پیش از انتشار، اتصال وردپرس را تکمیل کنید.",
    providerConfigured: "اعتبارنامه ارائه‌دهنده هوش مصنوعی تنظیم شده است.",
    localProvider: "ارائه‌دهنده تنظیم‌شده: مدل محلی.",
    redisReachable: "اتصال Redis برقرار است.",
    activeWorkers: (count: string) => `${new Intl.NumberFormat("fa-IR").format(Number(count))} پردازشگر فعال شناسایی شد.`,
    noWorkers: "هیچ پردازشگر فعالی شناسایی نشد.",
    startWorker: "پیش از تولید محتوا، حداقل یک پردازشگر را فعال کنید.",
    budgetWithinLimit: "هزینه روزانه هوش مصنوعی در محدوده بودجه تنظیم‌شده است.",
    llmTimeout: "زمان بررسی سلامت ارائه‌دهنده هوش مصنوعی به پایان رسید.",
    noRecentFailures: "هیچ خطای اخیر در پردازش‌های پروژه ثبت نشده است.",
    domainMissing: "دامنه پروژه تنظیم نشده است.",
    addDomain: "برای بهبود تحلیل، زمینه سئو و آمادگی انتشار، دامنه پروژه را اضافه کنید.",
    rulebookConfigured: "قوانین محتوای پروژه تنظیم شده است.",
  },
  ar: {
    labels: {
      projectProfile: "ملف المشروع",
      contentRules: "قواعد المحتوى",
      wordpressPublishing: "النشر إلى ووردبريس",
      aiProvider: "مزود الذكاء الاصطناعي",
      redis: "وسيط وذاكرة Redis",
      workerQueue: "قائمة انتظار المعالجة",
      dailyBudget: "الميزانية اليومية",
      recentFailures: "الإخفاقات الأخيرة",
    },
    projectConfigured: "تم إعداد هوية المشروع.",
    noRulebook: "لم يتم إعداد قواعد محتوى للمشروع.",
    addRules: "أضف صوت العلامة وقواعد تحسين البحث والتحرير للحصول على مخرجات أكثر اتساقاً.",
    wordpressMissing: "ووردبريس غير متصل. يظل إنشاء المحتوى متاحاً، لكن النشر إلى ووردبريس غير متاح.",
    configureWordPress: "أكمل اتصال ووردبريس قبل النشر.",
    providerConfigured: "تم إعداد بيانات اعتماد مزود الذكاء الاصطناعي.",
    localProvider: "المزود المُعد: نموذج محلي.",
    redisReachable: "اتصال Redis متاح.",
    activeWorkers: (count: string) => `${new Intl.NumberFormat("ar").format(Number(count))} عامل معالجة نشط متاح.`,
    noWorkers: "لم يتم العثور على أي عامل معالجة نشط.",
    startWorker: "شغّل عامل معالجة واحداً على الأقل قبل إنشاء المحتوى.",
    budgetWithinLimit: "تكلفة الذكاء الاصطناعي اليومية ضمن الحد المحدد.",
    llmTimeout: "انتهت مهلة فحص صحة مزود الذكاء الاصطناعي.",
    noRecentFailures: "لم يتم تسجيل أي إخفاقات حديثة في مهام المشروع.",
    domainMissing: "نطاق المشروع غير مُعد.",
    addDomain: "أضف نطاقاً لتحسين التحليل وسياق السيو وجاهزية النشر.",
    rulebookConfigured: "تم إعداد قواعد محتوى المشروع.",
  },
} as const;

function readinessItemKind(id: string, label: string) {
  const value = `${id} ${label}`.toLowerCase();
  if (/wordpress/.test(value)) return "wordpressPublishing" as const;
  if (/content.*rule|rulebook/.test(value)) return "contentRules" as const;
  if (/project.*profile|project.*identity/.test(value)) return "projectProfile" as const;
  if (/ai.*provider|llm.*provider/.test(value)) return "aiProvider" as const;
  if (/redis/.test(value)) return "redis" as const;
  if (/worker|celery|queue/.test(value)) return "workerQueue" as const;
  if (/daily.*budget|budget/.test(value)) return "dailyBudget" as const;
  if (/recent.*failure|failure/.test(value)) return "recentFailures" as const;
  return null;
}

function localizeReadinessLabel(id: string, label: string, locale: ReadinessLocale) {
  const kind = readinessItemKind(id, label);
  return kind ? READINESS_ITEM_COPY[locale].labels[kind] : label;
}

function localizeReadinessText(value: string, locale: ReadinessLocale) {
  const normalized = value.trim().toLowerCase();
  const copy = READINESS_ITEM_COPY[locale];
  if (normalized.includes("project identity is configured")) return copy.projectConfigured;
  if (normalized.includes("no content rulebook is configured")) return copy.noRulebook;
  if (normalized.includes("add brand voice") && normalized.includes("editorial rules")) return copy.addRules;
  if (normalized.includes("wordpress is missing") || normalized.includes("wordpress is not connected")) return copy.wordpressMissing;
  if (normalized.includes("complete the wordpress integration") || normalized.includes("complete the wordpress connection")) return copy.configureWordPress;
  if (normalized.includes("ai provider credentials are configured")) return copy.providerConfigured;
  if (normalized.includes("configured providers: local llm") || normalized.includes("configured provider: local llm")) return copy.localProvider;
  if (normalized.includes("redis is reachable")) return copy.redisReachable;
  const activeWorkerMatch = normalized.match(/(\d+)\s+workers?\(s\)\s+are available|(\d+)\s+workers?\s+are available|(\d+)\s+workers?\s+active/);
  const activeWorkerCount = activeWorkerMatch?.[1] ?? activeWorkerMatch?.[2] ?? activeWorkerMatch?.[3];
  if (activeWorkerCount) return copy.activeWorkers(activeWorkerCount);
  if (normalized.includes("no active celery workers") || normalized.includes("no active workers")) return copy.noWorkers;
  if (normalized.includes("start at least one worker before generation")) return copy.startWorker;
  if (normalized.includes("daily cost is within the configured threshold")) return copy.budgetWithinLimit;
  if (normalized.includes("llm ping timed out") || normalized.includes("ai provider ping timed out")) return copy.llmTimeout;
  if (normalized.includes("no recent project task failures detected")) return copy.noRecentFailures;
  if (normalized.includes("project domain is missing")) return copy.domainMissing;
  if (normalized.includes("add a domain to improve analysis")) return copy.addDomain;
  if (normalized.includes("content rulebook is configured")) return copy.rulebookConfigured;
  return value;
}

function formatReadinessDate(value: string, locale: ReadinessLocale) {
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  try {
    return new Intl.DateTimeFormat(localeName, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(value));
  } catch {
    return value;
  }
}

const PERFORMANCE_COPY = {
  en: {
    tab: "Performance",
    title: "Performance Feedback",
    subtitle: "Read-only Search Console style snapshots for editorial prioritization.",
    refresh: "Refresh",
    import: "Import CSV",
    importTitle: "Import performance snapshot",
    importSubtitle: "Paste a CSV export containing the following columns.",
    importColumns: "url, clicks, impressions, ctr, average_position, date_from, date_to",
    importPlaceholder: "url,clicks,impressions,ctr,average_position,date_from,date_to\nhttps://example.com/article,24,3200,0.75%,11.4,2026-05-01,2026-05-31",
    importEmpty: "Paste CSV data before importing.",
    importSuccess: "Performance snapshot imported.",
    snapshots: "Snapshots",
    opportunities: "Open opportunities",
    highPriority: "High priority",
    latestImport: "Latest import",
    noImport: "No imports yet",
    emptyTitle: "No performance data yet",
    emptyBody: "Import a manual Search Console CSV to surface read-only improvement opportunities. Nothing here rewrites or publishes content.",
    opportunitiesTitle: "Improvement opportunities",
    recentSnapshots: "Recent snapshots",
    dismiss: "Dismiss",
    loading: "Loading performance feedback...",
    failed: "Performance feedback failed.",
    noOpportunities: "No open opportunities. Imported pages do not currently need action by the configured rules.",
    noSnapshots: "No snapshots imported yet.",
    source: "Source",
    period: "Period",
    clicks: "Clicks",
    impressions: "Impressions",
    ctr: "CTR",
    position: "Avg. position",
    previousClicks: "Previous clicks",
    article: "Article",
    unmapped: "Unmapped URL",
    severity: { low: "Low", medium: "Medium", high: "High" },
    types: {
      low_ctr_high_impressions: "Low CTR, high impressions",
      striking_distance_position: "Striking distance",
      declining_clicks: "Declining clicks",
      missing_performance_data: "Missing performance data",
      unmapped_url: "Unmapped URL",
    },
  },
  fa: {
    tab: "عملکرد",
    title: "بازخورد عملکرد",
    subtitle: "اسنپ‌شات‌های خواندنی از عملکرد محتوا برای اولویت‌بندی ویرایشی.",
    refresh: "بروزرسانی",
    import: "ورود CSV",
    importTitle: "ورود اسنپ‌شات عملکرد",
    importSubtitle: "یک خروجی CSV با ستون‌های زیر وارد کنید.",
    importColumns: "url, clicks, impressions, ctr, average_position, date_from, date_to",
    importPlaceholder: "url,clicks,impressions,ctr,average_position,date_from,date_to\nhttps://example.com/article,24,3200,0.75%,11.4,2026-05-01,2026-05-31",
    importEmpty: "قبل از ورود، داده CSV را وارد کنید.",
    importSuccess: "اسنپ‌شات عملکرد وارد شد.",
    snapshots: "اسنپ‌شات‌ها",
    opportunities: "فرصت‌های باز",
    highPriority: "اولویت بالا",
    latestImport: "آخرین ورود",
    noImport: "هنوز ورودی ثبت نشده",
    emptyTitle: "هنوز داده عملکردی وجود ندارد",
    emptyBody: "برای دیدن فرصت‌های بهبود، CSV دستی سرچ کنسول را وارد کنید. این بخش هیچ محتوا را بازنویسی یا منتشر نمی‌کند.",
    opportunitiesTitle: "فرصت‌های بهبود",
    recentSnapshots: "اسنپ‌شات‌های اخیر",
    dismiss: "نادیده گرفتن",
    loading: "در حال بارگذاری بازخورد عملکرد...",
    failed: "بازخورد عملکرد ناموفق بود.",
    noOpportunities: "فرصت بازی وجود ندارد. صفحات واردشده طبق قواعد فعلی نیازمند اقدام نیستند.",
    noSnapshots: "هنوز اسنپ‌شاتی وارد نشده است.",
    source: "منبع",
    period: "بازه",
    clicks: "کلیک",
    impressions: "نمایش",
    ctr: "CTR",
    position: "میانگین رتبه",
    previousClicks: "کلیک قبلی",
    article: "مقاله",
    unmapped: "URL بدون اتصال",
    severity: { low: "کم", medium: "متوسط", high: "بالا" },
    types: {
      low_ctr_high_impressions: "CTR پایین با نمایش بالا",
      striking_distance_position: "نزدیک به صفحه اول",
      declining_clicks: "کاهش کلیک",
      missing_performance_data: "داده عملکرد ناقص",
      unmapped_url: "URL بدون اتصال",
    },
  },
  ar: {
    tab: "الأداء",
    title: "ملاحظات الأداء",
    subtitle: "لقطات قراءة فقط تشبه Search Console لتحديد أولويات التحرير.",
    refresh: "تحديث",
    import: "استيراد CSV",
    importTitle: "استيراد لقطة أداء",
    importSubtitle: "ألصق ملف CSV يحتوي على الأعمدة التالية.",
    importColumns: "url, clicks, impressions, ctr, average_position, date_from, date_to",
    importPlaceholder: "url,clicks,impressions,ctr,average_position,date_from,date_to\nhttps://example.com/article,24,3200,0.75%,11.4,2026-05-01,2026-05-31",
    importEmpty: "الصق بيانات CSV قبل الاستيراد.",
    importSuccess: "تم استيراد لقطة الأداء.",
    snapshots: "اللقطات",
    opportunities: "الفرص المفتوحة",
    highPriority: "أولوية عالية",
    latestImport: "آخر استيراد",
    noImport: "لا توجد واردات بعد",
    emptyTitle: "لا توجد بيانات أداء بعد",
    emptyBody: "استورد CSV يدوي من Search Console لإظهار فرص تحسين قراءة فقط. هذا القسم لا يعيد الكتابة ولا ينشر المحتوى.",
    opportunitiesTitle: "فرص التحسين",
    recentSnapshots: "اللقطات الأخيرة",
    dismiss: "تجاهل",
    loading: "جارٍ تحميل ملاحظات الأداء...",
    failed: "فشل تحميل ملاحظات الأداء.",
    noOpportunities: "لا توجد فرص مفتوحة. الصفحات المستوردة لا تحتاج إجراءً وفق القواعد الحالية.",
    noSnapshots: "لم يتم استيراد أي لقطة بعد.",
    source: "المصدر",
    period: "الفترة",
    clicks: "النقرات",
    impressions: "الظهور",
    ctr: "CTR",
    position: "متوسط الترتيب",
    previousClicks: "النقرات السابقة",
    article: "المقالة",
    unmapped: "رابط غير مربوط",
    severity: { low: "منخفض", medium: "متوسط", high: "عالٍ" },
    types: {
      low_ctr_high_impressions: "CTR منخفض مع ظهور عالٍ",
      striking_distance_position: "قريب من الصفحة الأولى",
      declining_clicks: "انخفاض النقرات",
      missing_performance_data: "بيانات أداء ناقصة",
      unmapped_url: "رابط غير مربوط",
    },
  },
};

function extractError(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  return "Unexpected error";
}

const PROJECT_ERROR_COPY = {
  en: {
    timeout: "The project readiness check did not respond in time. Please try again or check service health.",
  },
  fa: {
    timeout: "بررسی آمادگی پروژه در زمان مجاز پاسخ نداد. لطفاً دوباره تلاش کنید یا وضعیت سرویس‌ها را بررسی کنید.",
  },
  ar: {
    timeout: "لم يستجب فحص جاهزية المشروع ضمن المهلة. يرجى المحاولة مرة أخرى أو التحقق من حالة الخدمات.",
  },
} as const;

function localizeProjectError(error: string, locale: ReadinessLocale): string {
  const normalized = error.trim().toLowerCase();
  if (normalized.includes("request timeout") || normalized.includes("timed out") || normalized === "timeout") {
    return PROJECT_ERROR_COPY[locale].timeout;
  }
  return error;
}

/* ── Hooks ── */
function useClickOutside(ref: React.RefObject<HTMLElement | null>, handler: () => void) {
  useEffect(() => {
    const listener = (e: MouseEvent | TouchEvent) => {
      if (!ref.current || ref.current.contains(e.target as Node)) return;
      handler();
    };
    document.addEventListener("mousedown", listener);
    document.addEventListener("touchstart", listener);
    return () => {
      document.removeEventListener("mousedown", listener);
      document.removeEventListener("touchstart", listener);
    };
  }, [ref, handler]);
}

/* ── Workspace illustration for empty state ── */
function FolderIllustration() {
  return (
    <div className="mx-auto mb-6 flex h-24 w-24 items-center justify-center rounded-[2rem] bg-gradient-to-br from-teal-50 to-teal-100/60 shadow-sm border border-teal-100/50">
      <svg viewBox="0 0 48 48" fill="none" className="h-10 w-10 text-teal-600">
        <path d="M4 12C4 9.79086 5.79086 8 8 8H18.8284C19.8893 8 20.9067 8.42143 21.6569 9.17157L24 11.5147M24 11.5147L26.3431 13.8579C27.0933 14.608 28.1107 15.0294 29.1716 15.0294H40C42.2091 15.0294 44 16.8203 44 19.0294V36C44 38.2091 42.2091 40 40 40H8C5.79086 40 4 38.2091 4 36V12Z" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M14 26H28" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
        <path d="M14 32H22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
      </svg>
    </div>
  );
}

export function ProjectsPanel({
  token, projects, selectedProjectId, canManageProjects, onSelectProject, onProjectsRefresh,
}: ProjectsPanelProps) {
  const { t, locale } = useI18n();
  const { showToast } = useToast();

  // Creation state
  const [creating, setCreating] = useState(false);
  const [newProject, setNewProject] = useState({
    name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "",
  });

  // Editor states
  const [activeTab, setActiveTab] = useState<ProjectTab>("general");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [deletingProjectId, setDeletingProjectId] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<ProjectReadiness | null>(null);
  const [readinessLoading, setReadinessLoading] = useState(false);
  const [readinessError, setReadinessError] = useState<string | null>(null);
  const [performance, setPerformance] = useState<ProjectPerformanceFeedback | null>(null);
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const [performanceError, setPerformanceError] = useState<string | null>(null);
  const [performanceImportOpen, setPerformanceImportOpen] = useState(false);
  const [performanceCsv, setPerformanceCsv] = useState("");
  const [performanceImporting, setPerformanceImporting] = useState(false);
  const [dismissingOpportunityId, setDismissingOpportunityId] = useState<string | null>(null);

  // Kebab Menu State
  const [kebabOpen, setKebabOpen] = useState(false);
  const kebabRef = useRef<HTMLDivElement>(null);
  useClickOutside(kebabRef, () => setKebabOpen(false));

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId]
  );
  const readinessCopy = READINESS_COPY[locale];
  const performanceCopy = PERFORMANCE_COPY[locale];

  // If selected project gets deleted, reset selection
  useEffect(() => {
    if (selectedProjectId === "__new__") return;

    if (selectedProjectId && projects.length > 0 && !projects.find(p => p.id === selectedProjectId)) {
      onSelectProject(projects[0].id);
    } else if (!selectedProjectId && projects.length > 0) {
      onSelectProject(projects[0].id);
    }
  }, [projects, selectedProjectId, onSelectProject]);

  useEffect(() => {
    if (!selectedProject || selectedProjectId === "__new__") {
      setReadiness(null);
      setReadinessError(null);
      setReadinessLoading(false);
      return;
    }

    const controller = new AbortController();
    setReadinessLoading(true);
    setReadinessError(null);

    apiRequest<ProjectReadiness>(`/projects/${selectedProject.id}/readiness`, {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => {
        if (!controller.signal.aborted) setReadiness(payload);
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setReadiness(null);
          setReadinessError(extractError(error));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setReadinessLoading(false);
      });

    return () => controller.abort();
  }, [selectedProject, selectedProjectId, token]);

  useEffect(() => {
    if (!selectedProject || selectedProjectId === "__new__") {
      setPerformance(null);
      setPerformanceError(null);
      setPerformanceLoading(false);
      return;
    }
    if (activeTab !== "performance") return;

    const controller = new AbortController();
    setPerformanceLoading(true);
    setPerformanceError(null);

    apiRequest<ProjectPerformanceFeedback>(`/projects/${selectedProject.id}/performance`, {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => {
        if (!controller.signal.aborted) setPerformance(payload);
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setPerformance(null);
          setPerformanceError(extractError(error));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setPerformanceLoading(false);
      });

    return () => controller.abort();
  }, [activeTab, selectedProject, selectedProjectId, token]);

  const refreshReadiness = async () => {
    if (!selectedProject) return;
    setReadinessLoading(true);
    setReadinessError(null);
    try {
      const payload = await apiRequest<ProjectReadiness>(`/projects/${selectedProject.id}/readiness`, {
        token,
        timeoutMs: 10000,
      });
      setReadiness(payload);
    } catch (error) {
      setReadiness(null);
      setReadinessError(extractError(error));
    } finally {
      setReadinessLoading(false);
    }
  };

  const refreshPerformance = async () => {
    if (!selectedProject) return;
    setPerformanceLoading(true);
    setPerformanceError(null);
    try {
      const payload = await apiRequest<ProjectPerformanceFeedback>(`/projects/${selectedProject.id}/performance`, {
        token,
        timeoutMs: 10000,
      });
      setPerformance(payload);
    } catch (error) {
      setPerformance(null);
      setPerformanceError(extractError(error));
    } finally {
      setPerformanceLoading(false);
    }
  };

  const importPerformanceCsv = async () => {
    if (!selectedProject || performanceImporting) return;
    if (!performanceCsv.trim()) {
      showToast("error", performanceCopy.importEmpty);
      return;
    }
    setPerformanceImporting(true);
    try {
      await apiRequest<PerformanceImportResponse, { csv_text: string; source: "manual_csv" }>(
        `/projects/${selectedProject.id}/performance/import-csv`,
        {
          method: "POST",
          token,
          body: { csv_text: performanceCsv, source: "manual_csv" },
          timeoutMs: 15000,
        }
      );
      showToast("success", performanceCopy.importSuccess);
      setPerformanceCsv("");
      setPerformanceImportOpen(false);
      await refreshPerformance();
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setPerformanceImporting(false);
    }
  };

  const dismissOpportunity = async (opportunityId: string) => {
    if (!selectedProject || !canManageProjects || dismissingOpportunityId) return;
    setDismissingOpportunityId(opportunityId);
    try {
      await apiRequest(`/projects/${selectedProject.id}/performance/opportunities/${opportunityId}/dismiss`, {
        method: "POST",
        token,
        timeoutMs: 10000,
      });
      await refreshPerformance();
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setDismissingOpportunityId(null);
    }
  };

  const verticalOptions: SelectOption[] = useMemo(() => {
    const base = VERTICAL_OPTIONS.map((v) => ({
      value: v.value,
      label: locale === "fa" ? v.fa : locale === "ar" ? v.ar : v.en,
    }));
    base.push({ value: "__custom__", label: t("projects.customVertical") });
    return base;
  }, [locale, t]);

  const resolvedVertical = newProject.vertical === "__custom__"
    ? newProject.customVertical.trim()
    : VERTICAL_OPTIONS.find((v) => v.value === newProject.vertical)?.en ?? newProject.vertical;

  const onCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setCreating(true);
    try {
      const res = await apiRequest<Project, Record<string, string>>("/projects", {
        method: "POST", token,
        body: { name: newProject.name.trim(), domain: newProject.domain.trim(), vertical: resolvedVertical, description: newProject.description.trim() },
      });
      showToast("success", t("toast.projectCreated"));
      setNewProject({ name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "" });
      await onProjectsRefresh();
      onSelectProject(res.id); // Auto-select new project
    } catch (e) {
      showToast("error", extractError(e));
    } finally { setCreating(false); }
  };

  const onDelete = async (projectId: string) => {
    if (!canManageProjects || deletingProjectId) return;
    setDeletingProjectId(projectId);
    try {
      await apiRequest<void>(`/projects/${projectId}`, { method: "DELETE", token }, { cascade: true });
      showToast("success", t("toast.projectDeleted"));
      setDeleteConfirmId(null);
      if (selectedProjectId === projectId) onSelectProject(null);
      await onProjectsRefresh();
    } catch (e) {
      showToast("error", extractError(e));
    } finally {
      setDeletingProjectId(null);
    }
  };

  const domainPattern = /^[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$/;
  const domainValid = newProject.domain.length > 0 ? domainPattern.test(newProject.domain) : null;

  /* ═══════════════════════════════════════════════════════════════
     STATE A: EMPTY (0 Projects)
     ═══════════════════════════════════════════════════════════════ */
  if (projects.length === 0) {
    return (
      <section className="animate-fade-in flex min-h-[calc(100vh-96px)] items-center justify-center p-4">
        <div className="relative w-full max-w-lg overflow-hidden rounded-xl border border-black/5 bg-white p-8 dark:border-white/10 dark:bg-surface">
          {/* Subtle gradient orb for Apple feel */}
          <div className="pointer-events-none absolute inset-inline-start-0 top-0 h-1 w-full bg-teal-600/70" />

          <div className="text-center mb-8 relative z-10">
            <FolderIllustration />
            <h2 className="mb-2 text-[18px] font-semibold tracking-tight text-slate-900 dark:text-gray-100">{t("projects.emptyTitle")}</h2>
            <p className="text-[14px] text-slate-500 dark:text-gray-400">{t("projects.emptySubtitle")}</p>
          </div>

          {canManageProjects ? (
          <form className="space-y-6 relative z-10" onSubmit={onCreate}>
            <div className="space-y-4">
              <InputField
                label={t("projects.projectName")}
                required
                helperText={t("projects.projectNameHelper")}
                value={newProject.name}
                onChange={(e) => setNewProject((p) => ({ ...p, name: e.target.value }))}
              />
              <InputField
                label={t("projects.domain")}
                helperText={t("projects.domainHelper")}
                successText={domainValid === true ? t("projects.domainValid") : undefined}
                errorText={domainValid === false ? t("projects.domainInvalid") : undefined}
                value={newProject.domain}
                onChange={(e) => setNewProject((p) => ({ ...p, domain: e.target.value }))}
                dir="ltr"
              />
              <SelectDropdown
                label={t("projects.industry")}
                options={verticalOptions}
                value={newProject.vertical}
                onChange={(v) => setNewProject((p) => ({ ...p, vertical: v }))}
              />
              {newProject.vertical === "__custom__" && (
                <InputField
                  label={t("projects.customVertical")}
                  required
                  value={newProject.customVertical}
                  onChange={(e) => setNewProject((p) => ({ ...p, customVertical: e.target.value }))}
                />
              )}
              <div className="flex flex-col gap-[6px]">
                <label className="text-[13px] font-semibold text-slate-700 dark:text-gray-200">{t("projects.description")}</label>
                <textarea
                  placeholder={t("projects.descriptionPlaceholder")}
                  className="min-h-[100px] w-full resize-none rounded-xl border border-black/5 bg-white px-3 py-2 text-[14px] text-slate-900 outline-none transition-colors duration-150 placeholder:text-slate-400 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 dark:border-white/10 dark:bg-surface-alt dark:text-gray-100 dark:placeholder:text-gray-400"
                  value={newProject.description}
                  onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
                />
              </div>
            </div>

            <Button type="submit" variant="primary" loading={creating} fullWidth size="lg">
              {t("projects.createProject")}
            </Button>
          </form>
          ) : (
            <p className="relative z-10 rounded-lg border border-black/5 bg-slate-50 px-4 py-3 text-center text-[13px] text-slate-600 dark:border-white/10 dark:bg-white/[0.04] dark:text-gray-300">
              {t("toast.accessDenied")}
            </p>
          )}
        </div>
      </section>
    );
  }

  /* ═══════════════════════════════════════════════════════════════
     STATE B: MASTER-DETAIL (1+ Projects)
     ═══════════════════════════════════════════════════════════════ */
  return (
    <section className="macos-content-scope animate-fade-in grid min-h-full min-w-0 items-start gap-3 lg:grid-cols-[280px_minmax(0,1fr)]">

      {/* Delete confirmation modal */}
      <Modal
        open={Boolean(deleteConfirmId)}
        onClose={() => {
          if (!deletingProjectId) setDeleteConfirmId(null);
        }}
        title={t("projects.confirmDelete")}
        footer={
          <>
            <Button
              variant="outlined"
              disabled={Boolean(deletingProjectId)}
              onClick={() => setDeleteConfirmId(null)}
            >
              {t("common.cancel")}
            </Button>
            <Button
              variant="danger"
              loading={Boolean(deletingProjectId)}
              onClick={() => deleteConfirmId && void onDelete(deleteConfirmId)}
            >
              {t("common.delete")}
            </Button>
          </>
        }
      >
        <p className="text-[14px] text-slate-600 dark:text-gray-300 leading-relaxed">{t("projects.confirmDeleteMsg")}</p>
      </Modal>

      <Modal
        open={performanceImportOpen}
        onClose={() => {
          if (!performanceImporting) setPerformanceImportOpen(false);
        }}
        title={performanceCopy.importTitle}
        maxWidth="42rem"
        footer={
          <>
            <Button
              variant="outlined"
              disabled={performanceImporting}
              onClick={() => setPerformanceImportOpen(false)}
            >
              {t("common.cancel")}
            </Button>
            <Button
              variant="primary"
              loading={performanceImporting}
              disabled={!performanceCsv.trim()}
              onClick={() => void importPerformanceCsv()}
            >
              {performanceCopy.import}
            </Button>
          </>
        }
      >
        <div className="space-y-3">
          <p className="text-[13px] leading-5 text-slate-600 dark:text-gray-300">
            {performanceCopy.importSubtitle}
          </p>
          <code className="block overflow-x-auto rounded-md bg-black/[0.04] px-3 py-2 text-[12px] text-slate-700 dark:bg-white/[0.06] dark:text-gray-200" dir="ltr">
            {performanceCopy.importColumns}
          </code>
          <textarea
            className="min-h-[220px] w-full resize-y rounded-xl border border-black/5 bg-white px-3 py-3 font-mono text-[13px] leading-5 text-slate-900 outline-none transition-colors duration-150 placeholder:text-slate-400 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 dark:border-white/10 dark:bg-surface-alt dark:text-gray-100 dark:placeholder:text-gray-500"
            placeholder={performanceCopy.importPlaceholder}
            value={performanceCsv}
            onChange={(event) => setPerformanceCsv(event.target.value)}
            dir="ltr"
            spellCheck={false}
          />
        </div>
      </Modal>

      {/* ── LEFT COLUMN (MASTER: macOS style sidebar list) ── */}
      <aside className="relative z-10 flex max-h-[280px] min-h-[220px] min-w-0 flex-col overflow-hidden rounded-xl border border-black/5 bg-slate-100/80 dark:border-white/10 dark:bg-surface-alt lg:sticky lg:top-0 lg:max-h-[calc(100dvh-112px)]">
        <header className="flex h-12 shrink-0 items-center justify-between border-block-end border-black/5 bg-white/60 px-4 dark:border-white/10 dark:bg-white/[0.03]">
          <h2 className="text-[15px] font-semibold tracking-tight text-slate-900 dark:text-gray-100">{t("projects.title")}</h2>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => void onProjectsRefresh()}
              className="h-8 w-8 flex items-center justify-center rounded-md text-slate-400 hover:bg-black/5 hover:text-slate-700 dark:text-gray-500 dark:hover:bg-white/10 dark:hover:text-gray-200 transition-all duration-200"
              title={t("common.refresh")}
            >
              <svg className="w-[15px] h-[15px]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
            {canManageProjects && (
              <>
                <div className="w-[1px] h-4 bg-black/5 dark:bg-white/10 mx-0.5" />
                <button
                  onClick={() => {
                    setNewProject({ name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "" });
                    onSelectProject("__new__");
                  }}
                  className="h-8 w-8 flex items-center justify-center rounded-md text-teal-600 bg-teal-500/10 hover:bg-teal-500/20 transition-all duration-200"
                  title={t("projects.createNew")}
                >
                  <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                </button>
              </>
            )}
          </div>
        </header>

        {/* The Seamless List */}
        <div className="flex-1 overflow-y-auto py-3">
          {projects.map((project) => (
            <button
              key={project.id}
              onClick={() => onSelectProject(project.id)}
              className={clsx(
                "w-full text-start px-6 py-3 transition-all duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] relative group focus:outline-none",
                selectedProjectId === project.id
                  ? "bg-teal-50/50 shadow-[inset_3px_0_0_0_var(--tw-shadow-color)] shadow-teal-600 dark:bg-white/10 dark:shadow-teal-300"
                  : "bg-transparent hover:bg-slate-900/5 focus:bg-slate-900/5 dark:hover:bg-white/[0.08] dark:focus:bg-white/[0.08]" // Native feel hover
              )}
            >
              <div className="flex items-center justify-between gap-2 mb-0.5">
                <span className={clsx("truncate text-[14px]", selectedProjectId === project.id ? "font-bold text-teal-900 dark:text-teal-200" : "font-medium text-slate-900 group-hover:text-black dark:text-gray-200 dark:group-hover:text-white")}>
                  {project.name}
                </span>
                {project.wordpress_url && (
                  <span className={clsx("shrink-0 rounded-[4px] px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider", selectedProjectId === project.id ? "bg-teal-100 text-teal-700 dark:bg-teal-400/15 dark:text-teal-200" : "bg-slate-100 text-slate-500 dark:bg-white/10 dark:text-gray-400")}>WP</span>
                )}
              </div>
              <span className={clsx("truncate block text-[12px]", selectedProjectId === project.id ? "text-teal-700/80 font-medium dark:text-teal-200/80" : "text-slate-500 dark:text-gray-400")} dir="ltr">
                {project.domain || t("projects.noDomain")}
              </span>
            </button>
          ))}
        </div>
      </aside>

      {/* ── RIGHT COLUMN (DETAIL) ── */}
      <main className="min-w-0 overflow-hidden rounded-xl border border-black/5 bg-white dark:border-white/10 dark:bg-surface">

        {selectedProjectId === "__new__" ? (
          // Create Mode
          <div className="p-6 md:p-8">
            <div className="max-w-xl">
              <h3 className="mb-6 text-[18px] font-semibold tracking-tight text-slate-900 dark:text-gray-100">{t("projects.createNew")}</h3>
              <form className="space-y-6" onSubmit={onCreate}>
                <div className="space-y-4">
                  <InputField
                    label={t("projects.projectName")}
                    required
                    helperText={t("projects.projectNameHelper")}
                    value={newProject.name}
                    onChange={(e) => setNewProject((p) => ({ ...p, name: e.target.value }))}
                  />
                  <InputField
                    label={t("projects.domain")}
                    helperText={t("projects.domainHelper")}
                    successText={domainValid === true ? t("projects.domainValid") : undefined}
                    errorText={domainValid === false ? t("projects.domainInvalid") : undefined}
                    value={newProject.domain}
                    onChange={(e) => setNewProject((p) => ({ ...p, domain: e.target.value }))}
                    dir="ltr"
                  />
                  <SelectDropdown
                    label={t("projects.industry")}
                    options={verticalOptions}
                    value={newProject.vertical}
                    onChange={(v) => setNewProject((p) => ({ ...p, vertical: v }))}
                  />
                  {newProject.vertical === "__custom__" && (
                    <InputField
                      label={t("projects.customVertical")}
                      required
                      value={newProject.customVertical}
                      onChange={(e) => setNewProject((p) => ({ ...p, customVertical: e.target.value }))}
                    />
                  )}
                  <div className="flex flex-col gap-[6px]">
                    <label className="text-[13px] font-semibold text-slate-700 dark:text-gray-200">{t("projects.description")}</label>
                    <textarea
                      placeholder={t("projects.descriptionPlaceholder")}
                      className="min-h-[100px] w-full resize-none rounded-xl border border-black/5 bg-white px-3 py-2 text-[14px] text-slate-900 outline-none transition-colors duration-150 placeholder:text-slate-400 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 dark:border-white/10 dark:bg-surface-alt dark:text-gray-100 dark:placeholder:text-gray-400"
                      value={newProject.description}
                      onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
                    />
                  </div>
                </div>

                <div className="flex flex-row gap-3 pt-6 border-block-start border-black/5 dark:border-white/10">
                  <Button type="button" variant="outlined" onClick={() => onSelectProject(projects[0]?.id || null)} size="lg">
                    {t("common.cancel")}
                  </Button>
                  <Button type="submit" variant="primary" loading={creating} size="lg" className="min-w-[140px]">
                    {t("projects.createProject")}
                  </Button>
                </div>
              </form>
            </div>
          </div>
        ) : selectedProject ? (
          // View/Edit Mode
          <>
            <header className="flex flex-col border-block-end border-black/5 dark:border-white/10 shrink-0">
              <div className="flex min-w-0 items-start justify-between gap-4 px-6 pb-4 pt-6">
                <div className="min-w-0 flex-1">
                  <h2 className="mb-1.5 truncate text-[18px] font-semibold leading-none tracking-tight text-slate-900 dark:text-gray-100">{selectedProject.name}</h2>
                  <p className="truncate text-[13px] font-medium text-slate-500 dark:text-gray-400" dir="ltr">{selectedProject.domain || ""}</p>
                </div>

                {/* ── Polished Kebab Kenu ── */}
                {canManageProjects && <div className="relative shrink-0" ref={kebabRef}>
                  <button
                    onClick={() => setKebabOpen(!kebabOpen)}
                    className={clsx(
                      "flex items-center justify-center w-8 h-8 rounded-md transition-all duration-200",
                      kebabOpen ? "bg-slate-100 text-slate-900 dark:bg-white/10 dark:text-gray-100" : "text-slate-400 hover:text-slate-700 hover:bg-slate-50 dark:text-gray-500 dark:hover:text-gray-200 dark:hover:bg-white/10"
                    )}
                    aria-label="More options"
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" /></svg>
                  </button>
                  {kebabOpen && (
                    <div className="absolute top-full inset-inline-end-0 z-50 mt-1 w-48 origin-top-right animate-fade-in rounded-xl border border-black/5 bg-white py-1 dark:border-white/10 dark:bg-surface-alt">
                      <button
                        onClick={() => { setKebabOpen(false); setDeleteConfirmId(selectedProject.id); }}
                        className="w-full text-start px-4 py-2 text-[13px] font-medium text-red-600 hover:bg-red-50 dark:text-red-300 dark:hover:bg-red-500/10 flex items-center gap-2 transition-colors duration-fast"
                      >
                        <svg className="w-[14px] h-[14px]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                        {t("common.delete")}
                      </button>
                    </div>
                  )}
                </div>}
              </div>

              <div className="overflow-x-auto px-6">
                <div className="flex min-w-max border-b border-black/5 dark:border-white/10">
                  {[
                    { id: "readiness", label: readinessCopy.tab },
                    { id: "general", label: t("projects.tabGeneral") },
                    { id: "wordpress", label: t("projects.tabWordpress") },
                    { id: "performance", label: performanceCopy.tab },
                    { id: "rules", label: t("projects.tabRules") },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id as ProjectTab)}
                      className={clsx(
                        "min-h-10 border-b-2 px-4 py-2 text-center text-[12px] font-semibold leading-4 transition-colors duration-150",
                        activeTab === tab.id
                          ? "border-teal-500 text-teal-700 dark:border-teal-300 dark:text-teal-200"
                          : "border-transparent text-slate-500 hover:border-slate-300 hover:text-slate-800 dark:text-gray-400 dark:hover:border-white/20 dark:hover:text-gray-100"
                      )}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              </div>
            </header>

            <div className="relative min-w-0 p-6">
              {activeTab === "readiness" && (
                <ReadinessTab
                  copy={readinessCopy}
                  locale={locale}
                  readiness={readiness}
                  loading={readinessLoading}
                  error={readinessError}
                  onRefresh={() => void refreshReadiness()}
                  onOpenRulebook={() => setActiveTab("rules")}
                  onOpenWordPress={() => setActiveTab("wordpress")}
                />
              )}
              {activeTab === "general" && (
                <GeneralTab
                  token={token}
                  project={selectedProject}
                  canManageProjects={canManageProjects}
                  verticalOptions={verticalOptions}
                  onProjectsRefresh={onProjectsRefresh}
                />
              )}
              {activeTab === "wordpress" && (
                <WordPressTab
                  token={token}
                  project={selectedProject}
                  canManageProjects={canManageProjects}
                  onProjectsRefresh={onProjectsRefresh}
                />
              )}
              {activeTab === "performance" && (
                <PerformanceTab
                  copy={performanceCopy}
                  locale={locale}
                  canManageProjects={canManageProjects}
                  feedback={performance}
                  loading={performanceLoading}
                  error={performanceError}
                  dismissingOpportunityId={dismissingOpportunityId}
                  onRefresh={() => void refreshPerformance()}
                  onOpenImport={() => setPerformanceImportOpen(true)}
                  onDismiss={(opportunityId) => void dismissOpportunity(opportunityId)}
                />
              )}
              {activeTab === "rules" && (
                <RulebookTab
                  token={token}
                  project={selectedProject}
                  canManageProjects={canManageProjects}
                />
              )}
            </div>
          </>
        ) : null}
      </main>

    </section>
  );
}

/* ═══════════════════════════════════════════════════════════════
   TAB COMPONENTS — Form Containment and Spatial Grouping
   ═══════════════════════════════════════════════════════════════ */

function readinessStatusClasses(status: string) {
  if (status === "ready" || status === "pass") {
    return "border-emerald-500/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
  }
  if (status === "blocked" || status === "fail") {
    return "border-red-500/20 bg-red-500/10 text-red-700 dark:text-red-300";
  }
  return "border-amber-500/20 bg-amber-500/10 text-amber-700 dark:text-amber-300";
}

function readinessDotClasses(status: string) {
  if (status === "ready" || status === "pass") return "bg-emerald-500";
  if (status === "blocked" || status === "fail") return "bg-red-500";
  return "bg-amber-400";
}

function ReadinessTab({
  copy,
  locale,
  readiness,
  loading,
  error,
  onRefresh,
  onOpenRulebook,
  onOpenWordPress,
}: {
  copy: typeof READINESS_COPY.en;
  locale: ReadinessLocale;
  readiness: ProjectReadiness | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
  onOpenRulebook: () => void;
  onOpenWordPress: () => void;
}) {
  const wordpressOnlyBlocking = !!readiness
    && readiness.blocking_items.length > 0
    && readiness.blocking_items.every((item) => readinessItemKind(item.id, item.label) === "wordpressPublishing");
  const displayStatus = wordpressOnlyBlocking ? "warning" : readiness?.status;
  const canGenerateForDisplay = !!readiness && (readiness.can_generate || wordpressOnlyBlocking);
  const generationBlocker = readiness?.blocking_items.find(
    (item) => readinessItemKind(item.id, item.label) !== "wordpressPublishing"
  );
  const publishingBlocker = readiness?.blocking_items.find(
    (item) => readinessItemKind(item.id, item.label) === "wordpressPublishing"
  );
  const statusLabel =
    displayStatus === "ready"
      ? copy.ready
      : displayStatus === "blocked"
        ? copy.blocked
        : copy.warning;

  return (
    <div className="max-w-4xl space-y-4 animate-fade-in">
      <section className="rounded-xl border border-black/5 bg-white p-5 dark:border-white/10 dark:bg-surface-alt">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-[12px] font-medium text-slate-500 dark:text-gray-400">{copy.subtitle}</p>
            <h3 className="mt-1 text-[18px] font-semibold tracking-tight text-slate-900 dark:text-gray-100">
              {copy.title}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            {readiness && (
              <span className={clsx("inline-flex h-8 items-center gap-2 rounded-lg border px-3 text-[12px] font-semibold", readinessStatusClasses(displayStatus ?? "warning"))}>
                <span className={clsx("h-2 w-2 rounded-full", readinessDotClasses(displayStatus ?? "warning"))} aria-hidden />
                {statusLabel}
              </span>
            )}
            <Button variant="outlined" size="sm" loading={loading} onClick={onRefresh}>
              {copy.refresh}
            </Button>
          </div>
        </div>

        {loading && !readiness && (
          <div className="mt-5 rounded-lg border border-black/5 bg-slate-50 px-4 py-3 text-[13px] font-medium text-slate-600 dark:border-white/10 dark:bg-white/[0.04] dark:text-gray-300">
            {copy.loading}
          </div>
        )}

        {error && (
          <div className="mt-5 rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 text-[13px] font-medium text-red-700 dark:text-red-300" role="alert">
            {copy.failed} {localizeProjectError(error, locale)}
          </div>
        )}

        {readiness && (
          <>
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <div
                className={clsx(
                  "rounded-xl border p-4",
                  canGenerateForDisplay
                    ? "border-emerald-500/20 bg-emerald-500/[0.06]"
                    : "border-red-500/20 bg-red-500/[0.06]"
                )}
              >
                <p className="text-[12px] font-medium text-slate-500 dark:text-gray-400">{copy.canGenerate}</p>
                <p className="mt-2 text-[16px] font-semibold text-slate-900 dark:text-gray-100">
                  {canGenerateForDisplay ? copy.available : copy.unavailable}
                </p>
                {!canGenerateForDisplay && generationBlocker && (
                  <p className="mt-2 text-[12px] leading-5 text-slate-500 dark:text-gray-400">
                    {localizeReadinessText(generationBlocker.message, locale)}
                  </p>
                )}
              </div>
              <div
                className={clsx(
                  "rounded-xl border p-4",
                  readiness.can_publish
                    ? "border-emerald-500/20 bg-emerald-500/[0.06]"
                    : "border-amber-500/20 bg-amber-500/[0.06]"
                )}
              >
                <p className="text-[12px] font-medium text-slate-500 dark:text-gray-400">{copy.canPublish}</p>
                <p className="mt-2 text-[16px] font-semibold text-slate-900 dark:text-gray-100">
                  {readiness.can_publish ? copy.available : copy.unavailable}
                </p>
                {!readiness.can_publish && publishingBlocker && (
                  <p className="mt-2 text-[12px] leading-5 text-slate-500 dark:text-gray-400">
                    {localizeReadinessText(publishingBlocker.message, locale)}
                  </p>
                )}
              </div>
            </div>
            <p className="mt-4 text-[12px] text-slate-500 dark:text-gray-400">
              {copy.lastChecked}: {formatReadinessDate(readiness.last_checked_at, locale)}
            </p>
          </>
        )}
      </section>

      {readiness && (
        <section className="grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
          <div className="min-w-0 rounded-xl border border-black/5 bg-white dark:border-white/10 dark:bg-surface-alt">
            <div className="border-b border-black/5 px-4 py-3 dark:border-white/10">
              <h4 className="text-[14px] font-semibold text-slate-900 dark:text-gray-100">{copy.allChecks}</h4>
            </div>
            <div className="divide-y divide-black/5 dark:divide-white/10">
              {readiness.checks.map((check) => (
                <div key={check.id} className="grid gap-3 px-4 py-3 sm:grid-cols-[160px_minmax(0,1fr)]">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className={clsx("h-2 w-2 shrink-0 rounded-full", readinessDotClasses(check.status))} aria-hidden />
                    <span className="truncate text-[13px] font-semibold text-slate-900 dark:text-gray-100">
                      {localizeReadinessLabel(check.id, check.label, locale)}
                    </span>
                  </div>
                  <div className="min-w-0">
                    <p className="text-[13px] leading-5 text-slate-600 dark:text-gray-300">
                      {localizeReadinessText(check.message, locale)}
                    </p>
                    {check.remediation && (
                      <p className="mt-1 text-[12px] leading-5 text-slate-500 dark:text-gray-400">
                        {localizeReadinessText(check.remediation, locale)}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <aside className="min-w-0 space-y-4">
            <div className="rounded-xl border border-black/5 bg-white p-4 dark:border-white/10 dark:bg-surface-alt">
              <h4 className="text-[13px] font-semibold text-slate-900 dark:text-gray-100">{copy.blockers}</h4>
              <p className="mt-2 text-[20px] font-semibold tabular-nums text-slate-900 dark:text-gray-100">
                {readiness.blocking_items.length}
              </p>
            </div>
            <div className="rounded-xl border border-black/5 bg-white p-4 dark:border-white/10 dark:bg-surface-alt">
              <h4 className="text-[13px] font-semibold text-slate-900 dark:text-gray-100">{copy.warnings}</h4>
              <p className="mt-2 text-[20px] font-semibold tabular-nums text-slate-900 dark:text-gray-100">
                {readiness.warnings.length}
              </p>
            </div>
            <div className="rounded-xl border border-black/5 bg-white p-4 dark:border-white/10 dark:bg-surface-alt">
              <h4 className="text-[13px] font-semibold text-slate-900 dark:text-gray-100">{copy.actions}</h4>
              <div className="mt-3 space-y-2">
                {readiness.manager_actions.length === 0 ? (
                  <p className="text-[12px] text-slate-500 dark:text-gray-400">{copy.noActions}</p>
                ) : (
                  readiness.manager_actions.map((action) => (
                    <Button
                      key={action.id}
                      variant="outlined"
                      size="sm"
                      fullWidth
                      onClick={() => {
                        if (action.id === "open_rulebook") onOpenRulebook();
                        if (action.id === "test_wordpress_connection") onOpenWordPress();
                      }}
                    >
                      {action.id === "open_rulebook"
                        ? copy.openRulebook
                        : action.id === "test_wordpress_connection"
                          ? copy.openWordPress
                          : action.label}
                    </Button>
                  ))
                )}
              </div>
            </div>
          </aside>
        </section>
      )}
    </div>
  );
}

function performanceSeverityClasses(severity: string) {
  if (severity === "high") {
    return "border-red-500/20 bg-red-500/10 text-red-700 dark:text-red-300";
  }
  if (severity === "medium") {
    return "border-amber-500/20 bg-amber-500/10 text-amber-700 dark:text-amber-300";
  }
  return "border-sky-500/20 bg-sky-500/10 text-sky-700 dark:text-sky-300";
}

function performanceTypeLabel(copy: typeof PERFORMANCE_COPY.en, type: string) {
  const key = type as keyof typeof PERFORMANCE_COPY.en.types;
  return copy.types[key] ?? type.replaceAll("_", " ");
}

function formatCompactNumber(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function formatFixedNumber(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

function formatCtr(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${formatFixedNumber(value * 100, 2)}%`;
}

function formatShortDate(value: string | null | undefined, locale: ReadinessLocale) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  return date.toLocaleDateString(localeName, { month: "short", day: "numeric", year: "numeric" });
}

function metricFromOpportunity(
  opportunity: PerformanceOpportunity,
  key: string,
  fallback?: number | string | null,
) {
  const value = opportunity.supporting_metrics?.[key] ?? fallback;
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
  }
  return null;
}

function PerformanceTab({
  copy,
  locale,
  canManageProjects,
  feedback,
  loading,
  error,
  dismissingOpportunityId,
  onRefresh,
  onOpenImport,
  onDismiss,
}: {
  copy: typeof PERFORMANCE_COPY.en;
  locale: ReadinessLocale;
  canManageProjects: boolean;
  feedback: ProjectPerformanceFeedback | null;
  loading: boolean;
  error: string | null;
  dismissingOpportunityId: string | null;
  onRefresh: () => void;
  onOpenImport: () => void;
  onDismiss: (opportunityId: string) => void;
}) {
  const hasData = Boolean(
    feedback && (feedback.snapshots.length > 0 || feedback.opportunities.length > 0)
  );

  return (
    <div className="max-w-5xl space-y-4 animate-fade-in">
      <section className="rounded-xl border border-black/5 bg-white p-5 dark:border-white/10 dark:bg-surface-alt">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-[12px] font-medium text-slate-500 dark:text-gray-400">{copy.subtitle}</p>
            <h3 className="mt-1 text-[18px] font-semibold tracking-tight text-slate-900 dark:text-gray-100">
              {copy.title}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outlined" size="sm" loading={loading} onClick={onRefresh}>
              {copy.refresh}
            </Button>
            {canManageProjects && (
              <Button variant="primary" size="sm" onClick={onOpenImport}>
                {copy.import}
              </Button>
            )}
          </div>
        </div>

        {loading && !feedback && (
          <div className="mt-5 rounded-lg border border-black/5 bg-slate-50 px-4 py-3 text-[13px] font-medium text-slate-600 dark:border-white/10 dark:bg-white/[0.04] dark:text-gray-300">
            {copy.loading}
          </div>
        )}

        {error && (
          <div className="mt-5 rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 text-[13px] font-medium text-red-700 dark:text-red-300" role="alert">
            {copy.failed} {localizeProjectError(error, locale)}
          </div>
        )}

        {feedback && (
          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <PerformanceSummaryCard label={copy.snapshots} value={feedback.summary.snapshot_count} />
            <PerformanceSummaryCard label={copy.opportunities} value={feedback.summary.opportunity_count} />
            <PerformanceSummaryCard label={copy.highPriority} value={feedback.summary.high_priority_count} tone={feedback.summary.high_priority_count > 0 ? "warning" : "default"} />
            <PerformanceSummaryCard
              label={copy.latestImport}
              value={feedback.summary.latest_imported_at ? formatShortDate(feedback.summary.latest_imported_at, locale) : copy.noImport}
              valueClassName="text-[14px]"
            />
          </div>
        )}
      </section>

      {feedback && !hasData && !loading && (
        <section className="rounded-xl border border-dashed border-black/10 bg-white p-6 text-center dark:border-white/15 dark:bg-surface-alt">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-teal-500/15 bg-teal-500/10 text-teal-700 dark:text-teal-200">
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 19V5m0 14h16M8 16v-5m4 5V8m4 8v-7" />
            </svg>
          </div>
          <h4 className="text-[15px] font-semibold text-slate-900 dark:text-gray-100">{copy.emptyTitle}</h4>
          <p className="mx-auto mt-2 max-w-xl text-[13px] leading-5 text-slate-500 dark:text-gray-400">
            {copy.emptyBody}
          </p>
          {canManageProjects && (
            <Button variant="outlined" size="sm" className="mt-5" onClick={onOpenImport}>
              {copy.import}
            </Button>
          )}
        </section>
      )}

      {feedback && hasData && (
        <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
          <div className="rounded-xl border border-black/5 bg-white dark:border-white/10 dark:bg-surface-alt">
            <div className="border-b border-black/5 px-4 py-3 dark:border-white/10">
              <h4 className="text-[14px] font-semibold text-slate-900 dark:text-gray-100">{copy.opportunitiesTitle}</h4>
            </div>
            {feedback.opportunities.length === 0 ? (
              <p className="px-4 py-6 text-[13px] leading-5 text-slate-500 dark:text-gray-400">{copy.noOpportunities}</p>
            ) : (
              <div className="divide-y divide-black/5 dark:divide-white/10">
                {feedback.opportunities.map((opportunity) => (
                  <PerformanceOpportunityCard
                    key={opportunity.id}
                    copy={copy}
                    opportunity={opportunity}
                    canManageProjects={canManageProjects}
                    dismissing={dismissingOpportunityId === opportunity.id}
                    onDismiss={() => onDismiss(opportunity.id)}
                  />
                ))}
              </div>
            )}
          </div>

          <aside className="rounded-xl border border-black/5 bg-white dark:border-white/10 dark:bg-surface-alt">
            <div className="border-b border-black/5 px-4 py-3 dark:border-white/10">
              <h4 className="text-[14px] font-semibold text-slate-900 dark:text-gray-100">{copy.recentSnapshots}</h4>
            </div>
            {feedback.snapshots.length === 0 ? (
              <p className="px-4 py-5 text-[13px] leading-5 text-slate-500 dark:text-gray-400">{copy.noSnapshots}</p>
            ) : (
              <div className="divide-y divide-black/5 dark:divide-white/10">
                {feedback.snapshots.slice(0, 8).map((snapshot) => (
                  <PerformanceSnapshotRow key={snapshot.id} copy={copy} locale={locale} snapshot={snapshot} />
                ))}
              </div>
            )}
          </aside>
        </section>
      )}
    </div>
  );
}

function PerformanceSummaryCard({
  label,
  value,
  tone = "default",
  valueClassName,
}: {
  label: string;
  value: number | string;
  tone?: "default" | "warning";
  valueClassName?: string;
}) {
  return (
    <div className="rounded-xl border border-black/5 bg-slate-50 p-4 dark:border-white/10 dark:bg-white/[0.04]">
      <p className="text-[12px] font-medium text-slate-500 dark:text-gray-400">{label}</p>
      <p className={clsx(
        "mt-2 truncate font-semibold tabular-nums text-slate-900 dark:text-gray-100",
        typeof value === "number" ? "text-[22px]" : "text-[15px]",
        tone === "warning" && "text-amber-700 dark:text-amber-300",
        valueClassName,
      )}>
        {typeof value === "number" ? formatCompactNumber(value) : value}
      </p>
    </div>
  );
}

function PerformanceOpportunityCard({
  copy,
  opportunity,
  canManageProjects,
  dismissing,
  onDismiss,
}: {
  copy: typeof PERFORMANCE_COPY.en;
  opportunity: PerformanceOpportunity;
  canManageProjects: boolean;
  dismissing: boolean;
  onDismiss: () => void;
}) {
  const clicks = metricFromOpportunity(opportunity, "clicks", metricFromOpportunity(opportunity, "current_clicks"));
  const impressions = metricFromOpportunity(opportunity, "impressions");
  const ctr = metricFromOpportunity(opportunity, "ctr");
  const position = metricFromOpportunity(opportunity, "average_position");
  const previousClicks = metricFromOpportunity(opportunity, "previous_clicks");

  return (
    <article className="px-4 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span className={clsx("inline-flex items-center rounded-lg border px-2.5 py-1 text-[12px] font-semibold", performanceSeverityClasses(opportunity.severity))}>
              {copy.severity[opportunity.severity as keyof typeof copy.severity] ?? opportunity.severity}
            </span>
            <span className="text-[13px] font-semibold text-slate-900 dark:text-gray-100">
              {performanceTypeLabel(copy, opportunity.type)}
            </span>
          </div>
          <p className="text-[13px] leading-5 text-slate-600 dark:text-gray-300">{opportunity.reason}</p>
          <p className="mt-1 text-[13px] leading-5 text-slate-500 dark:text-gray-400">{opportunity.suggested_action}</p>
        </div>
        {canManageProjects && (
          <Button variant="ghost" size="sm" loading={dismissing} onClick={onDismiss}>
            {copy.dismiss}
          </Button>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {opportunity.article_title ? (
          <PerformancePill label={copy.article} value={opportunity.article_title} />
        ) : (
          <PerformancePill label={copy.article} value={copy.unmapped} />
        )}
        {clicks !== null && <PerformancePill label={copy.clicks} value={formatCompactNumber(clicks)} />}
        {previousClicks !== null && <PerformancePill label={copy.previousClicks} value={formatCompactNumber(previousClicks)} />}
        {impressions !== null && <PerformancePill label={copy.impressions} value={formatCompactNumber(impressions)} />}
        {ctr !== null && <PerformancePill label={copy.ctr} value={formatCtr(ctr)} />}
        {position !== null && <PerformancePill label={copy.position} value={formatFixedNumber(position)} />}
      </div>

      <p className="mt-3 truncate text-[12px] text-slate-400 dark:text-gray-500" dir="ltr">
        {opportunity.url}
      </p>
    </article>
  );
}

function PerformancePill({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1 rounded-lg border border-black/5 bg-slate-50 px-2.5 py-1 text-[12px] dark:border-white/10 dark:bg-white/[0.04]">
      <span className="shrink-0 text-slate-400 dark:text-gray-500">{label}</span>
      <span className="min-w-0 truncate font-semibold text-slate-700 dark:text-gray-200">{value}</span>
    </span>
  );
}

function PerformanceSnapshotRow({
  copy,
  locale,
  snapshot,
}: {
  copy: typeof PERFORMANCE_COPY.en;
  locale: ReadinessLocale;
  snapshot: PerformanceSnapshot;
}) {
  return (
    <div className="px-4 py-3">
      <p className="truncate text-[13px] font-semibold text-slate-900 dark:text-gray-100" dir="ltr">
        {snapshot.url}
      </p>
      <p className="mt-1 text-[12px] text-slate-500 dark:text-gray-400">
        {copy.period}: {formatShortDate(snapshot.date_from, locale)} - {formatShortDate(snapshot.date_to, locale)}
      </p>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <PerformanceMiniMetric label={copy.clicks} value={formatCompactNumber(snapshot.clicks)} />
        <PerformanceMiniMetric label={copy.impressions} value={formatCompactNumber(snapshot.impressions)} />
        <PerformanceMiniMetric label={copy.ctr} value={formatCtr(snapshot.ctr)} />
        <PerformanceMiniMetric label={copy.position} value={formatFixedNumber(snapshot.average_position)} />
      </div>
    </div>
  );
}

function PerformanceMiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-black/5 bg-slate-50 px-2.5 py-2 dark:border-white/10 dark:bg-white/[0.04]">
      <p className="text-[11px] font-medium text-slate-400 dark:text-gray-500">{label}</p>
      <p className="mt-1 text-[13px] font-semibold tabular-nums text-slate-900 dark:text-gray-100">{value}</p>
    </div>
  );
}

function projectDraft(project: Project) {
  const preset = VERTICAL_OPTIONS.find(
    (option) => option.value === project.vertical || option.en === project.vertical
  );
  return {
    name: project.name,
    domain: project.domain ?? "",
    description: project.description ?? "",
    vertical: preset?.value ?? (project.vertical ? "__custom__" : VERTICAL_OPTIONS[0].value),
    customVertical: preset ? "" : project.vertical ?? "",
  };
}

function GeneralTab({
  token,
  project,
  canManageProjects,
  verticalOptions,
  onProjectsRefresh,
}: {
  token: string;
  project: Project;
  canManageProjects: boolean;
  verticalOptions: SelectOption[];
  onProjectsRefresh: () => Promise<void>;
}) {
  const { t } = useI18n();
  const [draft, setDraft] = useState(() => projectDraft(project));
  const [saving, setSaving] = useState(false);
  const { showToast } = useToast();

  useEffect(() => {
    setDraft(projectDraft(project));
  }, [project]);

  const initialDraft = projectDraft(project);
  const isDirty = JSON.stringify(draft) !== JSON.stringify(initialDraft);
  const normalizedName = draft.name.trim();
  const normalizedDomain = draft.domain.trim();
  const domainPattern = /^[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$/;
  const domainValid = normalizedDomain.length === 0 || domainPattern.test(
    normalizedDomain.replace(/^https?:\/\//, "").replace(/\/+$/, "")
  );
  const resolvedVertical = draft.vertical === "__custom__"
    ? draft.customVertical.trim()
    : VERTICAL_OPTIONS.find((option) => option.value === draft.vertical)?.en ?? draft.vertical;
  const canSave = canManageProjects
    && isDirty
    && normalizedName.length > 0
    && domainValid
    && resolvedVertical.length > 0;

  const onSave = async () => {
    if (!canSave || saving) return;
    setSaving(true);
    try {
      await apiRequest(`/projects/${project.id}`, {
        method: "PUT", token,
        body: {
          name: normalizedName,
          domain: normalizedDomain,
          description: draft.description.trim(),
          vertical: resolvedVertical,
        },
      });
      showToast("success", t("common.success"));
      await onProjectsRefresh();
    } catch (e) { showToast("error", extractError(e)); }
    finally { setSaving(false); }
  };

  return (
    <div className="max-w-xl space-y-6 animate-fade-in">
      <div className="space-y-4">
        <InputField
          label={t("projects.projectName")}
          required
          value={draft.name}
          disabled={!canManageProjects}
          onChange={(e) => setDraft((p) => ({ ...p, name: e.target.value }))}
        />
        <InputField
          label={t("projects.domain")}
          value={draft.domain}
          disabled={!canManageProjects}
          errorText={!domainValid ? t("projects.domainInvalid") : undefined}
          successText={domainValid && normalizedDomain ? t("projects.domainValid") : undefined}
          onChange={(e) => setDraft((p) => ({ ...p, domain: e.target.value }))}
          dir="ltr"
        />
        <SelectDropdown
          label={t("projects.industry")}
          options={verticalOptions}
          value={draft.vertical}
          disabled={!canManageProjects}
          onChange={(vertical) => setDraft((current) => ({ ...current, vertical }))}
        />
        {draft.vertical === "__custom__" && (
          <InputField
            label={t("projects.customVertical")}
            required
            disabled={!canManageProjects}
            value={draft.customVertical}
            onChange={(event) => setDraft((current) => ({
              ...current,
              customVertical: event.target.value,
            }))}
          />
        )}
        <div className="flex flex-col gap-[6px]">
          <label className="text-[13px] font-semibold text-slate-700 dark:text-gray-200">{t("projects.description")}</label>
          <textarea
            disabled={!canManageProjects}
            className="min-h-[120px] w-full resize-none rounded-xl border border-black/5 bg-white px-3 py-2 text-[14px] text-slate-900 outline-none transition-colors duration-150 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 disabled:cursor-not-allowed disabled:opacity-60 dark:border-white/10 dark:bg-surface-alt dark:text-gray-100"
            value={draft.description}
            onChange={(e) => setDraft((p) => ({ ...p, description: e.target.value }))}
          />
        </div>
      </div>

      {canManageProjects && (
        <div className="flex justify-end pt-2">
          <Button
            variant="primary"
            loading={saving}
            disabled={!canSave}
            onClick={() => void onSave()}
            className="min-w-[120px]"
          >
            {t("common.save")}
          </Button>
        </div>
      )}
    </div>
  );
}

function WordPressTab({
  token,
  project,
  canManageProjects,
  onProjectsRefresh,
}: {
  token: string;
  project: Project;
  canManageProjects: boolean;
  onProjectsRefresh: () => Promise<void>;
}) {
  const { t } = useI18n();
  const { showToast } = useToast();
  const [wpUrl, setWpUrl] = useState("");
  const [wpUsername, setWpUsername] = useState("");
  const [wpPassword, setWpPassword] = useState("");
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setWpUrl(project.wordpress_url ?? "");
    setWpUsername(project.wordpress_username ?? "");
    setWpPassword("");
  }, [project]);

  const save = async () => {
    if (!canManageProjects || saving) return;
    setSaving(true);
    try {
      const payload: Record<string, string> = {
        wordpress_url: wpUrl.trim(), wordpress_username: wpUsername.trim(),
      };
      if (wpPassword.trim()) payload.wordpress_app_password = wpPassword.trim();
      await apiRequest(`/projects/${project.id}`, { method: "PUT", token, body: payload });
      showToast("success", t("toast.wpSaved"));
      await onProjectsRefresh();
    } catch (e) { showToast("error", extractError(e)); }
    finally { setSaving(false); }
  };

  const testConnection = async () => {
    setTesting(true);
    try {
      const payload = await apiRequest<{ connected?: boolean; actionable_message?: string }>(
        `/projects/${project.id}/wordpress/test-connection`, { method: "POST", token }
      );
      if (payload.connected) showToast("success", t("toast.wpTestSuccess"));
      else showToast("error", payload.actionable_message ?? t("toast.wpTestFailed"));
    } catch (e) { showToast("error", extractError(e)); }
    finally { setTesting(false); }
  };

  return (
    <div className="max-w-2xl animate-fade-in">
      <div className="mb-6">
        <h3 className="mb-1 text-[15px] font-bold text-slate-900 dark:text-gray-100">{t("projects.tabWordpress")}</h3>
        <p className="text-[13px] font-medium text-slate-500 dark:text-gray-400 leading-relaxed">{t("projects.wpSubtitle")}</p>
      </div>

      {/* Form Spatial Containment */}
      <div className="space-y-6 rounded-xl border border-black/5 bg-white p-5 dark:border-white/10 dark:bg-surface-alt md:p-6">
        <InputField
          label={t("projects.wpUrl")}
          helperText={t("projects.wpUrlHelper")}
          value={wpUrl}
          disabled={!canManageProjects}
          onChange={(e) => setWpUrl(e.target.value)}
          dir="ltr"
        />
        <div className="grid md:grid-cols-2 gap-6">
          <InputField
            label={t("projects.wpUsername")}
            value={wpUsername}
            disabled={!canManageProjects}
            onChange={(e) => setWpUsername(e.target.value)}
            dir="ltr"
          />
          <InputField
            label={t("projects.wpPassword")}
            type="password"
            helperText={t("projects.wpPasswordTooltip")}
            value={wpPassword}
            disabled={!canManageProjects}
            onChange={(e) => setWpPassword(e.target.value)}
            dir="ltr"
          />
        </div>

        <div className="flex justify-end gap-3 pt-6 border-block-start border-black/5 dark:border-white/10">
          <Button variant="outlined" loading={testing} onClick={() => void testConnection()}>{t("projects.wpTestConnection")}</Button>
          {canManageProjects && (
            <Button variant="primary" loading={saving} onClick={() => void save()} className="min-w-[120px]">
              {t("projects.wpSave")}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function RulebookTab({
  token,
  project,
  canManageProjects,
}: {
  token: string;
  project: Project;
  canManageProjects: boolean;
}) {
  const { t, locale } = useI18n();
  const { showToast } = useToast();
  const [rulebook, setRulebook] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const rulebookPlaceholder = locale === "fa"
    ? "- از لحن رسمی استفاده شود\n- نام رقیب ذکر نشود..."
    : locale === "ar"
      ? "- استخدم نبرة رسمية\n- تجنب ذكر أسماء المنافسين..."
      : "- Use formal tone\n- Avoid competitor names...";

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    apiRequest<RulebookResponse>(`/projects/${project.id}/rulebook`, { token, signal: controller.signal })
      .then(res => {
        if (!controller.signal.aborted) setRulebook(res.content ?? "");
      })
      .catch(() => {
        if (!controller.signal.aborted) setRulebook("");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [project.id, token]);

  const save = async () => {
    if (!canManageProjects || saving || loading || !rulebook.trim()) return;
    setSaving(true);
    try {
      await apiRequest(`/projects/${project.id}/rulebook`, { method: "POST", token, body: { content: rulebook } });
      showToast("success", t("toast.rulebookSaved"));
    } catch (e) { showToast("error", extractError(e)); }
    finally { setSaving(false); }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl animate-fade-in relative pb-16">

      <div className="mb-4 flex items-center justify-between">
        <p className="text-[13px] font-medium text-slate-500 dark:text-gray-400">{t("projects.rulebookEmpty")}</p>
      </div>

      {/* AI-Native Smart Container */}
      <div className="group relative flex min-h-[400px] flex-1 flex-col overflow-hidden rounded-xl border border-black/5 bg-white transition-colors duration-150 focus-within:border-teal-500 focus-within:ring-2 focus-within:ring-teal-500/20 dark:border-white/10 dark:bg-surface-alt dark:focus-within:bg-surface">

          <textarea
            disabled={loading || !canManageProjects}
            className="w-full h-full flex-1 bg-transparent p-6 text-[14px] text-slate-900 dark:text-gray-100 leading-relaxed outline-none border-none resize-y disabled:opacity-50"
            value={rulebook}
            onChange={(e) => setRulebook(e.target.value)}
            placeholder={rulebookPlaceholder}
          />
      </div>

      {/* Primary Action Button (Spatially separated from the textarea) */}
      {canManageProjects && (
        <div className="absolute bottom-0 inset-inline-end-0 flex justify-end">
          <Button
            variant="primary"
            loading={saving || loading}
            disabled={loading || !rulebook.trim()}
            onClick={() => void save()}
            className="min-w-[140px] shadow-sm"
          >
            {t("common.save")}
          </Button>
        </div>
      )}

    </div>
  );
}
