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
  SearchConsoleStatus,
  SeoIntelligenceResponse,
} from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { useToast } from "@/components/ui/toast";
import { InputField } from "@/components/ui/input-field";
import { SelectDropdown } from "@/components/ui/select-dropdown";
import type { SelectOption } from "@/components/ui/select-dropdown";

/* Projects workspace: master list + contextual configuration and operational state. */

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
    tab: "Readiness",
    title: "Project readiness",
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

const SEARCH_CONSOLE_COPY = {
  en: {
    title: "Google Search Console",
    subtitle: "Secure read-only synchronization. Smarlux cannot modify Search Console or publish from this connection.",
    connect: "Connect Google",
    reconnect: "Reconnect",
    disconnect: "Disconnect",
    refreshProperties: "Refresh properties",
    syncNow: "Sync now",
    property: "Search Console property",
    selectProperty: "Select a property",
    connected: "Connected",
    disconnected: "Not connected",
    notConfigured: "OAuth is not configured on the server.",
    loading: "Loading Search Console status…",
    syncing: "Sync queued. Finalized Search Console data will be imported in the background.",
    synced: "Search Console synchronization completed.",
    failed: "Search Console synchronization failed.",
    noProperties: "No accessible properties were returned for this Google account.",
    lastSync: "Last sync",
    never: "Never",
    readOnly: "Read-only",
  },
  fa: {
    title: "Google Search Console",
    subtitle: "همگام‌سازی امن و فقط‌خواندنی؛ این اتصال امکان تغییر سرچ کنسول یا انتشار محتوا را ندارد.",
    connect: "اتصال به گوگل",
    reconnect: "اتصال مجدد",
    disconnect: "قطع اتصال",
    refreshProperties: "بروزرسانی سایت‌ها",
    syncNow: "همگام‌سازی",
    property: "سایت سرچ کنسول",
    selectProperty: "یک سایت انتخاب کنید",
    connected: "متصل",
    disconnected: "متصل نیست",
    notConfigured: "OAuth سرچ کنسول روی سرور تنظیم نشده است.",
    loading: "در حال دریافت وضعیت سرچ کنسول…",
    syncing: "همگام‌سازی در صف قرار گرفت و داده نهایی در پس‌زمینه وارد می‌شود.",
    synced: "همگام‌سازی سرچ کنسول تکمیل شد.",
    failed: "همگام‌سازی سرچ کنسول ناموفق بود.",
    noProperties: "برای این حساب گوگل سایت قابل‌دسترسی پیدا نشد.",
    lastSync: "آخرین همگام‌سازی",
    never: "هرگز",
    readOnly: "فقط‌خواندنی",
  },
  ar: {
    title: "Google Search Console",
    subtitle: "مزامنة آمنة للقراءة فقط؛ لا يمكن لهذا الاتصال تعديل Search Console أو نشر المحتوى.",
    connect: "ربط Google",
    reconnect: "إعادة الربط",
    disconnect: "قطع الاتصال",
    refreshProperties: "تحديث المواقع",
    syncNow: "مزامنة الآن",
    property: "موقع Search Console",
    selectProperty: "اختر موقعًا",
    connected: "متصل",
    disconnected: "غير متصل",
    notConfigured: "لم يتم إعداد OAuth على الخادم.",
    loading: "جارٍ تحميل حالة Search Console…",
    syncing: "تمت إضافة المزامنة إلى الطابور وسيتم استيراد البيانات النهائية في الخلفية.",
    synced: "اكتملت مزامنة Search Console.",
    failed: "فشلت مزامنة Search Console.",
    noProperties: "لم يتم العثور على مواقع متاحة لهذا الحساب.",
    lastSync: "آخر مزامنة",
    never: "أبدًا",
    readOnly: "قراءة فقط",
  },
};

const SEO_INTELLIGENCE_COPY = {
  en: {
    title: "SEO Intelligence",
    subtitle: "Explainable prioritization from first-party performance data",
    health: "Portfolio health",
    coverage: "Measured coverage",
    highPriority: "High priority",
    queue: "Recommended queue",
    dataQuality: "Data quality",
    dataGood: "Good",
    dataLimited: "Limited",
    dataInsufficient: "Insufficient",
    confidence: "confidence",
    noQueue: "No prioritized action is currently available.",
    loading: "Building the prioritized SEO portfolio…",
    failed: "SEO intelligence could not be loaded.",
    safe: "Read-only · no automatic rewrite or publishing",
  },
  fa: {
    title: "هوشمندی سئو",
    subtitle: "اولویت‌بندی توضیح‌پذیر بر پایه داده‌های عملکردی خود پروژه",
    health: "سلامت مجموعه محتوا",
    coverage: "پوشش اندازه‌گیری",
    highPriority: "اولویت بالا",
    queue: "صف پیشنهادی",
    dataQuality: "کیفیت داده",
    dataGood: "مناسب",
    dataLimited: "محدود",
    dataInsufficient: "ناکافی",
    confidence: "اطمینان",
    noQueue: "در حال حاضر اقدام اولویت‌داری وجود ندارد.",
    loading: "در حال ساخت اولویت‌های سئو…",
    failed: "هوشمندی سئو قابل دریافت نیست.",
    safe: "فقط‌خواندنی · بدون بازنویسی یا انتشار خودکار",
  },
  ar: {
    title: "ذكاء تحسين محركات البحث",
    subtitle: "ترتيب أولويات قابل للتفسير اعتمادًا على بيانات الأداء المباشرة",
    health: "صحة المحتوى",
    coverage: "التغطية المقاسة",
    highPriority: "أولوية مرتفعة",
    queue: "قائمة العمل المقترحة",
    dataQuality: "جودة البيانات",
    dataGood: "جيدة",
    dataLimited: "محدودة",
    dataInsufficient: "غير كافية",
    confidence: "الثقة",
    noQueue: "لا يوجد إجراء ذو أولوية حاليًا.",
    loading: "جارٍ بناء أولويات تحسين محركات البحث…",
    failed: "تعذر تحميل ذكاء تحسين محركات البحث.",
    safe: "للقراءة فقط · دون إعادة كتابة أو نشر تلقائي",
  },
};

const SEO_NEXT_ACTION_COPY = {
  en: {
    low_ctr_high_impressions: "Improve the title and meta description after validating search intent.",
    striking_distance_position: "Strengthen the highest-value section and internal links.",
    declining_clicks: "Audit freshness and separate demand decline from page deterioration.",
    unmapped_url: "Map the measured URL to the correct article.",
    missing_performance_data: "Collect a complete Search Console or manual performance window.",
  },
  fa: {
    low_ctr_high_impressions: "پس از بررسی نیت جست‌وجو، عنوان و توضیحات متا را بهبود دهید.",
    striking_distance_position: "بخش باارزش مقاله و پیوندهای داخلی را تقویت کنید.",
    declining_clicks: "تازگی محتوا را بررسی و افت تقاضا را از افت عملکرد صفحه جدا کنید.",
    unmapped_url: "URL اندازه‌گیری‌شده را به مقاله درست متصل کنید.",
    missing_performance_data: "یک بازه کامل از سرچ کنسول یا داده عملکرد دستی جمع‌آوری کنید.",
  },
  ar: {
    low_ctr_high_impressions: "حسّن العنوان والوصف التعريفي بعد التحقق من نية البحث.",
    striking_distance_position: "عزّز القسم الأعلى قيمة والروابط الداخلية.",
    declining_clicks: "راجع حداثة المحتوى وافصل تراجع الطلب عن تدهور الصفحة.",
    unmapped_url: "اربط الرابط المقاس بالمقال الصحيح.",
    missing_performance_data: "اجمع نافذة أداء مكتملة من Search Console أو الاستيراد اليدوي.",
  },
};

const SEO_WARNING_COPY = {
  en: {
    search_console_disconnected: "Search Console is not connected; recommendations may rely on manual or stale data.",
    no_performance_data: "No performance snapshot is available.",
    performance_data_very_stale: "Performance data is too old for a high-confidence recommendation.",
    performance_data_stale: "Performance data should be refreshed.",
    unmapped_urls: "Some measured URLs are not mapped to generated articles.",
    truncated_sync_runs: "Recent Search Console coverage was incomplete.",
    failed_sync_runs: "One or more recent Search Console syncs failed.",
  },
  fa: {
    search_console_disconnected: "سرچ کنسول متصل نیست و پیشنهادها ممکن است بر داده دستی یا قدیمی تکیه کنند.",
    no_performance_data: "هیچ داده عملکردی در دسترس نیست.",
    performance_data_very_stale: "داده عملکردی برای پیشنهاد با اطمینان بالا بیش از حد قدیمی است.",
    performance_data_stale: "داده عملکردی باید بروزرسانی شود.",
    unmapped_urls: "برخی URLهای اندازه‌گیری‌شده به مقاله‌های تولیدشده متصل نیستند.",
    truncated_sync_runs: "پوشش اخیر سرچ کنسول ناقص بوده است.",
    failed_sync_runs: "یک یا چند همگام‌سازی اخیر سرچ کنسول ناموفق بوده است.",
  },
  ar: {
    search_console_disconnected: "Search Console غير متصل وقد تعتمد التوصيات على بيانات يدوية أو قديمة.",
    no_performance_data: "لا تتوفر لقطة لبيانات الأداء.",
    performance_data_very_stale: "بيانات الأداء قديمة جدًا لتوصية عالية الثقة.",
    performance_data_stale: "يجب تحديث بيانات الأداء.",
    unmapped_urls: "بعض الروابط المقاسة غير مرتبطة بالمقالات المنشأة.",
    truncated_sync_runs: "تغطية Search Console الأخيرة غير مكتملة.",
    failed_sync_runs: "فشلت مزامنة واحدة أو أكثر مؤخرًا.",
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
    <div className="mx-auto mb-5 flex h-12 w-12 items-center justify-center text-ink-tertiary">
      <svg viewBox="0 0 48 48" fill="none" className="h-10 w-10" aria-hidden>
        <path d="M5 13a4 4 0 0 1 4-4h10l5 5h15a4 4 0 0 1 4 4v17a4 4 0 0 1-4 4H9a4 4 0 0 1-4-4V13Z" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function DomainHelperText({
  accessibleLabel,
  withoutLabel,
}: {
  accessibleLabel: string;
  withoutLabel: string;
}) {
  return (
    <span dir="ltr" aria-label={accessibleLabel}>
      <bdi dir="ltr">example.com</bdi>
      {" ("}
      <bdi dir="auto">{withoutLabel}</bdi>
      {" "}
      <bdi dir="ltr">https://</bdi>
      {")"}
    </span>
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
  const [seoIntelligence, setSeoIntelligence] = useState<SeoIntelligenceResponse | null>(null);
  const [seoIntelligenceLoading, setSeoIntelligenceLoading] = useState(false);
  const [seoIntelligenceError, setSeoIntelligenceError] = useState<string | null>(null);
  const [performanceImportOpen, setPerformanceImportOpen] = useState(false);
  const [performanceCsv, setPerformanceCsv] = useState("");
  const [performanceImporting, setPerformanceImporting] = useState(false);
  const [dismissingOpportunityId, setDismissingOpportunityId] = useState<string | null>(null);
  const [searchConsole, setSearchConsole] = useState<SearchConsoleStatus | null>(null);
  const [searchConsoleLoading, setSearchConsoleLoading] = useState(false);
  const [searchConsoleAction, setSearchConsoleAction] = useState<string | null>(null);
  const [searchConsoleError, setSearchConsoleError] = useState<string | null>(null);

  // Kebab Menu State
  const [kebabOpen, setKebabOpen] = useState(false);
  const kebabRef = useRef<HTMLDivElement>(null);
  const searchConsoleRefreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useClickOutside(kebabRef, () => setKebabOpen(false));

  useEffect(() => {
    return () => {
      if (searchConsoleRefreshTimerRef.current !== null) {
        globalThis.clearTimeout(searchConsoleRefreshTimerRef.current);
        searchConsoleRefreshTimerRef.current = null;
      }
    };
  }, [selectedProjectId]);

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId]
  );
  const readinessCopy = READINESS_COPY[locale];
  const performanceCopy = PERFORMANCE_COPY[locale];
  const searchConsoleCopy = SEARCH_CONSOLE_COPY[locale];
  const seoIntelligenceCopy = SEO_INTELLIGENCE_COPY[locale];

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

  useEffect(() => {
    if (!selectedProject || selectedProjectId === "__new__") {
      setSeoIntelligence(null);
      setSeoIntelligenceError(null);
      setSeoIntelligenceLoading(false);
      return;
    }
    if (activeTab !== "performance") return;

    const controller = new AbortController();
    setSeoIntelligenceLoading(true);
    setSeoIntelligenceError(null);
    apiRequest<SeoIntelligenceResponse>(`/projects/${selectedProject.id}/seo-intelligence`, {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => {
        if (!controller.signal.aborted) setSeoIntelligence(payload);
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setSeoIntelligence(null);
          setSeoIntelligenceError(extractError(error));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setSeoIntelligenceLoading(false);
      });
    return () => controller.abort();
  }, [activeTab, selectedProject, selectedProjectId, token]);

  useEffect(() => {
    if (!selectedProject || selectedProjectId === "__new__" || activeTab !== "performance") return;
    const controller = new AbortController();
    setSearchConsoleLoading(true);
    setSearchConsoleError(null);
    apiRequest<SearchConsoleStatus>(`/projects/${selectedProject.id}/search-console/status`, {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => { if (!controller.signal.aborted) setSearchConsole(payload); })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setSearchConsole(null);
          setSearchConsoleError(extractError(error));
        }
      })
      .finally(() => { if (!controller.signal.aborted) setSearchConsoleLoading(false); });
    return () => controller.abort();
  }, [activeTab, selectedProject, selectedProjectId, token]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const callbackState = params.get("search_console");
    const callbackProject = params.get("project_id");
    if (!callbackState) return;

    if (callbackProject && callbackProject !== selectedProject?.id) {
      if (projects.some((project) => project.id === callbackProject)) {
        onSelectProject(callbackProject);
        return;
      }
      showToast("error", searchConsoleCopy.failed);
    } else if (!selectedProject) {
      return;
    } else {
      setActiveTab("performance");
      if (callbackState === "connected") showToast("success", searchConsoleCopy.connected);
      if (callbackState === "error") showToast("error", params.get("message") || searchConsoleCopy.failed);
    }

    ["search_console", "project_id", "category", "message"].forEach((key) => params.delete(key));
    const next = `${window.location.pathname}${params.toString() ? `?${params.toString()}` : ""}${window.location.hash}`;
    window.history.replaceState({}, "", next);
  }, [onSelectProject, projects, searchConsoleCopy, selectedProject, showToast]);

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

  const refreshSeoIntelligence = async () => {
    if (!selectedProject) return;
    setSeoIntelligenceLoading(true);
    setSeoIntelligenceError(null);
    try {
      const payload = await apiRequest<SeoIntelligenceResponse>(
        `/projects/${selectedProject.id}/seo-intelligence`,
        { token, timeoutMs: 10000 }
      );
      setSeoIntelligence(payload);
    } catch (error) {
      setSeoIntelligence(null);
      setSeoIntelligenceError(extractError(error));
    } finally {
      setSeoIntelligenceLoading(false);
    }
  };

  const refreshSearchConsole = async () => {
    if (!selectedProject) return;
    setSearchConsoleLoading(true);
    setSearchConsoleError(null);
    try {
      const payload = await apiRequest<SearchConsoleStatus>(`/projects/${selectedProject.id}/search-console/status`, { token, timeoutMs: 10000 });
      setSearchConsole(payload);
    } catch (error) {
      setSearchConsoleError(extractError(error));
    } finally {
      setSearchConsoleLoading(false);
    }
  };

  const connectSearchConsole = async () => {
    if (!selectedProject || searchConsoleAction) return;
    setSearchConsoleAction("connect");
    try {
      const payload = await apiRequest<{ authorization_url: string }>(`/projects/${selectedProject.id}/search-console/connect`, { method: "POST", token, timeoutMs: 10000 });
      window.location.assign(payload.authorization_url);
    } catch (error) {
      showToast("error", extractError(error));
      setSearchConsoleAction(null);
    }
  };

  const refreshSearchConsoleProperties = async () => {
    if (!selectedProject || searchConsoleAction) return;
    setSearchConsoleAction("properties");
    try {
      await apiRequest(`/projects/${selectedProject.id}/search-console/properties/refresh`, { method: "POST", token, timeoutMs: 30000 });
      await refreshSearchConsole();
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setSearchConsoleAction(null);
    }
  };

  const selectSearchConsoleProperty = async (siteUrl: string) => {
    if (!selectedProject || searchConsoleAction) return;
    setSearchConsoleAction("property");
    try {
      await apiRequest(`/projects/${selectedProject.id}/search-console/property`, {
        method: "PUT", token, body: { site_url: siteUrl }, timeoutMs: 15000,
      });
      await refreshSearchConsole();
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setSearchConsoleAction(null);
    }
  };

  const syncSearchConsole = async () => {
    if (!selectedProject || searchConsoleAction) return;
    setSearchConsoleAction("sync");
    try {
      await apiRequest(`/projects/${selectedProject.id}/search-console/sync`, { method: "POST", token, body: {}, timeoutMs: 15000 });
      showToast("success", searchConsoleCopy.syncing);
      await refreshSearchConsole();
      if (searchConsoleRefreshTimerRef.current !== null) {
        globalThis.clearTimeout(searchConsoleRefreshTimerRef.current);
      }
      searchConsoleRefreshTimerRef.current = globalThis.setTimeout(() => {
        searchConsoleRefreshTimerRef.current = null;
        void refreshSearchConsole();
        void refreshPerformance();
        void refreshSeoIntelligence();
      }, 2500);
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setSearchConsoleAction(null);
    }
  };

  const disconnectSearchConsole = async () => {
    if (!selectedProject || searchConsoleAction) return;
    setSearchConsoleAction("disconnect");
    try {
      await apiRequest(`/projects/${selectedProject.id}/search-console/disconnect`, { method: "POST", token, timeoutMs: 15000 });
      await refreshSearchConsole();
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setSearchConsoleAction(null);
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
      await Promise.all([refreshPerformance(), refreshSeoIntelligence()]);
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
      await Promise.all([refreshPerformance(), refreshSeoIntelligence()]);
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
      <section className="smx-page flex min-h-[calc(100dvh-110px)] items-center justify-center">
        <div className="w-full max-w-[520px] py-10">
          <div className="mb-8 text-center">
            <FolderIllustration />
            <h2 className="mb-2 text-xl font-semibold text-ink">{t("projects.emptyTitle")}</h2>
            <p className="text-base leading-[22px] text-ink-secondary">{t("projects.emptySubtitle")}</p>
          </div>

          {canManageProjects ? (
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
                helperText={
                  <DomainHelperText
                    accessibleLabel={t("projects.domainHelper")}
                    withoutLabel={t("projects.domainWithoutProtocol")}
                  />
                }
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
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium text-ink-secondary">{t("projects.description")}</label>
                <textarea
                  aria-label={t("projects.description")}
                  placeholder={t("projects.descriptionPlaceholder")}
                  className="smx-input min-h-[100px] w-full resize-none"
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
            <p className="border-s-2 border-warning bg-warning-subtle px-4 py-3 text-center text-sm text-warning">
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
    <section className="smx-page !max-w-none !py-0 grid min-h-full min-w-0 items-start lg:grid-cols-[240px_minmax(0,1fr)]">

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
        <p className="text-base text-ink-secondary leading-relaxed">{t("projects.confirmDeleteMsg")}</p>
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
          <p className="text-sm leading-5 text-ink-secondary">
            {performanceCopy.importSubtitle}
          </p>
          <code className="block overflow-x-auto rounded-md bg-ink/[0.04] px-3 py-2 text-xs text-ink-secondary" dir="ltr">
            {performanceCopy.importColumns}
          </code>
          <textarea
            aria-label={performanceCopy.importTitle}
            className="min-h-[220px] w-full resize-y rounded-xl border border-line bg-surface px-3 py-3 font-mono text-sm leading-5 text-ink outline-none transition-colors duration-150 placeholder:text-ink-muted focus:border-brand focus:ring-1 focus:ring-brand/20"
            placeholder={performanceCopy.importPlaceholder}
            value={performanceCsv}
            onChange={(event) => setPerformanceCsv(event.target.value)}
            dir="ltr"
            spellCheck={false}
          />
        </div>
      </Modal>

      {/* Project list */}
      <aside className="relative z-10 flex max-h-[280px] min-h-[220px] min-w-0 flex-col overflow-hidden border-e border-line bg-[rgb(var(--bg-secondary)/0.55)] lg:sticky lg:top-0 lg:max-h-[calc(100dvh-96px)]">
        <header className="flex h-[52px] shrink-0 items-center justify-between gap-3 border-b border-line px-4">
          <h2 className="text-base font-semibold text-ink">{t("projects.title")}</h2>
          <div className="flex items-center gap-1.5">
            <button type="button"
              onClick={() => void onProjectsRefresh()}
              className="smx-icon-button !h-8 !w-8"
              aria-label={t("common.refresh")}
              title={t("common.refresh")}
            >
              <svg className="w-[15px] h-[15px]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
            {canManageProjects && (
              <>
                <div className="w-[1px] h-4 bg-ink/[0.06] mx-0.5" />
                <button type="button"
                  onClick={() => {
                    setNewProject({ name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "" });
                    onSelectProject("__new__");
                  }}
                  className="flex h-8 w-8 items-center justify-center rounded-md bg-brand text-white transition-colors hover:bg-brand-hover"
                  aria-label={t("projects.createNew")}
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
        <div className="flex-1 overflow-y-auto py-2">
          {projects.map((project) => (
            <button type="button"
              key={project.id}
              onClick={() => onSelectProject(project.id)}
              className={clsx(
                "group relative w-full px-4 py-3 text-start transition-colors duration-fast",
                selectedProjectId === project.id
                  ? "bg-ink/[0.055]"
                  : "bg-transparent hover:bg-ink/[0.035]"
              )}
            >
              <div className="flex items-center justify-between gap-2 mb-0.5">
                <span className={clsx("truncate text-base", selectedProjectId === project.id ? "font-semibold text-ink" : "font-medium text-ink-secondary group-hover:text-ink")}>
                  {project.name}
                </span>
                {project.wordpress_url && (
                  <span className={clsx("shrink-0 text-xs font-medium", selectedProjectId === project.id ? "text-success" : "text-ink-tertiary")}>WP</span>
                )}
              </div>
              <span className={clsx("truncate block text-xs", selectedProjectId === project.id ? "text-ink-secondary" : "text-ink-tertiary")} dir="ltr">
                {project.domain || t("projects.noDomain")}
              </span>
            </button>
          ))}
        </div>
      </aside>

      {/* ── RIGHT COLUMN (DETAIL) ── */}
      <main className="min-w-0 overflow-hidden">

        {selectedProjectId === "__new__" ? (
          // Create Mode
          <div className="p-6 lg:p-8">
            <div className="max-w-xl">
              <h3 className="mb-6 text-xl font-semibold text-ink">{t("projects.createNew")}</h3>
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
                    helperText={
                  <DomainHelperText
                    accessibleLabel={t("projects.domainHelper")}
                    withoutLabel={t("projects.domainWithoutProtocol")}
                  />
                }
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
                  <div className="flex flex-col gap-1.5">
                    <label className="text-sm font-medium text-ink-secondary">{t("projects.description")}</label>
                    <textarea
                      aria-label={t("projects.description")}
                      placeholder={t("projects.descriptionPlaceholder")}
                      className="smx-input min-h-[100px] w-full resize-none"
                      value={newProject.description}
                      onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
                    />
                  </div>
                </div>

                <div className="flex flex-row gap-2 border-block-start border-line pt-5">
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
            <header className="flex shrink-0 flex-col border-block-end border-line">
              <div className="flex min-w-0 items-start justify-between gap-4 px-6 pb-4 pt-7 lg:px-8">
                <div className="min-w-0 flex-1">
                  <h2 className="mb-1.5 truncate text-xl font-semibold leading-6 text-ink">{selectedProject.name}</h2>
                  <p className="truncate text-sm text-ink-tertiary" dir="ltr">{selectedProject.domain || ""}</p>
                </div>

                {/* Project actions */}
                {canManageProjects && <div className="relative shrink-0" ref={kebabRef}>
                  <button type="button"
                    onClick={() => setKebabOpen(!kebabOpen)}
                    className={clsx(
                      "flex items-center justify-center w-8 h-8 rounded-md transition-all duration-200",
                      kebabOpen ? "bg-ink/[0.055] text-ink" : "text-ink-muted hover:bg-ink/[0.045] hover:text-ink"
                    )}
                    aria-label={t("common.moreOptions")}
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" /></svg>
                  </button>
                  {kebabOpen && (
                    <div className="absolute top-full inset-inline-end-0 z-50 mt-1 w-48 origin-top-right animate-fade-in rounded-xl border border-line bg-surface py-1">
                      <button type="button"
                        onClick={() => { setKebabOpen(false); setDeleteConfirmId(selectedProject.id); }}
                        className="w-full text-start px-4 py-2 text-sm font-medium text-danger hover:bg-danger-subtle flex items-center gap-2 transition-colors duration-fast"
                      >
                        <svg className="w-[14px] h-[14px]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                        {t("common.delete")}
                      </button>
                    </div>
                  )}
                </div>}
              </div>

              <div className="overflow-x-auto px-6 lg:px-8">
                <div className="flex min-w-max gap-5">
                  {[
                    { id: "readiness", label: readinessCopy.tab },
                    { id: "general", label: t("projects.tabGeneral") },
                    { id: "wordpress", label: t("projects.tabWordpress") },
                    { id: "performance", label: performanceCopy.tab },
                    { id: "rules", label: t("projects.tabRules") },
                  ].map((tab) => (
                    <button type="button"
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id as ProjectTab)}
                      className={clsx(
                        "min-h-10 border-b-2 px-0.5 py-2 text-center text-sm font-medium leading-5 transition-colors duration-fast",
                        activeTab === tab.id
                          ? "border-brand text-ink"
                          : "border-transparent text-ink-tertiary hover:text-ink"
                      )}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              </div>
            </header>

            <div className="relative min-w-0 p-6 lg:p-8">
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
                  onRefresh={() => {
                    void refreshPerformance();
                    void refreshSeoIntelligence();
                  }}
                  seoIntelligence={seoIntelligence}
                  seoIntelligenceCopy={seoIntelligenceCopy}
                  seoIntelligenceLoading={seoIntelligenceLoading}
                  seoIntelligenceError={seoIntelligenceError}
                  onOpenImport={() => setPerformanceImportOpen(true)}
                  onDismiss={(opportunityId) => void dismissOpportunity(opportunityId)}
                  searchConsole={searchConsole}
                  searchConsoleCopy={searchConsoleCopy}
                  searchConsoleLoading={searchConsoleLoading}
                  searchConsoleAction={searchConsoleAction}
                  searchConsoleError={searchConsoleError}
                  onConnectSearchConsole={() => void connectSearchConsole()}
                  onRefreshSearchConsole={() => void refreshSearchConsole()}
                  onRefreshSearchConsoleProperties={() => void refreshSearchConsoleProperties()}
                  onSelectSearchConsoleProperty={(siteUrl) => void selectSearchConsoleProperty(siteUrl)}
                  onSyncSearchConsole={() => void syncSearchConsole()}
                  onDisconnectSearchConsole={() => void disconnectSearchConsole()}
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
   Project detail sections
   ═══════════════════════════════════════════════════════════════ */

function readinessStatusClasses(status: string) {
  if (status === "ready" || status === "pass") {
    return "border-success/20 bg-success/10 text-success";
  }
  if (status === "blocked" || status === "fail") {
    return "border-danger/20 bg-danger/10 text-danger";
  }
  return "border-warning/20 bg-warning/10 text-warning";
}

function readinessDotClasses(status: string) {
  if (status === "ready" || status === "pass") return "bg-success";
  if (status === "blocked" || status === "fail") return "bg-danger";
  return "bg-warning";
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
      <section className="smx-panel-subtle p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-xs font-medium text-ink-muted">{copy.subtitle}</p>
            <h3 className="mt-1 text-xl font-semibold tracking-tight text-ink">
              {copy.title}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            {readiness && (
              <span className={clsx("inline-flex h-8 items-center gap-2 rounded-lg border px-3 text-xs font-semibold", readinessStatusClasses(displayStatus ?? "warning"))}>
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
          <div className="mt-5 rounded-lg border border-line bg-surface-alt px-4 py-3 text-sm font-medium text-ink-secondary">
            {copy.loading}
          </div>
        )}

        {error && (
          <div className="mt-5 rounded-lg border border-danger/20 bg-danger/10 px-4 py-3 text-sm font-medium text-danger" role="alert">
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
                    ? "border-success/20 bg-success/[0.06]"
                    : "border-danger/20 bg-danger/[0.06]"
                )}
              >
                <p className="text-xs font-medium text-ink-muted">{copy.canGenerate}</p>
                <p className="mt-2 text-lg font-semibold text-ink">
                  {canGenerateForDisplay ? copy.available : copy.unavailable}
                </p>
                {!canGenerateForDisplay && generationBlocker && (
                  <p className="mt-2 text-xs leading-5 text-ink-muted">
                    {localizeReadinessText(generationBlocker.message, locale)}
                  </p>
                )}
              </div>
              <div
                className={clsx(
                  "rounded-xl border p-4",
                  readiness.can_publish
                    ? "border-success/20 bg-success/[0.06]"
                    : "border-warning/20 bg-warning/[0.06]"
                )}
              >
                <p className="text-xs font-medium text-ink-muted">{copy.canPublish}</p>
                <p className="mt-2 text-lg font-semibold text-ink">
                  {readiness.can_publish ? copy.available : copy.unavailable}
                </p>
                {!readiness.can_publish && publishingBlocker && (
                  <p className="mt-2 text-xs leading-5 text-ink-muted">
                    {localizeReadinessText(publishingBlocker.message, locale)}
                  </p>
                )}
              </div>
            </div>
            <p className="mt-4 text-xs text-ink-muted">
              {copy.lastChecked}: {formatReadinessDate(readiness.last_checked_at, locale)}
            </p>
          </>
        )}
      </section>

      {readiness && (
        <section className="grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
          <div className="min-w-0 rounded-xl border border-line bg-surface">
            <div className="border-b border-line px-4 py-3">
              <h4 className="text-base font-semibold text-ink">{copy.allChecks}</h4>
            </div>
            <div className="divide-y divide-line">
              {readiness.checks.map((check) => (
                <div key={check.id} className="grid gap-3 px-4 py-3 sm:grid-cols-[160px_minmax(0,1fr)]">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className={clsx("h-2 w-2 shrink-0 rounded-full", readinessDotClasses(check.status))} aria-hidden />
                    <span className="truncate text-sm font-semibold text-ink">
                      {localizeReadinessLabel(check.id, check.label, locale)}
                    </span>
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm leading-5 text-ink-secondary">
                      {localizeReadinessText(check.message, locale)}
                    </p>
                    {check.remediation && (
                      <p className="mt-1 text-xs leading-5 text-ink-muted">
                        {localizeReadinessText(check.remediation, locale)}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <aside className="min-w-0 space-y-4">
            <div className="rounded-xl border border-line bg-surface p-4">
              <h4 className="text-sm font-semibold text-ink">{copy.blockers}</h4>
              <p className="mt-2 text-metric font-semibold tabular-nums text-ink">
                {readiness.blocking_items.length}
              </p>
            </div>
            <div className="rounded-xl border border-line bg-surface p-4">
              <h4 className="text-sm font-semibold text-ink">{copy.warnings}</h4>
              <p className="mt-2 text-metric font-semibold tabular-nums text-ink">
                {readiness.warnings.length}
              </p>
            </div>
            <div className="rounded-xl border border-line bg-surface p-4">
              <h4 className="text-sm font-semibold text-ink">{copy.actions}</h4>
              <div className="mt-3 space-y-2">
                {readiness.manager_actions.length === 0 ? (
                  <p className="text-xs text-ink-muted">{copy.noActions}</p>
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
    return "border-danger/20 bg-danger/10 text-danger";
  }
  if (severity === "medium") {
    return "border-warning/20 bg-warning/10 text-warning";
  }
  return "border-info/20 bg-info-subtle text-info";
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

function SeoIntelligenceCard({
  payload,
  copy,
  locale,
  loading,
  error,
}: {
  payload: SeoIntelligenceResponse | null;
  copy: typeof SEO_INTELLIGENCE_COPY.en;
  locale: ReadinessLocale;
  loading: boolean;
  error: string | null;
}) {
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  if (loading && !payload) {
    return <section className="smx-panel-subtle animate-pulse px-5 py-6 text-sm text-ink-muted">{copy.loading}</section>;
  }
  if (error && !payload) {
    return <section className="rounded-xl border border-warning/20 bg-warning/10 px-5 py-4 text-sm text-warning" role="alert">{copy.failed} {localizeProjectError(error, locale)}</section>;
  }
  if (!payload) return null;

  const coverage = Math.round(payload.portfolio.coverage_ratio * 100);
  const dataStatusLabel = payload.data_quality.status === "good"
    ? copy.dataGood
    : payload.data_quality.status === "insufficient"
      ? copy.dataInsufficient
      : copy.dataLimited;
  const nextActionCopy = SEO_NEXT_ACTION_COPY[locale] ?? SEO_NEXT_ACTION_COPY.en;
  return (
    <section className="smx-panel-subtle overflow-hidden" aria-live="polite">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-line px-5 py-4">
        <div>
          <p className="text-xs font-medium text-ink-muted">{copy.subtitle}</p>
          <h3 className="mt-1 text-xl font-semibold tracking-tight text-ink">{copy.title}</h3>
          <p className="mt-1 text-xs text-ink-muted">{copy.safe}</p>
        </div>
        <span className={clsx("rounded-full border px-2.5 py-1 text-xs font-semibold", payload.data_quality.status === "good" ? "border-success/20 bg-success/10 text-success" : payload.data_quality.status === "insufficient" ? "border-danger/20 bg-danger-subtle text-danger" : "border-warning/20 bg-warning/10 text-warning")}>
          {copy.dataQuality}: {dataStatusLabel}
        </span>
      </div>
      <div className="grid gap-3 p-5 sm:grid-cols-3">
        <PerformanceSummaryCard label={copy.health} value={`${payload.portfolio.health_score}/100`} tone={payload.portfolio.health_score < 55 ? "warning" : "default"} />
        <PerformanceSummaryCard label={copy.coverage} value={`${new Intl.NumberFormat(localeName).format(coverage)}%`} />
        <PerformanceSummaryCard label={copy.highPriority} value={payload.portfolio.high_priority_count} tone={payload.portfolio.high_priority_count > 0 ? "warning" : "default"} />
      </div>
      <div className="border-t border-line px-5 py-4">
        <h4 className="text-sm font-semibold text-ink">{copy.queue}</h4>
        {payload.recommended_queue.length === 0 ? (
          <p className="mt-3 text-sm text-ink-muted">{copy.noQueue}</p>
        ) : (
          <div className="mt-3 space-y-2">
            {payload.recommended_queue.slice(0, 5).map((item) => (
              <article key={item.opportunity_id} className="flex items-start gap-3 rounded-xl border border-line bg-surface px-3 py-3">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-brand/10 text-xs font-bold text-brand">{item.rank}</span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="truncate text-sm font-semibold text-ink">{item.article_title || item.url}</p>
                    <span className="rounded-md bg-ink/[0.04] px-2 py-0.5 text-xs font-semibold text-ink-secondary">{item.priority_score}/100</span>
                  </div>
                  <p className="mt-1 text-xs leading-5 text-ink-muted">
                    {nextActionCopy[item.type as keyof typeof SEO_NEXT_ACTION_COPY.en] ?? item.next_action?.title ?? item.type}
                  </p>
                  <p className="mt-1 text-xs text-ink-muted">{Math.round(item.confidence * 100)}% {copy.confidence}</p>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
      {payload.data_quality.warnings.length > 0 ? (
        <details className="border-t border-line px-5 py-3 text-xs text-ink-secondary">
          <summary className="cursor-pointer font-medium">{copy.dataQuality} ({payload.data_quality.warnings.length})</summary>
          <ul className="mt-2 space-y-1.5">
            {payload.data_quality.warnings.map((warning) => (
              <li key={warning.code}>
                • {SEO_WARNING_COPY[locale]?.[warning.code as keyof typeof SEO_WARNING_COPY.en] ?? warning.message}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
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
  seoIntelligence,
  seoIntelligenceCopy,
  seoIntelligenceLoading,
  seoIntelligenceError,
  onOpenImport,
  onDismiss,
  searchConsole,
  searchConsoleCopy,
  searchConsoleLoading,
  searchConsoleAction,
  searchConsoleError,
  onConnectSearchConsole,
  onRefreshSearchConsole,
  onRefreshSearchConsoleProperties,
  onSelectSearchConsoleProperty,
  onSyncSearchConsole,
  onDisconnectSearchConsole,
}: {
  copy: typeof PERFORMANCE_COPY.en;
  locale: ReadinessLocale;
  canManageProjects: boolean;
  feedback: ProjectPerformanceFeedback | null;
  loading: boolean;
  error: string | null;
  dismissingOpportunityId: string | null;
  onRefresh: () => void;
  seoIntelligence: SeoIntelligenceResponse | null;
  seoIntelligenceCopy: typeof SEO_INTELLIGENCE_COPY.en;
  seoIntelligenceLoading: boolean;
  seoIntelligenceError: string | null;
  onOpenImport: () => void;
  onDismiss: (opportunityId: string) => void;
  searchConsole: SearchConsoleStatus | null;
  searchConsoleCopy: typeof SEARCH_CONSOLE_COPY.en;
  searchConsoleLoading: boolean;
  searchConsoleAction: string | null;
  searchConsoleError: string | null;
  onConnectSearchConsole: () => void;
  onRefreshSearchConsole: () => void;
  onRefreshSearchConsoleProperties: () => void;
  onSelectSearchConsoleProperty: (siteUrl: string) => void;
  onSyncSearchConsole: () => void;
  onDisconnectSearchConsole: () => void;
}) {
  const hasData = Boolean(
    feedback && (feedback.snapshots.length > 0 || feedback.opportunities.length > 0)
  );

  return (
    <div className="max-w-5xl space-y-4 animate-fade-in">
      <SearchConsoleCard
        copy={searchConsoleCopy}
        locale={locale}
        canManageProjects={canManageProjects}
        status={searchConsole}
        loading={searchConsoleLoading}
        action={searchConsoleAction}
        error={searchConsoleError}
        onConnect={onConnectSearchConsole}
        onRefresh={onRefreshSearchConsole}
        onRefreshProperties={onRefreshSearchConsoleProperties}
        onSelectProperty={onSelectSearchConsoleProperty}
        onSync={onSyncSearchConsole}
        onDisconnect={onDisconnectSearchConsole}
      />
      <SeoIntelligenceCard
        payload={seoIntelligence}
        copy={seoIntelligenceCopy}
        locale={locale}
        loading={seoIntelligenceLoading}
        error={seoIntelligenceError}
      />
      <section className="smx-panel-subtle p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-xs font-medium text-ink-muted">{copy.subtitle}</p>
            <h3 className="mt-1 text-xl font-semibold tracking-tight text-ink">
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
          <div className="mt-5 rounded-lg border border-line bg-surface-alt px-4 py-3 text-sm font-medium text-ink-secondary">
            {copy.loading}
          </div>
        )}

        {error && (
          <div className="mt-5 rounded-lg border border-danger/20 bg-danger/10 px-4 py-3 text-sm font-medium text-danger" role="alert">
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
              valueClassName="text-base"
            />
          </div>
        )}
      </section>

      {feedback && !hasData && !loading && (
        <section className="rounded-xl border border-dashed border-line bg-surface p-6 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-brand/15 bg-brand/10 text-brand">
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 19V5m0 14h16M8 16v-5m4 5V8m4 8v-7" />
            </svg>
          </div>
          <h4 className="text-body-lg font-semibold text-ink">{copy.emptyTitle}</h4>
          <p className="mx-auto mt-2 max-w-xl text-sm leading-5 text-ink-muted">
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
          <div className="rounded-xl border border-line bg-surface">
            <div className="border-b border-line px-4 py-3">
              <h4 className="text-base font-semibold text-ink">{copy.opportunitiesTitle}</h4>
            </div>
            {feedback.opportunities.length === 0 ? (
              <p className="px-4 py-6 text-sm leading-5 text-ink-muted">{copy.noOpportunities}</p>
            ) : (
              <div className="divide-y divide-line">
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

          <aside className="rounded-xl border border-line bg-surface">
            <div className="border-b border-line px-4 py-3">
              <h4 className="text-base font-semibold text-ink">{copy.recentSnapshots}</h4>
            </div>
            {feedback.snapshots.length === 0 ? (
              <p className="px-4 py-5 text-sm leading-5 text-ink-muted">{copy.noSnapshots}</p>
            ) : (
              <div className="divide-y divide-line">
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

function SearchConsoleCard({
  copy,
  locale,
  canManageProjects,
  status,
  loading,
  action,
  error,
  onConnect,
  onRefresh,
  onRefreshProperties,
  onSelectProperty,
  onSync,
  onDisconnect,
}: {
  copy: typeof SEARCH_CONSOLE_COPY.en;
  locale: ReadinessLocale;
  canManageProjects: boolean;
  status: SearchConsoleStatus | null;
  loading: boolean;
  action: string | null;
  error: string | null;
  onConnect: () => void;
  onRefresh: () => void;
  onRefreshProperties: () => void;
  onSelectProperty: (siteUrl: string) => void;
  onSync: () => void;
  onDisconnect: () => void;
}) {
  const latestRun = status?.recent_sync_runs?.[0];
  const propertyOptions = (status?.properties ?? []).map((item) => ({
    value: item.site_url,
    label: item.site_url,
  }));
  const statusTone = latestRun?.status === "failed"
    ? "border-danger/20 bg-danger/10 text-danger"
    : latestRun?.status === "succeeded"
      ? "border-success/20 bg-success/10 text-success"
      : "border-info/20 bg-info/10 text-info";
  return (
    <section className="smx-panel-subtle p-5" aria-labelledby="search-console-title">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 id="search-console-title" className="text-lg font-semibold text-ink">{copy.title}</h3>
            <span className="rounded-md border border-brand/20 bg-brand/10 px-2 py-0.5 text-xs font-semibold text-brand">{copy.readOnly}</span>
            <span className={clsx(
              "rounded-md border px-2 py-0.5 text-xs font-semibold",
              status?.connected
                ? "border-success/20 bg-success/10 text-success"
                : "border-line bg-ink/[0.03] text-ink-muted",
            )}>
              {status?.connected ? copy.connected : copy.disconnected}
            </span>
          </div>
          <p className="mt-2 max-w-2xl text-xs leading-5 text-ink-muted">{copy.subtitle}</p>
        </div>
        <Button variant="outlined" size="sm" loading={loading} onClick={onRefresh}>{PERFORMANCE_COPY[locale].refresh}</Button>
      </div>

      {loading && !status && <p className="mt-4 text-sm text-ink-muted">{copy.loading}</p>}
      {error && <div className="mt-4 rounded-lg border border-danger/20 bg-danger/10 px-3 py-2 text-xs text-danger" role="alert">{error}</div>}
      {status && !status.configured && <div className="mt-4 rounded-lg border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">{copy.notConfigured}</div>}

      {status?.configured && !status.connected && canManageProjects && (
        <Button className="mt-4" variant="primary" size="sm" loading={action === "connect"} onClick={onConnect}>{copy.connect}</Button>
      )}

      {status?.connected && (
        <div className="mt-4 space-y-4">
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
            <SelectDropdown
              label={copy.property}
              options={propertyOptions}
              value={status.selected_site_url ?? undefined}
              placeholder={propertyOptions.length ? copy.selectProperty : copy.noProperties}
              disabled={!canManageProjects || Boolean(action) || propertyOptions.length === 0}
              onChange={onSelectProperty}
            />
            <div className="flex flex-wrap gap-2">
              {canManageProjects && (
                <>
                  <Button variant="outlined" size="sm" loading={action === "properties"} disabled={Boolean(action) && action !== "properties"} onClick={onRefreshProperties}>{copy.refreshProperties}</Button>
                  <Button variant="primary" size="sm" loading={action === "sync"} disabled={!status.selected_site_url || (Boolean(action) && action !== "sync")} onClick={onSync}>{copy.syncNow}</Button>
                </>
              )}
            </div>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-line bg-surface-alt px-3 py-3">
            <div className="min-w-0 text-xs text-ink-muted">
              <span className="font-semibold text-ink-secondary">{copy.lastSync}: </span>
              {status.last_sync_at ? formatReadinessDate(status.last_sync_at, locale) : copy.never}
              {latestRun && (
                <span className={clsx("ms-2 inline-flex rounded-md border px-2 py-0.5 font-semibold", statusTone)}>
                  {latestRun.status}
                  {latestRun.status === "succeeded" ? ` · ${formatCompactNumber(latestRun.row_count)}` : ""}
                </span>
              )}
            </div>
            {canManageProjects && (
              <div className="flex gap-2">
                <Button variant="ghost" size="sm" loading={action === "connect"} onClick={onConnect}>{copy.reconnect}</Button>
                <Button variant="ghost" size="sm" loading={action === "disconnect"} disabled={Boolean(action) && action !== "disconnect"} onClick={onDisconnect}>{copy.disconnect}</Button>
              </div>
            )}
          </div>
          {(status.last_error_message || latestRun?.error_message) && (
            <div className="rounded-lg border border-danger/20 bg-danger/10 px-3 py-2 text-xs text-danger" role="alert">
              {status.last_error_message || latestRun?.error_message}
            </div>
          )}
        </div>
      )}
    </section>
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
    <div className="rounded-xl border border-line bg-surface-alt p-4">
      <p className="text-xs font-medium text-ink-muted">{label}</p>
      <p className={clsx(
        "mt-2 truncate font-semibold tabular-nums text-ink",
        typeof value === "number" ? "text-2xl" : "text-body-lg",
        tone === "warning" && "text-warning",
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
            <span className={clsx("inline-flex items-center rounded-lg border px-2.5 py-1 text-xs font-semibold", performanceSeverityClasses(opportunity.severity))}>
              {copy.severity[opportunity.severity as keyof typeof copy.severity] ?? opportunity.severity}
            </span>
            <span className="text-sm font-semibold text-ink">
              {performanceTypeLabel(copy, opportunity.type)}
            </span>
          </div>
          <p className="text-sm leading-5 text-ink-secondary">{opportunity.reason}</p>
          <p className="mt-1 text-sm leading-5 text-ink-muted">{opportunity.suggested_action}</p>
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

      <p className="mt-3 truncate text-xs text-ink-muted" dir="ltr">
        {opportunity.url}
      </p>
    </article>
  );
}

function PerformancePill({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1 rounded-lg border border-line bg-surface-alt px-2.5 py-1 text-xs">
      <span className="shrink-0 text-ink-muted">{label}</span>
      <span className="min-w-0 truncate font-semibold text-ink-secondary">{value}</span>
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
      <p className="truncate text-sm font-semibold text-ink" dir="ltr">
        {snapshot.url}
      </p>
      <p className="mt-1 text-xs text-ink-muted">
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
    <div className="rounded-lg border border-line bg-surface-alt px-2.5 py-2">
      <p className="text-xs font-medium text-ink-muted">{label}</p>
      <p className="mt-1 text-sm font-semibold tabular-nums text-ink">{value}</p>
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
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-ink-secondary">{t("projects.description")}</label>
          <textarea
            aria-label={t("projects.description")}
            disabled={!canManageProjects}
            className="min-h-[120px] w-full resize-none rounded-xl border border-line bg-surface px-3 py-2 text-base text-ink outline-none transition-colors duration-150 focus:border-brand focus:ring-1 focus:ring-brand/20 disabled:cursor-not-allowed disabled:opacity-60"
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
        <h3 className="mb-1 text-body-lg font-bold text-ink">{t("projects.tabWordpress")}</h3>
        <p className="text-sm text-ink-tertiary leading-relaxed">{t("projects.wpSubtitle")}</p>
      </div>

      {/* WordPress settings */}
      <div className="space-y-6 smx-panel-subtle p-5 md:p-6">
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

        <div className="flex justify-end gap-3 pt-6 border-block-start border-line">
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
        <p className="text-sm text-ink-tertiary">{t("projects.rulebookEmpty")}</p>
      </div>

      {/* Rulebook editor */}
      <div className="group relative flex min-h-[400px] flex-1 flex-col overflow-hidden rounded-xl border border-line bg-surface transition-colors duration-150 focus-within:border-brand focus-within:ring-2 focus-within:ring-brand/20">

          <textarea
            aria-label={t("projects.rulebook")}
            disabled={loading || !canManageProjects}
            className="w-full h-full flex-1 bg-transparent p-6 text-base text-ink leading-relaxed outline-none border-none resize-y disabled:opacity-50"
            value={rulebook}
            onChange={(e) => setRulebook(e.target.value)}
            placeholder={rulebookPlaceholder}
          />
      </div>

      {/* Save action */}
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
