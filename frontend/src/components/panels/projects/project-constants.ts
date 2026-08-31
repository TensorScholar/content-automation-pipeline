"use client";

import { ApiError } from "@/lib/api";

export const VERTICAL_OPTIONS = [
  { value: "tech", fa: "فناوری و نرم‌افزار", ar: "التكنولوجيا والبرمجيات", en: "Technology and Software" },
  { value: "health", fa: "سلامت و پزشکی", ar: "الصحة والطب", en: "Health and Medical" },
  { value: "ecommerce", fa: "فروشگاه و تجارت", ar: "المتاجر والتجارة", en: "E-Commerce" },
  { value: "education", fa: "آموزش و یادگیری", ar: "التعليم والتعلم", en: "Education and Learning" },
  { value: "finance", fa: "مالی و اقتصادی", ar: "المالية والاقتصاد", en: "Finance and Economy" },
  { value: "marketing", fa: "بازاریابی دیجیتال", ar: "التسويق الرقمي", en: "Digital Marketing" },
];

export const READINESS_COPY = {
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

export type ReadinessLocale = keyof typeof READINESS_COPY;

export const READINESS_ITEM_COPY = {
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

export function readinessItemKind(id: string, label: string) {
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

export function localizeReadinessLabel(id: string, label: string, locale: ReadinessLocale) {
  const kind = readinessItemKind(id, label);
  return kind ? READINESS_ITEM_COPY[locale].labels[kind] : label;
}

export function localizeReadinessText(value: string, locale: ReadinessLocale) {
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

export function formatReadinessDate(value: string, locale: ReadinessLocale) {
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

export const PERFORMANCE_COPY = {
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

export const SEARCH_CONSOLE_COPY = {
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

export const SEO_INTELLIGENCE_COPY = {
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

export const SEO_NEXT_ACTION_COPY = {
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

export const SEO_WARNING_COPY = {
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

export function extractError(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  return "Unexpected error";
}

export const PROJECT_ERROR_COPY = {
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

export function localizeProjectError(error: string, locale: ReadinessLocale): string {
  const normalized = error.trim().toLowerCase();
  if (normalized.includes("request timeout") || normalized.includes("timed out") || normalized === "timeout") {
    return PROJECT_ERROR_COPY[locale].timeout;
  }
  return error;
}

/* ── Hooks ── */
