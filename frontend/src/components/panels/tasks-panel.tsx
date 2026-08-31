"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import {
  TaskHistoryItem,
  TaskStatusResponse,
  ArticleDetail,
  DraftRiskAssessment,
  ArticleReviewAction,
  ArticleReviewState,
  ProjectReadiness,
} from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { StatusBadge as UiStatusBadge } from "@/components/ui/status-badge";
import { useToast } from "@/components/ui/toast";
import { Modal } from "@/components/ui/modal";
import { ToggleSwitch } from "@/components/ui/toggle-switch";

interface TasksPanelProps {
  token: string;
  canReview?: boolean;
}

interface QualityMetricsResponse {
  seo_score?: {
    score?: number;
    recommendations?: string[];
    component_scores?: Record<string, number>;
  };
  structure_score?: {
    score?: number;
    details?: Record<string, unknown>;
  };
  readability_grade?: string;
  overall_quality?: {
    score?: number;
    grade?: string;
  };
}

interface ContentHistoryResponse {
  current_version: Record<string, unknown>;
  revisions: Array<{
    id: string;
    content: string;
    revision_note: string;
    created_at: string;
    word_count: number;
  }>;
  total_revisions: number;
}

type FilterTab = "all" | "SUCCESS" | "FAILURE" | "RUNNING";
type DetailTab = "content" | "seo" | "export" | "history";
type ContentView = "reader" | "raw" | "edit";
type ReadinessItem = ProjectReadiness["blocking_items"][number];

const TASK_COPY = {
  en: {
    statuses: {
      SUCCESS: "Success",
      FAILURE: "Failed",
      FAILED: "Failed",
      BLOCKED: "Blocked",
      RUNNING: "Running",
      STARTED: "Running",
      PENDING: "Pending",
      RETRY: "Retrying",
    },
    live: "Live",
    completed: "Task completed successfully",
    failedSummary: "This task did not complete. Review the technical details if investigation is needed.",
    notRecorded: "Not recorded for this task",
    qualityDiagnostics: "Release gate details",
    actualWordCount: "Actual word count",
    allowedWordRange: "Allowed word range",
    headings: "Headings",
    paragraphs: "Paragraphs",
    regenerationAttempted: "Regeneration attempted",
    findings: "Quality findings",
    noFaq: "No FAQ section was detected.",
    seo: "SEO",
    htmlSource: "HTML source",
    markdownSource: "Markdown source",
    seoLoading: "Loading SEO analysis for this article...",
    seoEmpty: "SEO output has not been generated for this article yet.",
    seoError: "SEO analysis could not be loaded.",
    technicalDetails: "Technical details",
    downloadHtml: "Download HTML",
    downloadMarkdown: "Download Markdown source",
    bulkDownloadComplete: (count: number) => `${count} article${count === 1 ? "" : "s"} downloaded.`,
    bulkDownloadPartial: (downloaded: number, omitted: number) =>
      `${downloaded} article${downloaded === 1 ? "" : "s"} downloaded; ${omitted} not included.`,
    bulkDownloadUnavailable: "No completed articles could be retrieved for download.",
    articleTitle: "Article",
    language: "Language",
    keyword: "Keyword",
    generatedAt: "Generated",
    loadingDetail: "Loading article details\u2026",
    gradeExcellent: "Excellent",
    gradeGood: "Good",
    gradeFair: "Fair",
    gradeNeedsWork: "Needs work",
    history: "History",
    revisionHistory: "Revision history",
    noRevisions: "No revisions yet.",
    manualEdit: "Manual edit",
    revisionNote: "Manual edit in Task History",
    viewPastContent: "View previous content",
    unsavedChanges: "Unsaved changes",
    save: "Save",
    editSaved: "Article changes saved.",
    editSaveFailed: "Article changes could not be saved.",
    wordUnit: "words",
  },
  fa: {
    statuses: {
      SUCCESS: "موفق",
      FAILURE: "ناموفق",
      FAILED: "ناموفق",
      BLOCKED: "مسدود",
      RUNNING: "در حال اجرا",
      STARTED: "در حال اجرا",
      PENDING: "در انتظار",
      RETRY: "تلاش مجدد",
    },
    live: "زنده",
    completed: "پردازش با موفقیت کامل شد",
    failedSummary: "این پردازش کامل نشد. در صورت نیاز به بررسی، جزئیات فنی را باز کنید.",
    notRecorded: "برای این پردازش ثبت نشده است",
    qualityDiagnostics: "جزئیات دروازه کیفیت",
    actualWordCount: "تعداد کلمات واقعی",
    allowedWordRange: "بازه مجاز کلمات",
    headings: "تعداد تیترها",
    paragraphs: "تعداد پاراگراف‌ها",
    regenerationAttempted: "تلاش برای تولید دوباره",
    findings: "یافته‌های کیفیت",
    noFaq: "بخش پرسش‌های متداول شناسایی نشد.",
    seo: "سئو",
    htmlSource: "منبع HTML",
    markdownSource: "منبع Markdown",
    seoLoading: "در حال بارگذاری تحلیل سئوی این مقاله...",
    seoEmpty: "هنوز خروجی سئو برای این مقاله تولید نشده است.",
    seoError: "بارگذاری تحلیل سئو ممکن نبود.",
    technicalDetails: "جزئیات فنی",
    downloadHtml: "دانلود HTML",
    downloadMarkdown: "دانلود منبع Markdown",
    bulkDownloadComplete: (count: number) => `${count} مقاله دانلود شد.`,
    bulkDownloadPartial: (downloaded: number, omitted: number) =>
      `${downloaded} مقاله دانلود شد؛ ${omitted} مقاله در فایل قرار نگرفت.`,
    bulkDownloadUnavailable: "هیچ مقاله تکمیل‌شده‌ای برای دانلود دریافت نشد.",
    articleTitle: "مقاله",
    language: "زبان",
    keyword: "کلمه کلیدی",
    generatedAt: "تاریخ تولید",
    loadingDetail: "در حال بارگذاری جزئیات مقاله...",
    gradeExcellent: "عالی",
    gradeGood: "خوب",
    gradeFair: "متوسط",
    gradeNeedsWork: "نیازمند بهبود",
    history: "تاریخچه",
    revisionHistory: "تاریخچه ویرایش‌ها",
    noRevisions: "هنوز ویرایشی ثبت نشده است.",
    manualEdit: "ویرایش دستی",
    revisionNote: "ویرایش دستی در تاریخچه پردازش‌ها",
    viewPastContent: "مشاهده محتوای پیشین",
    unsavedChanges: "تغییرات ذخیره‌نشده",
    save: "ذخیره",
    editSaved: "تغییرات مقاله ذخیره شد.",
    editSaveFailed: "ذخیره تغییرات مقاله ممکن نبود.",
    wordUnit: "کلمه",
  },
  ar: {
    statuses: {
      SUCCESS: "ناجح",
      FAILURE: "فشل",
      FAILED: "فشل",
      BLOCKED: "محظور",
      RUNNING: "قيد التشغيل",
      STARTED: "قيد التشغيل",
      PENDING: "قيد الانتظار",
      RETRY: "إعادة المحاولة",
    },
    live: "مباشر",
    completed: "اكتملت المهمة بنجاح",
    failedSummary: "لم تكتمل هذه المهمة. افتح التفاصيل الفنية عند الحاجة إلى التحقيق.",
    notRecorded: "غير مسجل لهذه المهمة",
    qualityDiagnostics: "تفاصيل بوابة الجودة",
    actualWordCount: "عدد الكلمات الفعلي",
    allowedWordRange: "نطاق الكلمات المسموح",
    headings: "عدد العناوين",
    paragraphs: "عدد الفقرات",
    regenerationAttempted: "تمت محاولة إعادة الإنشاء",
    findings: "نتائج الجودة",
    noFaq: "لم يتم العثور على قسم للأسئلة الشائعة.",
    seo: "تحسين محركات البحث",
    htmlSource: "مصدر HTML",
    markdownSource: "مصدر Markdown",
    seoLoading: "جارٍ تحميل تحليل تحسين البحث لهذه المقالة...",
    seoEmpty: "لم يتم إنشاء مخرجات تحسين البحث لهذه المقالة بعد.",
    seoError: "تعذر تحميل تحليل تحسين البحث.",
    technicalDetails: "التفاصيل التقنية",
    downloadHtml: "تنزيل HTML",
    downloadMarkdown: "تنزيل مصدر Markdown",
    bulkDownloadComplete: (count: number) => `تم تنزيل ${count} مقالة.`,
    bulkDownloadPartial: (downloaded: number, omitted: number) =>
      `تم تنزيل ${downloaded} مقالة؛ لم يتم تضمين ${omitted} مقالة في الملف.`,
    bulkDownloadUnavailable: "تعذر استرداد أي مقالة مكتملة للتنزيل.",
    articleTitle: "المقال",
    language: "اللغة",
    keyword: "الكلمة المفتاحية",
    generatedAt: "تاريخ الإنشاء",
    loadingDetail: "جارٍ تحميل تفاصيل المقال...",
    gradeExcellent: "ممتاز",
    gradeGood: "جيد",
    gradeFair: "متوسط",
    gradeNeedsWork: "يحتاج تحسين",
    history: "السجل",
    revisionHistory: "سجل التعديلات",
    noRevisions: "لم تُسجل أي تعديلات بعد.",
    manualEdit: "تعديل يدوي",
    revisionNote: "تعديل يدوي في سجل المعالجات",
    viewPastContent: "عرض المحتوى السابق",
    unsavedChanges: "تغييرات غير محفوظة",
    save: "حفظ",
    editSaved: "تم حفظ تغييرات المقال.",
    editSaveFailed: "تعذر حفظ تغييرات المقال.",
    wordUnit: "كلمة",
  },
} as const;

type TaskLocale = keyof typeof TASK_COPY;

const RISK_COPY = {
  en: {
    title: "Draft Risk",
    low: "Low risk",
    medium: "Needs review",
    high: "High risk",
    blocked: "Blocked",
    loading: "Checking publish risk...",
    blockedPublish: "Publishing is blocked until the critical issue is fixed.",
  },
  fa: {
    title: "ریسک پیش‌نویس",
    low: "ریسک پایین",
    medium: "نیازمند بررسی",
    high: "ریسک بالا",
    blocked: "مسدود",
    loading: "در حال بررسی ریسک انتشار...",
    blockedPublish: "تا رفع مورد بحرانی، انتشار مسدود است.",
  },
  ar: {
    title: "مخاطر المسودة",
    low: "مخاطر منخفضة",
    medium: "يحتاج مراجعة",
    high: "مخاطر عالية",
    blocked: "محظور",
    loading: "جارٍ فحص مخاطر النشر...",
    blockedPublish: "النشر محظور حتى إصلاح المشكلة الحرجة.",
  },
};

const REVIEW_COPY = {
  en: {
    title: "Review",
    subtitle: "Manager decision before scheduled or public publishing.",
    checklist: "Quality checklist",
    pending_review: "Pending review",
    approved: "Approved",
    changes_requested: "Changes requested",
    rejected: "Rejected",
    approve: "Approve",
    reject: "Reject",
    request_changes: "Request changes",
    noteLabel: "Decision note",
    notePlaceholder: "Add concise feedback for the author or operator.",
    noteRequired: "Feedback is required for this decision.",
    noDecision: "No manager decision yet.",
    reviewedBy: "Reviewed by",
    blockedTitle: "Approval blocked",
    approveBlocked: "Resolve blocking checklist items before approval.",
    liveBlocked: "Public publishing requires manager approval.",
    publishNeedsApproval: "Needs manager approval",
    reviewUpdated: "Review decision saved.",
    reviewError: "Could not update review decision.",
    loading: "Loading review state...",
    unavailable: "Review status is unavailable.",
    checks: {
      title: "Title",
      content: "Article body",
      publish_risk: "Publish risk",
      metadata: "Search metadata",
      keywords: "Keywords",
    },
  },
  fa: {
    title: "بازبینی",
    subtitle: "تصمیم مدیر پیش از زمان‌بندی یا انتشار عمومی.",
    checklist: "چک‌لیست کیفیت",
    pending_review: "در انتظار بازبینی",
    approved: "تأیید شده",
    changes_requested: "نیازمند اصلاح",
    rejected: "رد شده",
    approve: "تأیید",
    reject: "رد",
    request_changes: "درخواست اصلاح",
    noteLabel: "یادداشت تصمیم",
    notePlaceholder: "بازخورد کوتاه برای نویسنده یا اپراتور بنویسید.",
    noteRequired: "برای این تصمیم، بازخورد لازم است.",
    noDecision: "هنوز تصمیم مدیری ثبت نشده است.",
    reviewedBy: "بازبینی توسط",
    blockedTitle: "تأیید مسدود است",
    approveBlocked: "قبل از تأیید، موارد مسدودکننده چک‌لیست را رفع کنید.",
    liveBlocked: "انتشار عمومی نیازمند تأیید مدیر است.",
    publishNeedsApproval: "نیازمند تأیید مدیر",
    reviewUpdated: "تصمیم بازبینی ذخیره شد.",
    reviewError: "امکان ثبت تصمیم بازبینی وجود نداشت.",
    loading: "در حال بارگذاری وضعیت بازبینی...",
    unavailable: "وضعیت بازبینی در دسترس نیست.",
    checks: {
      title: "عنوان",
      content: "متن مقاله",
      publish_risk: "ریسک انتشار",
      metadata: "متادیتای جستجو",
      keywords: "کلمات کلیدی",
    },
  },
  ar: {
    title: "المراجعة",
    subtitle: "قرار المدير قبل الجدولة أو النشر العام.",
    checklist: "قائمة الجودة",
    pending_review: "بانتظار المراجعة",
    approved: "تمت الموافقة",
    changes_requested: "تعديلات مطلوبة",
    rejected: "مرفوض",
    approve: "موافقة",
    reject: "رفض",
    request_changes: "طلب تعديلات",
    noteLabel: "ملاحظة القرار",
    notePlaceholder: "اكتب ملاحظات موجزة للكاتب أو المشغل.",
    noteRequired: "الملاحظات مطلوبة لهذا القرار.",
    noDecision: "لا يوجد قرار إداري بعد.",
    reviewedBy: "راجعه",
    blockedTitle: "الموافقة محظورة",
    approveBlocked: "أصلح عناصر القائمة الحاجبة قبل الموافقة.",
    liveBlocked: "النشر العام يتطلب موافقة المدير.",
    publishNeedsApproval: "يتطلب موافقة المدير",
    reviewUpdated: "تم حفظ قرار المراجعة.",
    reviewError: "تعذر تحديث قرار المراجعة.",
    loading: "جارٍ تحميل حالة المراجعة...",
    unavailable: "حالة المراجعة غير متاحة.",
    checks: {
      title: "العنوان",
      content: "نص المقال",
      publish_risk: "مخاطر النشر",
      metadata: "بيانات البحث",
      keywords: "الكلمات المفتاحية",
    },
  },
};

type ReviewCopy = (typeof REVIEW_COPY)["en"];

const PUBLISH_COPY = {
  en: {
    wordpressRequired: "WordPress connection required",
    checkingWordPress: "Checking WordPress",
    wordpressBlocked: "Configure WordPress before public publishing.",
    queued: "Publication queued. Waiting for WordPress verification…",
    retrying: "WordPress is temporarily unavailable. Retrying safely…",
    completed: "Published and verified on WordPress.",
    accepted: "Publication is still running in the background. You can safely close this view.",
  },
  fa: {
    wordpressRequired: "اتصال وردپرس لازم است",
    checkingWordPress: "در حال بررسی وردپرس",
    wordpressBlocked: "پیش از انتشار عمومی، وردپرس را متصل کنید.",
    queued: "انتشار در صف قرار گرفت؛ منتظر تأیید وردپرس هستیم…",
    retrying: "وردپرس موقتاً در دسترس نیست؛ تلاش مجدد امن در حال انجام است…",
    completed: "مطلب در وردپرس منتشر و تأیید شد.",
    accepted: "انتشار در پس‌زمینه ادامه دارد و بستن این صفحه امن است.",
  },
  ar: {
    wordpressRequired: "يلزم ربط ووردبريس",
    checkingWordPress: "جارٍ فحص ووردبريس",
    wordpressBlocked: "اربط ووردبريس قبل النشر العام.",
    queued: "تمت إضافة النشر إلى الطابور؛ جارٍ التحقق من ووردبريس…",
    retrying: "ووردبريس غير متاح مؤقتًا؛ تجري إعادة المحاولة بأمان…",
    completed: "تم النشر والتحقق في ووردبريس.",
    accepted: "يستمر النشر في الخلفية ويمكن إغلاق هذه الصفحة بأمان.",
  },
} as const;

function isWordPressPublishReadinessItem(item: ReadinessItem): boolean {
  const text = `${item.id} ${item.label} ${item.message} ${item.remediation ?? ""}`.toLowerCase();
  return /\bwordpress\b|وردپرس|ووردبريس/.test(text);
}

export function TasksPanel({ token, canReview = false }: TasksPanelProps) {
  const { t, locale } = useI18n();
  const { showToast } = useToast();
  const [tasks, setTasks] = useState<TaskHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterTab>("all");
  const [search, setSearch] = useState("");
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [liveStatus, setLiveStatus] = useState<TaskStatusResponse | null>(null);
  const [streamActive, setStreamActive] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // deep view state
  const [detailArticle, setDetailArticle] = useState<ArticleDetail | null>(null);
  const [detailTab, setDetailTab] = useState<DetailTab>("content");
  const [contentView, setContentView] = useState<ContentView>("reader");
  const [editContent, setEditContent] = useState("");
  const [wpPublishing, setWpPublishing] = useState(false);
  const publishControllerRef = useRef<AbortController | null>(null);
  const [wpResult, setWpResult] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      publishControllerRef.current?.abort();
    };
  }, []);
  const [riskAssessment, setRiskAssessment] = useState<DraftRiskAssessment | null>(null);
  const [riskLoading, setRiskLoading] = useState(false);
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetricsResponse | null>(null);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [qualityError, setQualityError] = useState<string | null>(null);
  const [reviewState, setReviewState] = useState<ArticleReviewState | null>(null);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [reviewAction, setReviewAction] = useState<Exclude<ArticleReviewAction, "approve"> | null>(null);
  const [reviewNote, setReviewNote] = useState("");
  const [reviewSubmitting, setReviewSubmitting] = useState(false);
  const [publishReadiness, setPublishReadiness] = useState<ProjectReadiness | null>(null);
  const [publishReadinessLoading, setPublishReadinessLoading] = useState(false);

  const [savingEdit, setSavingEdit] = useState(false);
  const [articleHistory, setArticleHistory] = useState<ContentHistoryResponse | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const loadHistory = useCallback(async (articleId: string, signal?: AbortSignal) => {
    setLoadingHistory(true);
    try {
      const res = await apiRequest<ContentHistoryResponse>(`/content/${articleId}/history`, { token, signal });
      if (signal?.aborted) return;
      setArticleHistory(res);
    } catch (e) {
      if (signal?.aborted) return;
      console.error(e);
      setArticleHistory(null);
    } finally {
      if (signal?.aborted) return;
      setLoadingHistory(false);
    }
  }, [token]);

  useEffect(() => {
    if (detailTab === "history" && detailArticle?.id) {
      const controller = new AbortController();
      void loadHistory(detailArticle.id, controller.signal);
      return () => controller.abort();
    }
  }, [detailTab, detailArticle?.id, loadHistory]);

  const handleSaveEdit = async () => {
    if (!detailArticle || editContent === detailArticle.content) return;
    setSavingEdit(true);
    try {
      const saved = await apiRequest<{
        content: string;
        word_count: number;
      }>(`/content/${detailArticle.id}`, {
        method: "PUT",
        token,
        body: { content: editContent, revision_note: TASK_COPY[locale].revisionNote },
      });
      setEditContent(saved.content);
      setDetailArticle({
        ...detailArticle,
        content: saved.content,
        word_count: saved.word_count,
      });
      showToast("success", TASK_COPY[locale].editSaved);
      // If history was already loaded, refresh it so the new revision appears
      if (articleHistory) {
        void loadHistory(detailArticle.id);
      }
    } catch (e) {
      console.error("Failed to save edit", e);
      showToast(
        "error",
        e instanceof ApiError ? e.detail : TASK_COPY[locale].editSaveFailed,
      );
    } finally {
      setSavingEdit(false);
    }
  };

  const loadTasks = useCallback(async (signal?: AbortSignal) => {
    try {
      const res = await apiRequest<TaskHistoryItem[]>("/content/tasks", { token, signal });
      if (signal?.aborted) return;
      setTasks(Array.isArray(res) ? res : []);
    } catch {
      if (signal?.aborted) return;
      setTasks([]);
    } finally {
      if (signal?.aborted) return;
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    const controller = new AbortController();
    void loadTasks(controller.signal);
    return () => controller.abort();
  }, [loadTasks]);

  // Auto-refresh without overlapping requests.
  // FIX: Prevent overlapping polls
  useEffect(() => {
    if (!autoRefresh) return;
    const controller = new AbortController();
    let mounted = true;
    let isPolling = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const schedule = () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      timeoutId = setTimeout(() => { void poll(); }, 30000);
    };

    const poll = async () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      isPolling = true;
      try {
        await loadTasks(controller.signal);
      } catch (error) {
        if (!controller.signal.aborted) {
          console.error("Task polling error:", error);
        }
      } finally {
        isPolling = false;
      }
      if (mounted && !controller.signal.aborted) {
        schedule();
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
  }, [autoRefresh, loadTasks]);

  // Poll selected task without exposing bearer tokens in URLs.
  // FIX: Prevent overlapping polls by tracking polling state with ref
  useEffect(() => {
    if (!selectedTaskId) {
      setStreamActive(false);
      setLiveStatus(null);
      return;
    }

    const controller = new AbortController();
    let mounted = true;
    let isPolling = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    setStreamActive(true);

    const schedule = () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      timeoutId = setTimeout(() => { void poll(); }, 4000);
    };

    const poll = async () => {
      if (!mounted || isPolling || controller.signal.aborted) return;
      isPolling = true;

      try {
        const payload = await apiRequest<TaskStatusResponse>(`/content/task/${selectedTaskId}`, {
          token,
          signal: controller.signal,
          timeoutMs: 8000,
        });
        if (!mounted || controller.signal.aborted) return;
        setLiveStatus(payload);
        if (payload.ready) {
          setStreamActive(false);
          isPolling = false;
          void loadTasks(controller.signal);
          return;
        }
        isPolling = false;
        if (mounted && !controller.signal.aborted) {
          schedule();
        }
      } catch (error) {
        if (!mounted || controller.signal.aborted) return;
        setStreamActive(false);
        isPolling = false;
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
  }, [selectedTaskId, token, loadTasks]);

  // Load article detail when task is SUCCESS
  useEffect(() => {
    const articleId = liveStatus?.result?.article_id;
    if (!articleId) {
      setDetailArticle(null);
      setRiskAssessment(null);
      setRiskLoading(false);
      setQualityMetrics(null);
      setQualityLoading(false);
      setQualityError(null);
      setReviewState(null);
      setReviewLoading(false);
      setReviewError(null);
      return;
    }
    const controller = new AbortController();
    const load = async () => {
      setRiskLoading(true);
      setReviewLoading(true);
      setQualityLoading(true);
      setReviewError(null);
      try {
        const [articleResult, riskResult, reviewResult, qualityResult] = await Promise.allSettled([
          apiRequest<ArticleDetail>(`/content/${articleId}`, {
            token,
            signal: controller.signal,
          }),
          apiRequest<DraftRiskAssessment>(`/content/${articleId}/risk-assessment`, {
            token,
            signal: controller.signal,
          }),
          apiRequest<ArticleReviewState>(`/content/${articleId}/review`, {
            token,
            signal: controller.signal,
          }),
          apiRequest<QualityMetricsResponse>(`/content/${articleId}/quality`, {
            token,
            signal: controller.signal,
            timeoutMs: 20000,
          }),
        ]);
        if (controller.signal.aborted) return;
        if (articleResult.status === "fulfilled") {
          setDetailArticle(articleResult.value);
          setEditContent(articleResult.value.content ?? "");
        } else {
          setDetailArticle(null);
        }
        if (qualityResult.status === "fulfilled") {
          setQualityMetrics(qualityResult.value);
          setQualityError(null);
        } else {
          setQualityMetrics(null);
          setQualityError(null);
        }
        setRiskAssessment(riskResult.status === "fulfilled" ? riskResult.value : null);
        if (reviewResult.status === "fulfilled") {
          setReviewState(reviewResult.value);
        } else {
          setReviewState(null);
          setReviewError(REVIEW_COPY[locale].unavailable);
        }
      } catch {
        if (!controller.signal.aborted) {
          setDetailArticle(null);
          setRiskAssessment(null);
          setReviewState(null);
          setReviewError(REVIEW_COPY[locale].unavailable);
        }
      } finally {
        if (!controller.signal.aborted) setRiskLoading(false);
        if (!controller.signal.aborted) setReviewLoading(false);
        if (!controller.signal.aborted) setQualityLoading(false);
      }
    };
    void load();
    return () => controller.abort();
  }, [liveStatus?.result?.article_id, locale, token]);

  const publishProjectId = detailArticle?.project_id ?? liveStatus?.result?.project_id;

  useEffect(() => {
    if (!publishProjectId) {
      setPublishReadiness(null);
      setPublishReadinessLoading(false);
      return;
    }

    const controller = new AbortController();
    setPublishReadiness(null);
    setPublishReadinessLoading(true);
    apiRequest<ProjectReadiness>(`/projects/${publishProjectId}/readiness`, {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => {
        if (!controller.signal.aborted) setPublishReadiness(payload);
      })
      .catch(() => {
        if (!controller.signal.aborted) setPublishReadiness(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setPublishReadinessLoading(false);
      });

    return () => controller.abort();
  }, [publishProjectId, token]);

  useEffect(() => {
    const articleId = detailArticle?.id;
    if (detailTab !== "seo" || !articleId || qualityMetrics || qualityLoading) return;

    const controller = new AbortController();
    setQualityLoading(true);
    setQualityError(null);

    apiRequest<QualityMetricsResponse>(`/content/${articleId}/quality`, {
      token,
      signal: controller.signal,
      timeoutMs: 20000,
    })
      .then((payload) => {
        if (!controller.signal.aborted) setQualityMetrics(payload);
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setQualityMetrics(null);
          setQualityError(error instanceof ApiError ? error.detail : t("common.unexpectedError"));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setQualityLoading(false);
      });

    return () => controller.abort();
  }, [detailArticle?.id, detailTab, qualityLoading, qualityMetrics, t, token]);

  // KPI counters
  const kpis = useMemo(() => {
    const total = tasks.length;
    const success = tasks.filter((t) => t.status?.toUpperCase() === "SUCCESS").length;
    const failure = tasks.filter((t) => ["FAILURE", "FAILED"].includes(t.status?.toUpperCase() ?? "")).length;
    const running = total - success - failure;
    return { total, success, failure, running };
  }, [tasks]);

  const filtered = useMemo(() => {
    let list = tasks;
    if (filter === "RUNNING") {
      list = list.filter((t) => !["SUCCESS", "FAILURE", "FAILED"].includes(t.status?.toUpperCase() ?? ""));
    } else if (filter !== "all") {
      list = list.filter((t) => t.status?.toUpperCase() === filter || (filter === "FAILURE" && t.status?.toUpperCase() === "FAILED"));
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter((t) => t.task_id.toLowerCase().includes(q) || (t.topic ?? "").toLowerCase().includes(q));
    }
    return list;
  }, [tasks, filter, search]);

  const onDeleteTask = async (taskId: string) => {
    setDeleteConfirmId(null);
    try {
      await apiRequest<void>(`/content/task/${taskId}`, { method: "DELETE", token });
      showToast("success", t("tasks.taskDeleted"));
      if (selectedTaskId === taskId) { setSelectedTaskId(null); setLiveStatus(null); }
      await loadTasks();
    } catch (e) {
      showToast("error", e instanceof ApiError ? e.detail : (t("common.unexpectedError")));
    }
  };

  const onBulkDownload = async () => {
    const allSuccessful = tasks.filter((t) => t.status?.toUpperCase() === "SUCCESS");
    const successful = allSuccessful.slice(0, 20);
    const results: string[] = [];
    let omitted = allSuccessful.length - successful.length;
    for (const task of successful) {
      const articleId = (task.result as Record<string, unknown> | undefined)?.article_id;
      if (!articleId) {
        omitted += 1;
        continue;
      }
      try {
        const article = await apiRequest<ArticleDetail>(`/content/${String(articleId)}`, { token });
        results.push(`--- ${article.title} ---\n\n${article.content}\n\n`);
      } catch {
        omitted += 1;
      }
    }
    if (results.length === 0) {
      showToast("error", TASK_COPY[locale].bulkDownloadUnavailable);
      return;
    }
    const blob = new Blob([results.join("\n\n")], { type: "text/plain;charset=utf-8" });
    downloadBlob(blob, "articles-bulk.txt");
    showToast(
      omitted > 0 ? "warning" : "success",
      omitted > 0
        ? TASK_COPY[locale].bulkDownloadPartial(results.length, omitted)
        : TASK_COPY[locale].bulkDownloadComplete(results.length),
    );
  };

  const onWpPublish = async (status: "draft" | "publish") => {
    if (!detailArticle || !publishProjectId) return;
    if (riskAssessment?.risk_level === "blocked") {
      setWpResult(RISK_COPY[locale].blockedPublish);
      return;
    }
    if (status === "publish" && reviewState?.status !== "approved") {
      setWpResult(REVIEW_COPY[locale].liveBlocked);
      return;
    }
    if (wordpressActionBlocked) {
      setWpResult(
        publishReadinessLoading
          ? PUBLISH_COPY[locale].checkingWordPress
          : PUBLISH_COPY[locale].wordpressBlocked,
      );
      return;
    }

    publishControllerRef.current?.abort();
    const controller = new AbortController();
    publishControllerRef.current = controller;
    const wait = (ms: number) => new Promise<void>((resolve, reject) => {
      if (controller.signal.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      const timer = window.setTimeout(() => {
        controller.signal.removeEventListener("abort", onAbort);
        resolve();
      }, ms);
      const onAbort = () => {
        window.clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      };
      controller.signal.addEventListener("abort", onAbort, { once: true });
    });

    setWpPublishing(true);
    setWpResult(null);
    try {
      const queued = await apiRequest<{ status: string; publish_status?: string }>(
        `/content/${detailArticle.id}/publish/wordpress`,
        { method: "POST", token, timeoutMs: 15000, signal: controller.signal },
        { project_id: publishProjectId, post_status: status },
      );
      if (queued.status === "success") {
        setWpResult(PUBLISH_COPY[locale].completed);
        return;
      }
      setWpResult(PUBLISH_COPY[locale].queued);
      const terminalSuccess = new Set(["published_as_draft", "published_scheduled", "published_public"]);
      const terminalFailure = new Set(["publish_failed", "publish_validation_failed"]);
      for (let attempt = 0; attempt < 40; attempt += 1) {
        await wait(1500);
        const publishState = await apiRequest<{
          publish_status: string;
          publish_error_message?: string | null;
        }>(`/content/${detailArticle.id}/publish/status`, { token, timeoutMs: 10000, signal: controller.signal });
        if (terminalSuccess.has(publishState.publish_status)) {
          setWpResult(PUBLISH_COPY[locale].completed);
          return;
        }
        if (terminalFailure.has(publishState.publish_status)) {
          throw new Error(publishState.publish_error_message || t("tasks.wpPublishError"));
        }
        if (publishState.publish_status === "publish_retrying") {
          setWpResult(PUBLISH_COPY[locale].retrying);
        }
      }
      setWpResult(PUBLISH_COPY[locale].accepted);
    } catch (error) {
      if (controller.signal.aborted) return;
      setWpResult(formatPublishResult(error instanceof ApiError ? error.detail : error, t("tasks.wpPublishError")));
    } finally {
      if (publishControllerRef.current === controller) publishControllerRef.current = null;
      if (!controller.signal.aborted) setWpPublishing(false);
    }
  };

  const submitReview = async (action: ArticleReviewAction, note?: string) => {
    if (!detailArticle) return;
    setReviewSubmitting(true);
    setReviewError(null);
    try {
      const updated = await apiRequest<ArticleReviewState, { action: ArticleReviewAction; note?: string }>(
        `/content/${detailArticle.id}/review`,
        {
          method: "POST",
          token,
          body: { action, note },
        }
      );
      setReviewState(updated);
      setReviewAction(null);
      setReviewNote("");
      showToast("success", REVIEW_COPY[locale].reviewUpdated);
    } catch (error) {
      const message = error instanceof ApiError ? error.detail : REVIEW_COPY[locale].reviewError;
      setReviewError(message);
      showToast("error", message);
    } finally {
      setReviewSubmitting(false);
    }
  };

  const filterTabs: Array<{ key: FilterTab; label: string; count: number }> = [
    { key: "all", label: t("common.all"), count: kpis.total },
    { key: "SUCCESS", label: t("common.success"), count: kpis.success },
    { key: "FAILURE", label: t("common.failure"), count: kpis.failure },
    { key: "RUNNING", label: t("common.running"), count: kpis.running },
  ];
  const seoFallback =
    TASK_COPY[locale].seoEmpty;
  const riskCopy = RISK_COPY[locale];
  const reviewCopy = REVIEW_COPY[locale];
  const publishCopy = PUBLISH_COPY[locale];
  const taskQualityScore = detailArticle?.quality_score ?? qualityMetrics?.overall_quality?.score;
  const taskCost = readFiniteNumber(liveStatus?.result?.cost) ?? detailArticle?.cost_usd;
  const sourceIsHtml = Boolean(detailArticle?.html_content?.trim());
  const sourceContent = sourceIsHtml ? detailArticle?.html_content ?? "" : detailArticle?.content ?? "";
  const articleDirection = resolveArticleDirection(detailArticle?.language, sourceContent);
  const wordpressReadinessUnavailable = Boolean(publishProjectId) && !publishReadinessLoading && !publishReadiness;
  const wordpressReadinessBlocked = publishReadiness ? !publishReadiness.can_publish : false;
  const wordpressBlockingItem = publishReadiness?.blocking_items.find(isWordPressPublishReadinessItem);
  const wordpressPublishBlocked = !publishProjectId || wordpressReadinessUnavailable || wordpressReadinessBlocked || Boolean(wordpressBlockingItem);
  const wordpressActionBlocked = publishReadinessLoading || wordpressPublishBlocked;
  const publicPublishBlocked = riskAssessment?.risk_level === "blocked" || reviewState?.status !== "approved" || publishReadinessLoading || wordpressPublishBlocked;
  const publicPublishLabel = reviewState?.status !== "approved"
    ? reviewCopy.publishNeedsApproval
    : publishReadinessLoading
      ? publishCopy.checkingWordPress
      : wordpressPublishBlocked
        ? publishCopy.wordpressRequired
        : t("tasks.wpLive");
  const publicPublishReason = reviewState?.status !== "approved"
    ? reviewCopy.liveBlocked
    : wordpressPublishBlocked
      ? publishCopy.wordpressBlocked
      : null;
  const wordpressActionReason = wordpressActionBlocked
    ? publishReadinessLoading
      ? publishCopy.checkingWordPress
      : publishCopy.wordpressBlocked
    : null;

  /* ════════════════════════════════════════════════════════════════════════
     Master-Detail Layout: Smooth Dynamic Drawers and Logical Properties Only
     ════════════════════════════════════════════════════════════════════════ */
  return (
    <section className="smx-page !max-w-none relative flex h-full min-h-0 min-w-0 flex-col gap-4 overflow-hidden" dir="auto">

      {/* Content header and toolbar */}
      <div className="smx-page-header">
        <div className="min-w-0 flex-1">
          <h2 className="smx-page-title">{t("tasks.title")}</h2>
          <p className="mt-1 text-sm text-ink-muted">{t("tasks.subtitle")}</p>
        </div>

        <div className="smx-toolbar min-w-0 flex-wrap md:w-auto">
          <div className="flex items-center px-1">
            <ToggleSwitch checked={autoRefresh} onChange={setAutoRefresh} label={t("tasks.autoRefresh")} />
          </div>

          <div className="hidden h-5 w-px bg-line sm:block" />

          <button
            type="button"
            onClick={() => void loadTasks()}
            className="smx-icon-button"
            title={t("common.refresh")}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
          </button>

          {kpis.success > 0 && (
            <Button variant="outlined" onClick={() => void onBulkDownload()} className="h-8 px-3 text-sm">
              <svg className="w-4 h-4 me-2 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
              {t("tasks.bulkDownload")}
            </Button>
          )}
        </div>
      </div>

      {/* Compact status filters: same filtering capability, less dashboard chrome. */}
      <div className="flex shrink-0 flex-wrap items-center gap-x-1 gap-y-2 border-b border-line pb-3">
        {[
          { key: "all", label: t("tasks.kpiTotal"), value: kpis.total },
          { key: "SUCCESS", label: t("tasks.kpiSuccess"), value: kpis.success },
          { key: "RUNNING", label: t("tasks.kpiRunning"), value: kpis.running },
          { key: "FAILURE", label: t("tasks.kpiFailure"), value: kpis.failure },
        ].map((item) => {
          const isActive = filter === item.key;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => setFilter(item.key as FilterTab)}
              aria-pressed={isActive}
              className={clsx(
                "flex min-h-9 items-center gap-2 rounded-md px-3 text-sm font-medium transition-colors duration-fast",
                isActive ? "bg-brand/[0.075] text-ink" : "text-ink-muted hover:bg-ink/[0.035] hover:text-ink",
              )}
            >
              <span>{item.label}</span>
              <span className="tabular-nums text-xs text-ink-tertiary">{item.value}</span>
            </button>
          );
        })}
      </div>

      <Modal open={Boolean(deleteConfirmId)} onClose={() => setDeleteConfirmId(null)} title={t("tasks.deleteTask")} footer={
        <>
          <Button variant="outlined" onClick={() => setDeleteConfirmId(null)}>{t("common.cancel")}</Button>
          <Button variant="danger" onClick={() => deleteConfirmId && void onDeleteTask(deleteConfirmId)}>{t("common.delete")}</Button>
        </>
      }>
        <p className="text-base text-ink-secondary">{t("tasks.confirmDeleteTask")}</p>
      </Modal>

      <Modal
        open={Boolean(reviewAction)}
        onClose={() => {
          if (!reviewSubmitting) {
            setReviewAction(null);
            setReviewNote("");
          }
        }}
        title={reviewAction ? reviewCopy[reviewAction] : reviewCopy.title}
        footer={
          <>
            <Button
              variant="outlined"
              disabled={reviewSubmitting}
              onClick={() => {
                setReviewAction(null);
                setReviewNote("");
              }}
            >
              {t("common.cancel")}
            </Button>
            <Button
              variant={reviewAction === "reject" ? "danger" : "primary"}
              loading={reviewSubmitting}
              disabled={!reviewNote.trim()}
              onClick={() => reviewAction && void submitReview(reviewAction, reviewNote)}
            >
              {reviewAction ? reviewCopy[reviewAction] : t("common.confirm")}
            </Button>
          </>
        }
      >
        <label className="block">
          <span className="text-sm font-semibold text-ink">
            {reviewCopy.noteLabel}
          </span>
          <textarea
            value={reviewNote}
            onChange={(event) => setReviewNote(event.target.value)}
            className="mt-2 min-h-[112px] w-full resize-none rounded-lg border border-line bg-surface px-3 py-2.5 text-base leading-6 text-ink outline-none transition-colors placeholder:text-ink-muted focus:border-brand focus:ring-2 focus:ring-brand/20"
            placeholder={reviewCopy.notePlaceholder}
            maxLength={2000}
          />
        </label>
        {!reviewNote.trim() && (
          <p className="mt-2 text-xs font-medium text-warning">
            {reviewCopy.noteRequired}
          </p>
        )}
      </Modal>

      {/* ── Search Bar ── */}
      <div className="flex flex-wrap items-center justify-end w-full">
        {/* Search Input with properly aligned Icon (pis) */}
        <div className="relative w-full md:w-80 shrink-0 group">
          <input
            aria-label={t("tasks.searchPlaceholder")}
            placeholder={t("tasks.searchPlaceholder")}
            className="smx-input w-full ps-10 pe-3 text-sm font-medium"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
            <svg className="absolute start-4 top-1/2 -translate-y-1/2 h-5 w-5 text-ink-muted group-focus-within:text-brand transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      {/* ── Dynamic Master-Detail Layout Wrapper ── */}
      <div className={clsx(
        "grid min-h-0 min-w-0 w-full flex-1 gap-0",
        selectedTaskId
          ? "grid-cols-1 lg:grid-cols-[minmax(220px,0.32fr)_minmax(0,0.68fr)] xl:grid-cols-[minmax(240px,0.28fr)_minmax(0,0.72fr)]"
          : "grid-cols-1"
      )}>

        {/* Master: Data Table */}
        <div className="flex min-h-0 min-w-0 flex-col overflow-hidden bg-surface">
          <div className="min-h-0 flex-1 overflow-auto">
            {!loading && filtered.length === 0 ? (
              <div className="flex w-full flex-col items-center justify-center px-6 py-24 text-center">
                <svg className="mb-4 h-16 w-16 text-ink-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                </svg>
                <p className="text-body-lg font-semibold text-ink-muted">{t("tasks.noTasks")}</p>
              </div>
            ) : (
            <table
              className={clsx(
                "w-full text-start border-collapse",
                selectedTaskId ? "table-fixed" : "min-w-[640px]"
              )}
            >
              <thead className="sticky top-0 z-10 border-b border-line bg-[rgb(var(--bg-secondary))]">
                <tr className="text-xs font-semibold text-ink-muted">
                  <th className={clsx("text-start font-bold", selectedTaskId ? "w-auto px-4 py-4" : "w-1/2 px-6 py-5")}>
                    {t("tasks.topic")}
                  </th>
                  <th className={clsx("text-start font-bold", selectedTaskId ? "w-28 px-4 py-4" : "w-1/4 px-6 py-5")}>
                    {t("tasks.status")}
                  </th>
                  {!selectedTaskId && (
                    <>
                      <th className="w-1/4 px-6 py-5 text-start font-bold">{t("tasks.created")}</th>
                      <th className="sr-only w-16 px-6 py-5 text-end font-bold">{t("users.action")}</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {loading ? (
                  [1, 2, 3, 4, 5].map((i) => (
                    <tr key={i} className="animate-pulse">
                      <td className={selectedTaskId ? "px-4 py-4" : "px-6 py-5"}>
                        <div className="h-5 bg-surface-tertiary rounded-md w-3/4 mb-2"></div>
                        <div className="h-3 bg-surface-alt rounded-md w-1/3"></div>
                      </td>
                      <td className={selectedTaskId ? "px-4 py-4" : "px-6 py-5"}>
                        <div className="h-7 w-20 max-w-full rounded-full bg-surface-tertiary"></div>
                      </td>
                      {!selectedTaskId && (
                        <>
                          <td className="px-6 py-5">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 rounded-full bg-surface-tertiary"></div>
                              <div className="h-4 w-20 bg-surface-tertiary rounded-md"></div>
                            </div>
                          </td>
                          <td className="px-6 py-5 text-end"><div className="h-8 w-8 bg-surface-tertiary rounded-full ms-auto"></div></td>
                        </>
                      )}
                    </tr>
                  ))
                ) : (
                  filtered.map((task) => {
                    const isSelected = task.task_id === selectedTaskId;
                    const statusUpper = task.status?.toUpperCase() ?? "";
                    return (
                      <tr
                        key={task.task_id}
                        className={clsx(
                          "border-b border-line  transition-colors duration-200 cursor-pointer",
                          isSelected ? "bg-ink/[0.05]" : "hover:bg-ink/[0.03]"
                        )}
                        tabIndex={0}
                        aria-selected={isSelected}
                        onClick={() => { setSelectedTaskId(task.task_id); setDetailArticle(null); setDetailTab("content"); setContentView("reader"); setWpResult(null); }}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            setSelectedTaskId(task.task_id); setDetailArticle(null); setDetailTab("content"); setContentView("reader"); setWpResult(null);
                          }
                        }}
                      >
                        <td className={selectedTaskId ? "px-4 py-4" : "px-6 py-4"}>
                          <div className="flex min-w-0 items-start gap-2">
                            <div className="flex min-w-0 flex-1 flex-col">
                              <span className={clsx("truncate text-base font-semibold", isSelected ? "text-ink" : "text-ink")}>
                                {task.topic || task.task_name || task.task_id.slice(0, 12)}
                              </span>
                              {selectedTaskId && (
                                <span className="mt-1 truncate text-xs font-medium text-ink-muted">
                                  {formatDate(task.created_at, locale)}
                                </span>
                              )}
                            </div>
                            {selectedTaskId && (
                              <button
                                type="button"
                                onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(task.task_id); }}
                                className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-ink-muted transition-colors hover:bg-danger-subtle hover:text-danger"
                                title={t("common.delete")}
                                aria-label={t("common.delete")}
                              >
                                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                              </button>
                            )}
                          </div>
                        </td>
                        <td className={selectedTaskId ? "px-4 py-4 align-top" : "px-6 py-4"}><StatusBadge status={statusUpper} locale={locale} /></td>
                        {!selectedTaskId && (
                          <>
                            <td className="px-6 py-4 text-sm text-ink-muted font-medium">{formatDate(task.created_at, locale)}</td>
                            <td className="px-6 py-4 text-end">
                              <button
                                type="button"
                                onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(task.task_id); }}
                                className="w-8 h-8 inline-flex items-center justify-center rounded-full text-ink-muted hover:text-danger hover:bg-danger-subtle transition-colors"
                                title={t("common.delete")}
                                aria-label={t("common.delete")}
                              >
                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                              </button>
                            </td>
                          </>
                        )}
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
            )}
          </div>
        </div>

        {/* Detail: Slide-over Context Panel */}
        {selectedTaskId && (
          <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden border-s border-line bg-[rgb(var(--bg-secondary))] lg:sticky lg:top-0 lg:max-h-full">
            <div className="flex items-center justify-between border-b border-line p-5 lg:p-6">
              <h3 className="text-body-lg font-semibold text-ink">{t("tasks.detail")}</h3>
              <div className="flex gap-2">
                {streamActive && <span className="flex items-center gap-1.5 text-xs font-medium text-success"><span className="w-2 h-2 rounded-full bg-success animate-pulse" /> {TASK_COPY[locale].live}</span>}
                <button type="button" onClick={() => setSelectedTaskId(null)} aria-label={t("common.close")} className="flex h-8 w-8 items-center justify-center rounded-md text-ink-muted transition-colors hover:bg-ink/[0.05] hover:text-ink">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" /></svg>
                </button>
              </div>
            </div>

            <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-5 lg:p-6">
              {/* Status Block */}
              {liveStatus ? (
                <div className="space-y-4">
                  <div className="space-y-3 border-t border-line pt-4">
                    <div className="flex items-center justify-between">
                      <StatusBadge status={liveStatus.state} locale={locale} />
                      <button type="button" onClick={() => void navigator.clipboard.writeText(selectedTaskId)} className="text-xs text-ink-muted hover:text-brand font-mono transition-colors active:scale-95 flex items-center gap-1">
                        {t("tasks.copyId")}
                      </button>
                    </div>
                    {liveStatus.status && (
                      <p className="text-base text-ink-muted font-medium leading-relaxed">
                        {localizeTaskResult(liveStatus.status, locale)}
                      </p>
                    )}

                    {/* Progress bar if numerical */}
                    {typeof liveStatus.progress === "number" && liveStatus.progress > 0 && liveStatus.progress < 100 && (
                      <div className="h-1.5 w-full bg-surface-tertiary rounded-full overflow-hidden mt-2">
                        <div className="h-full bg-brand transition-all duration-500 ease-out" style={{ width: `${liveStatus.progress}%` }} />
                      </div>
                    )}
                  </div>

                  {/* Failure Trace */}
                  {liveStatus.state === "FAILURE" && (
                    <div className="rounded-lg border border-danger/25 border-s-4 border-s-danger bg-danger-subtle p-4 border-danger/20 bg-danger/10">
                      <h4 className="text-sm font-semibold text-danger">{TASK_COPY[locale].statuses.FAILURE}</h4>
                      <p className="mt-1 text-xs leading-5 text-danger">{TASK_COPY[locale].failedSummary}</p>
                      {liveStatus.quality_diagnostics && (
                        <section className="mt-4 border-t border-danger/70 pt-3 border-danger/20">
                          <h5 className="text-xs font-semibold text-danger">{TASK_COPY[locale].qualityDiagnostics}</h5>
                          <dl className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                            <DiagnosticItem label={TASK_COPY[locale].actualWordCount} value={formatDiagnosticNumber(liveStatus.quality_diagnostics.actual_word_count, locale)} />
                            <DiagnosticItem label={TASK_COPY[locale].allowedWordRange} value={formatWordRange(liveStatus.quality_diagnostics, locale)} />
                            <DiagnosticItem label={TASK_COPY[locale].headings} value={formatDiagnosticNumber(liveStatus.quality_diagnostics.headings_count, locale)} />
                            <DiagnosticItem label={TASK_COPY[locale].paragraphs} value={formatDiagnosticNumber(liveStatus.quality_diagnostics.paragraphs_count, locale)} />
                            <DiagnosticItem label={TASK_COPY[locale].language} value={formatDiagnosticLanguage(liveStatus.quality_diagnostics.language, locale)} />
                            <DiagnosticItem label={TASK_COPY[locale].regenerationAttempted} value={formatBoolean(liveStatus.quality_diagnostics.regeneration_attempted, locale)} />
                          </dl>
                          {liveStatus.quality_diagnostics.findings?.length ? (
                            <div className="mt-3 border-t border-danger/70 pt-3 border-danger/20">
                              <h6 className="text-xs font-semibold text-danger">{TASK_COPY[locale].findings}</h6>
                              <ul className="mt-2 space-y-1.5 text-xs leading-5 text-danger">
                                {liveStatus.quality_diagnostics.findings.map((finding, index) => (
                                  <li key={`${finding.code ?? "finding"}-${index}`}>
                                    {localizeQualityFinding(finding.code, finding.message, locale)}
                                    {formatQualityFindingActual(finding.code, liveStatus.quality_diagnostics, finding.actual, locale)}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          ) : null}
                        </section>
                      )}
                      <details className="mt-3 text-xs text-danger">
                        <summary className="cursor-pointer font-medium">{TASK_COPY[locale].technicalDetails}</summary>
                        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap font-mono" dir="ltr">
                          {liveStatus.error ?? liveStatus.last_error ?? (t("common.unexpectedError"))}
                        </pre>
                      </details>
                    </div>
                  )}

                  {/* Success Article Payload */}
                  {liveStatus.state === "SUCCESS" && detailArticle && (
                    <div className="space-y-6">
                      {/* Article Metadata Header */}
                      <div className="border-t border-line pt-4">
                        <h4 className="text-lg font-semibold text-ink leading-snug" dir="auto">
                          {detailArticle.title || "\u2014"}
                        </h4>
                        <div className="mt-2.5 flex flex-wrap gap-2">
                          {detailArticle.language && (
                            <span className="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-1 text-xs font-medium text-ink-secondary" dir="auto">
                              {TASK_COPY[locale].language}: {detailArticle.language}
                            </span>
                          )}
                          {detailArticle.primary_keyword && (
                            <span className="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-1 text-xs font-medium text-ink-secondary" dir="auto">
                              {TASK_COPY[locale].keyword}: {detailArticle.primary_keyword}
                            </span>
                          )}
                          {detailArticle.generated_at && (
                            <span className="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-1 text-xs font-medium text-ink-secondary">
                              {TASK_COPY[locale].generatedAt}: {formatDate(detailArticle.generated_at, locale)}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Metric Chips */}
                      <div className="grid grid-cols-3 divide-x divide-line border-y border-line">
                        <div className="px-3 py-4 text-center">
                          <span className="block text-xs font-medium text-ink-muted">{t("tasks.wordCount")}</span>
                          <span className="block text-xl font-bold text-ink mt-1">{detailArticle.word_count ?? "—"}</span>
                        </div>
                        <div className="px-3 py-4 text-center">
                          <span className="block text-xs font-medium text-ink-muted">{t("tasks.qualityScore")}</span>
                          {typeof taskQualityScore === "number" ? (
                            <>
                              <span className={clsx("block text-xl font-bold mt-1", qualityGrade(taskQualityScore, locale).color)}>{taskQualityScore}</span>
                              <span className={clsx("block text-xs font-semibold mt-0.5", qualityGrade(taskQualityScore, locale).color)}>{qualityGrade(taskQualityScore, locale).label}</span>
                            </>
                          ) : (
                            <span className="mt-2 block text-xs font-medium leading-5 text-ink-muted">{TASK_COPY[locale].notRecorded}</span>
                          )}
                        </div>
                        <div className="px-3 py-4 text-center">
                          <span className="block text-xs font-medium text-ink-muted">{t("tasks.cost")}</span>
                          {typeof taskCost === "number" ? (
                            <span className="block text-xl font-bold text-brand mt-1">${taskCost.toFixed(3)}</span>
                          ) : (
                            <span className="mt-2 block text-xs font-medium leading-5 text-ink-muted">{TASK_COPY[locale].notRecorded}</span>
                          )}
                        </div>
                      </div>

                      <ReviewPanel
                        canReview={canReview}
                        reviewState={reviewState}
                        loading={reviewLoading}
                        error={reviewError}
                        copy={reviewCopy}
                        onApprove={() => void submitReview("approve")}
                        onRequestChanges={() => {
                          setReviewAction("request_changes");
                          setReviewNote("");
                        }}
                        onReject={() => {
                          setReviewAction("reject");
                          setReviewNote("");
                        }}
                        submitting={reviewSubmitting}
                      />

                      {/* Inner Sub-Navigation (Segmented) */}
                      <div className="flex w-full border-b border-line">
                        {[
                          { key: "content" as DetailTab, label: t("tasks.contentTab") },
                          { key: "seo" as DetailTab, label: t("tasks.seoTab") },
                          { key: "export" as DetailTab, label: t("tasks.exportTab") },
                          { key: "history" as DetailTab, label: TASK_COPY[locale].history },
                        ].map((tab) => (
                          <button type="button" key={tab.key} onClick={() => setDetailTab(tab.key)} className={clsx("relative flex-1 px-2 py-2.5 text-sm font-medium transition-colors after:absolute after:inset-x-2 after:-bottom-px after:h-px", detailTab === tab.key ? "text-ink after:bg-brand" : "text-ink-muted after:bg-transparent hover:text-ink")}>
                            {tab.label}
                          </button>
                        ))}
                      </div>

                      {/* Content Views */}
                      {detailTab === "content" && (
                        <div className="flex flex-col gap-3">
                          <div className="mb-2 flex flex-wrap gap-2">
                            {(["reader", "raw", "edit"] as ContentView[]).map(cv => (
                              <button type="button" key={cv} onClick={() => setContentView(cv)} className={clsx("border-b px-2 py-1.5 text-xs font-medium transition-colors", contentView === cv ? "border-brand text-ink" : "border-transparent text-ink-muted hover:text-ink")}>
                                {cv === "reader"
                                  ? t("tasks.readerMode")
                                  : cv === "raw"
                                    ? sourceIsHtml
                                      ? TASK_COPY[locale].htmlSource
                                      : TASK_COPY[locale].markdownSource
                                    : t("tasks.editMode")}
                              </button>
                            ))}
                          </div>
                          <div className="rounded-xl border border-line bg-surface">
                            {contentView === "reader" && (
                              <article className="max-w-none whitespace-pre-wrap p-5 font-sans text-base leading-relaxed text-ink" dir={articleDirection}>
                                {toReaderText(sourceContent)}
                              </article>
                            )}
                            {contentView === "raw" && (
                              <pre className="max-h-96 overflow-auto rounded-lg border border-line bg-surface-alt p-4 font-mono text-xs leading-5 text-ink whitespace-pre-wrap select-all" dir={articleDirection}>{sourceContent}</pre>
                            )}
                            {contentView === "edit" && (
                              <div className="relative group">
                                <textarea aria-label={t("tasks.editMode")} className="w-full h-96 p-4 outline-none resize-none bg-transparent font-mono text-sm text-ink-muted leading-relaxed" dir="auto" value={editContent} onChange={(e) => setEditContent(e.target.value)} />
                                <div className="absolute bottom-4 end-4 flex gap-2 opacity-100 transition-opacity">
                                  {editContent !== detailArticle.content && (
                                    <span className="text-xs text-warning flex items-center font-medium bg-warning-subtle px-2 py-1 rounded-md">{TASK_COPY[locale].unsavedChanges}</span>
                                  )}
                                  <Button size="sm" variant="primary" loading={savingEdit} disabled={editContent === detailArticle.content} onClick={() => void handleSaveEdit()}>
                                    {TASK_COPY[locale].save}
                                  </Button>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {detailTab === "seo" && (() => {
                        const storedSeo = detailArticle.seo_analysis;
                        const seoScore = qualityMetrics?.seo_score?.score ?? storedSeo?.score;
                        const componentScores = qualityMetrics?.seo_score?.component_scores ?? {};
                        const recommendations = qualityMetrics?.seo_score?.recommendations ?? storedSeo?.recommendations ?? [];
                        const checklist = storedSeo?.checklist ?? [];
                        const hasSeoData =
                          typeof seoScore === "number" ||
                          Object.keys(componentScores).length > 0 ||
                          recommendations.length > 0 ||
                          checklist.length > 0 ||
                          qualityMetrics?.readability_grade;

                        return (
                          <div className="space-y-4">
                            <section className="border-t border-line pt-4">
                              <div className="flex flex-wrap items-start justify-between gap-3">
                                <div>
                                  <h4 className="text-base font-bold text-ink">{t("tasks.seoTab")}</h4>
                                  <p className="mt-1 text-xs text-ink-muted">
                                    {qualityLoading
                                      ? TASK_COPY[locale].seoLoading
                                      : qualityMetrics?.readability_grade
                                        ? qualityMetrics.readability_grade
                                        : seoFallback}
                                  </p>
                                </div>
                                <span className="rounded-lg border border-line bg-surface-alt px-3 py-2 text-xl font-bold tabular-nums text-ink">
                                  {formatPercentScore(seoScore)}
                                </span>
                              </div>
                            </section>

                            {qualityError && (
                              <div className="rounded-xl border border-danger/20 bg-danger/10 px-4 py-3 text-sm font-medium text-danger">
                                <p>{TASK_COPY[locale].seoError}</p>
                                <details className="mt-2 text-xs opacity-80">
                                  <summary className="cursor-pointer">{TASK_COPY[locale].technicalDetails}</summary>
                                  <p className="mt-1 break-words" dir="auto">{qualityError}</p>
                                </details>
                              </div>
                            )}

                            {Object.keys(componentScores).length > 0 && (
                              <section className="border-t border-line pt-4">
                                <h5 className="mb-3 text-sm font-bold text-ink">{t("tasks.seoScore")}</h5>
                                <div className="grid gap-2">
                                  {Object.entries(componentScores).map(([key, value]) => (
                                    <div key={key} className="flex items-center justify-between gap-3 rounded-lg bg-surface-alt px-3 py-2">
                                      <span className="text-xs font-medium text-ink-secondary">{humanizeMetricKey(key)}</span>
                                      <span className="text-xs font-bold tabular-nums text-ink">{formatPercentScore(value)}</span>
                                    </div>
                                  ))}
                                </div>
                              </section>
                            )}

                            {checklist.length > 0 && (
                              <section className="border-t border-line pt-4">
                                <h5 className="mb-3 text-sm font-bold text-ink">{t("tasks.seoChecklist")}</h5>
                                <div className="space-y-2">
                                  {checklist.map((item) => (
                                    <div key={item.label} className="flex items-start gap-2 text-xs leading-5 text-ink-secondary">
                                      <span className={clsx("mt-1 h-2 w-2 shrink-0 rounded-full", item.passed ? "bg-success" : "bg-warning")} />
                                      <span>
                                        <span className="font-semibold text-ink">{item.label}</span>
                                        {item.detail ? `: ${item.detail}` : ""}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </section>
                            )}

                            {recommendations.length > 0 && (
                              <section className="border-t border-line pt-4">
                                <h5 className="mb-3 text-sm font-bold text-ink">{t("tasks.recommendations")}</h5>
                                <div className="space-y-2">
                                  {recommendations.map((recommendation) => (
                                    <p key={recommendation} className="rounded-lg bg-warning/10 px-3 py-2 text-xs leading-5 text-warning">
                                      {recommendation}
                                    </p>
                                  ))}
                                </div>
                              </section>
                            )}

                            {!qualityLoading && !qualityError && !hasSeoData && (
                              <div className="rounded-lg border border-line bg-surface p-5 text-center text-sm text-ink-muted">
                                {seoFallback}
                              </div>
                            )}
                          </div>
                        );
                      })()}
                      {detailTab === "export" && (
                        <div className="space-y-4">
                          <div className="border-t border-line pt-4">
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <h4 className="text-base font-bold text-ink">{riskCopy.title}</h4>
                                <p className="mt-1 text-xs text-ink-muted">
                                  {riskLoading
                                    ? riskCopy.loading
                                    : riskAssessment
                                      ? `${riskAssessment.overall_score}/100`
                                      : t("monitoring.statusUnknown")}
                                </p>
                              </div>
                              {riskAssessment && (
                                <span
                                  className={clsx(
                                    "inline-flex h-7 items-center rounded-md px-2.5 text-xs font-bold",
                                    riskAssessment.risk_level === "blocked" || riskAssessment.risk_level === "high"
                                      ? "bg-danger/10 text-danger"
                                      : riskAssessment.risk_level === "medium"
                                        ? "bg-warning/10 text-warning"
                                        : "bg-success/10 text-success"
                                  )}
                                >
                                  {riskAssessment.risk_level === "blocked"
                                    ? riskCopy.blocked
                                    : riskAssessment.risk_level === "high"
                                      ? riskCopy.high
                                      : riskAssessment.risk_level === "medium"
                                        ? riskCopy.medium
                                        : riskCopy.low}
                                </span>
                              )}
                            </div>
                            {riskAssessment?.issues?.length ? (
                              <div className="mt-3 space-y-2">
                                {riskAssessment.issues.slice(0, 3).map((issue) => (
                                  <p key={issue.id} className="text-xs leading-5 text-ink-secondary">
                                    <span className="font-semibold text-ink">{localizeRiskCategory(issue.category, locale)}: </span>
                                    {localizeRiskMessage(issue.message, locale)}
                                  </p>
                                ))}
                              </div>
                            ) : null}
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <Button variant="outlined" onClick={() => downloadContent(detailArticle, "txt")}>
                              <svg className="w-4 h-4 me-2 inline shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                              {t("tasks.downloadTxt")}
                            </Button>
                            <Button
                              variant="outlined"
                              onClick={() => downloadContent(detailArticle, sourceIsHtml ? "html" : "markdown")}
                            >
                              <svg className="w-4 h-4 me-2 inline shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                              {sourceIsHtml ? TASK_COPY[locale].downloadHtml : TASK_COPY[locale].downloadMarkdown}
                            </Button>
                            <Button variant="outlined" onClick={() => void navigator.clipboard.writeText(contentView === "edit" ? editContent : detailArticle.content)} className="col-span-2">
                              <svg className="w-4 h-4 me-2 inline shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" /></svg>
                              {t("tasks.copyContent")}
                            </Button>
                          </div>
                          <div className="border-t border-line pt-5">
                            <h4 className="text-base font-bold text-info mb-3">{t("tasks.wpPublish")}</h4>
                            <div className="flex gap-3">
                              <Button variant="outlined" size="sm" loading={wpPublishing} disabled={riskAssessment?.risk_level === "blocked" || wordpressActionBlocked} onClick={() => void onWpPublish("draft")}>{t("tasks.wpDraft")}</Button>
                              <Button
                                variant={publicPublishBlocked ? "outlined" : "primary"}
                                size="sm"
                                loading={wpPublishing}
                                disabled={publicPublishBlocked}
                                onClick={() => void onWpPublish("publish")}
                                className={publicPublishBlocked ? "border-warning/25 bg-warning-subtle text-warning shadow-none hover:bg-warning-subtle border-warning/20 bg-warning/10 text-warning" : undefined}
                              >
                                {publicPublishLabel}
                              </Button>
                            </div>
                            {publicPublishReason && (
                              <p className="mt-3 text-xs font-medium text-warning">
                                {publicPublishReason}
                              </p>
                            )}
                            {wordpressActionReason && wordpressActionReason !== publicPublishReason && (
                              <p className="mt-2 text-xs font-medium text-warning">
                                {wordpressActionReason}
                              </p>
                            )}
                            {wpResult && <p className={clsx("mt-3 text-xs font-medium", wpResult.includes("error") || wpResult.includes("خطا") ? "text-danger" : "text-info")}>{wpResult}</p>}
                          </div>
                        </div>
                      )}
                      {detailTab === "history" && (
                        <div className="space-y-4">
                          <h4 className="text-base font-bold text-ink">{TASK_COPY[locale].revisionHistory}</h4>
                          {loadingHistory ? (
                            <div className="animate-pulse space-y-3">
                              <div className="h-16 w-full rounded-xl bg-surface-tertiary" />
                              <div className="h-16 w-full rounded-xl bg-surface-tertiary" />
                            </div>
                          ) : !articleHistory?.revisions?.length ? (
                            <p className="px-4 py-10 text-center text-sm text-ink-muted">{TASK_COPY[locale].noRevisions}</p>
                          ) : (
                            <div className="space-y-3">
                              {articleHistory.revisions.map((rev) => (
                                <div key={rev.id} className="border-t border-line pt-4">
                                  <div className="flex justify-between items-start mb-2">
                                    <div className="text-sm font-medium text-ink">
                                      {rev.revision_note || TASK_COPY[locale].manualEdit}
                                    </div>
                                    <div className="text-xs text-ink-muted">
                                      {formatDate(rev.created_at, locale)}
                                    </div>
                                  </div>
                                  <div className="text-xs text-ink-muted mb-3">
                                    {formatDiagnosticNumber(rev.word_count, locale)} {TASK_COPY[locale].wordUnit}
                                  </div>
                                  <details className="text-xs">
                                    <summary className="cursor-pointer text-brand font-medium select-none">{TASK_COPY[locale].viewPastContent}</summary>
                                    <div className="mt-3 max-h-64 overflow-auto rounded-lg bg-surface-alt p-3 font-mono text-xs text-ink-secondary" dir="auto">
                                      {rev.content}
                                    </div>
                                  </details>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-4 animate-pulse">
                  <div className="h-24 w-full rounded-xl bg-surface-tertiary" />
                  <div className="h-64 w-full rounded-xl bg-surface-tertiary" />
                  <p className="text-center text-xs font-medium text-ink-muted">{TASK_COPY[locale].loadingDetail}</p>
                </div>
              )}
            </div>
          </aside>
        )}
      </div>

    </section>
  );
}

/* ─── Helper Components ─── */
function DiagnosticItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-b border-danger/60 pb-2 last:border-b-0 border-danger/15">
      <dt className="text-xs font-medium text-danger/80">{label}</dt>
      <dd className="mt-0.5 text-xs font-semibold text-danger" dir="auto">{value}</dd>
    </div>
  );
}

function StatusBadge({ status, locale }: { status: string; locale: TaskLocale }) {
  const s = status.toUpperCase();
  const cls = s === "SUCCESS"
    ? "border-success/60 bg-success-subtle text-success border-success/30 bg-success/[0.12] text-success"
    : ["FAILURE", "FAILED"].includes(s)
      ? "border-danger/60 bg-danger-subtle text-danger border-danger/30 bg-danger/[0.12] text-danger"
      : "border-brand/60 bg-brand-light text-brand animate-pulse-soft border-brand/30 bg-brand/[0.12] text-brand";

  return (
    <span className={clsx("inline-flex items-center justify-center rounded-lg border px-2.5 py-1 text-xs font-bold uppercase tracking-wider", cls)}>
      {localizeTaskStatus(status, locale)}
    </span>
  );
}

function ReviewPanel({
  canReview,
  reviewState,
  loading,
  error,
  copy,
  onApprove,
  onRequestChanges,
  onReject,
  submitting,
}: {
  canReview: boolean;
  reviewState: ArticleReviewState | null;
  loading: boolean;
  error: string | null;
  copy: ReviewCopy;
  onApprove: () => void;
  onRequestChanges: () => void;
  onReject: () => void;
  submitting: boolean;
}) {
  if (loading) {
    return (
      <section className="border-t border-line pt-4">
        <div className="h-4 w-28 rounded bg-surface-tertiary" />
        <div className="mt-4 grid gap-2">
          <div className="h-8 rounded-lg bg-surface-tertiary" />
          <div className="h-8 rounded-lg bg-surface-tertiary" />
        </div>
      </section>
    );
  }

  if (error || !reviewState) {
    return (
      <section className="rounded-xl border border-warning/20 bg-warning/10 p-4 text-sm font-medium text-warning">
        {error || copy.unavailable}
      </section>
    );
  }

  const statusLabel = reviewLabel(reviewState.status, copy);
  const approveBlocked = !reviewState.can_approve;

  return (
    <section className="border-t border-line pt-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h4 className="text-base font-bold text-ink">{copy.title}</h4>
            <UiStatusBadge variant={reviewVariant(reviewState.status)} dot={false}>
              {statusLabel}
            </UiStatusBadge>
          </div>
          <p className="mt-1 text-xs leading-5 text-ink-muted">{copy.subtitle}</p>
        </div>
        <div className="text-end text-xs text-ink-muted">
          {reviewState.reviewer_name ? (
            <>
              <span className="block font-medium text-ink-muted">{copy.reviewedBy}</span>
              <span>{reviewState.reviewer_name}</span>
            </>
          ) : (
            <span>{copy.noDecision}</span>
          )}
        </div>
      </div>

      {reviewState.note && (
        <p className="mt-3 rounded-lg border border-line bg-surface-alt px-3 py-2 text-xs leading-5 text-ink-secondary">
          {reviewState.note}
        </p>
      )}

      <div className="mt-4">
        <div className="mb-2 flex items-center justify-between">
          <h5 className="text-xs font-bold text-ink">{copy.checklist}</h5>
          {approveBlocked && (
            <span className="text-xs font-semibold text-warning">
              {copy.blockedTitle}
            </span>
          )}
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          {reviewState.checklist.map((item) => (
            <div
              key={item.id}
              className="flex items-center gap-2 rounded-lg border border-line bg-surface-alt px-3 py-2"
            >
              <span
                className={clsx(
                  "flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs font-bold",
                  item.passed
                    ? "bg-success/12 text-success"
                    : item.blocking
                      ? "bg-danger/12 text-danger"
                      : "bg-warning/12 text-warning"
                )}
              >
                {item.passed ? (
                  <svg className="h-3 w-3" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden><path d="m3.4 8.1 2.7 2.7 6.5-6.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
                ) : (
                  <svg className="h-3 w-3" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden><path d="M8 4.4v4.4M8 11.4h.01" strokeLinecap="round" /></svg>
                )}
              </span>
              <span className="min-w-0 text-xs font-medium text-ink-muted">
                {copy.checks[item.id as keyof ReviewCopy["checks"]] ?? item.label}
              </span>
            </div>
          ))}
        </div>
        {approveBlocked && (
          <p className="mt-2 text-xs leading-5 text-warning">
            {copy.approveBlocked}
          </p>
        )}
      </div>

      {canReview && (
        <div className="mt-4 flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="primary"
            loading={submitting}
            disabled={submitting || approveBlocked || reviewState.status === "approved"}
            onClick={onApprove}
          >
            {copy.approve}
          </Button>
          <Button
            size="sm"
            variant="outlined"
            disabled={submitting}
            onClick={onRequestChanges}
          >
            {copy.request_changes}
          </Button>
          <Button
            size="sm"
            variant="danger"
            disabled={submitting}
            onClick={onReject}
          >
            {copy.reject}
          </Button>
        </div>
      )}
    </section>
  );
}

function reviewVariant(status: string) {
  if (status === "approved") return "success";
  if (status === "rejected") return "error";
  if (status === "changes_requested") return "warning";
  return "neutral";
}

function reviewLabel(status: string, copy: ReviewCopy) {
  if (status === "approved") return copy.approved;
  if (status === "rejected") return copy.rejected;
  if (status === "changes_requested") return copy.changes_requested;
  return copy.pending_review;
}

/* ─── Helper Functions ─── */
function localizeTaskStatus(status: string, locale: TaskLocale): string {
  const normalized = status.trim().toUpperCase() as keyof typeof TASK_COPY.en.statuses;
  return TASK_COPY[locale].statuses[normalized]
    ?? status.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function localizeTaskResult(message: string, locale: TaskLocale): string {
  const normalized = message.trim().toLowerCase().replace(/[.!]+$/, "");
  if (normalized === "task completed successfully") {
    return TASK_COPY[locale].completed;
  }
  if (normalized === "task failed") {
    return TASK_COPY[locale].statuses.FAILURE;
  }
  return message;
}

function formatDiagnosticNumber(value: number | undefined, locale: TaskLocale): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return TASK_COPY[locale].notRecorded;
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  return new Intl.NumberFormat(localeName).format(value);
}

function formatWordRange(
  diagnostics: NonNullable<TaskStatusResponse["quality_diagnostics"]>,
  locale: TaskLocale,
): string {
  const { min_word_count: minimum, max_word_count: maximum } = diagnostics;
  if (typeof minimum !== "number" || typeof maximum !== "number") return TASK_COPY[locale].notRecorded;
  return `${formatDiagnosticNumber(minimum, locale)}–${formatDiagnosticNumber(maximum, locale)}`;
}

function formatDiagnosticLanguage(language: string | undefined, locale: TaskLocale): string {
  const normalized = language?.trim().toLowerCase();
  if (!normalized) return TASK_COPY[locale].notRecorded;
  const labels = {
    en: { fa: "Persian", ar: "Arabic", en: "English" },
    fa: { fa: "فارسی", ar: "عربی", en: "انگلیسی" },
    ar: { fa: "الفارسية", ar: "العربية", en: "الإنجليزية" },
  } as const;
  return labels[locale][normalized as keyof typeof labels.en] ?? language;
}

function formatBoolean(value: boolean | undefined, locale: TaskLocale): string {
  if (typeof value !== "boolean") return TASK_COPY[locale].notRecorded;
  return locale === "fa" ? (value ? "بله" : "خیر") : locale === "ar" ? (value ? "نعم" : "لا") : value ? "Yes" : "No";
}

function formatQualityFindingActual(
  code: string | undefined,
  diagnostics: TaskStatusResponse["quality_diagnostics"],
  fallback: string | undefined,
  locale: TaskLocale,
): string {
  const units = {
    en: { words: "words", headings: "headings", paragraphs: "paragraphs" },
    fa: { words: "کلمه", headings: "تیتر", paragraphs: "پاراگراف" },
    ar: { words: "كلمة", headings: "عناوين", paragraphs: "فقرات" },
  } as const;

  if (code === "word_count_below_minimum" || code === "word_count_above_maximum") {
    return typeof diagnostics?.actual_word_count === "number"
      ? `: ${formatDiagnosticNumber(diagnostics.actual_word_count, locale)} ${units[locale].words}`
      : fallback ? `: ${fallback}` : "";
  }
  if (code === "insufficient_headings") {
    return typeof diagnostics?.headings_count === "number"
      ? `: ${formatDiagnosticNumber(diagnostics.headings_count, locale)} ${units[locale].headings}`
      : fallback ? `: ${fallback}` : "";
  }
  if (code === "insufficient_paragraphs") {
    return typeof diagnostics?.paragraphs_count === "number"
      ? `: ${formatDiagnosticNumber(diagnostics.paragraphs_count, locale)} ${units[locale].paragraphs}`
      : fallback ? `: ${fallback}` : "";
  }
  return fallback ? `: ${fallback}` : "";
}

function localizeQualityFinding(code: string | undefined, fallback: string | undefined, locale: TaskLocale): string {
  const messages = {
    en: {
      word_count_below_minimum: "Article is shorter than the required minimum",
      word_count_above_maximum: "Article exceeds the allowed maximum",
      insufficient_headings: "Article does not have enough structural headings",
      insufficient_paragraphs: "Article does not have enough readable paragraphs",
      duplicate_adjacent_headings: "Consecutive duplicate headings were detected",
      missing_required_faq: "The requested FAQ section is missing",
      incomplete_required_faq: "The requested FAQ section needs more answered questions",
    },
    fa: {
      word_count_below_minimum: "مقاله از حداقل طول لازم کوتاه‌تر است",
      word_count_above_maximum: "مقاله از حداکثر طول مجاز بیشتر است",
      insufficient_headings: "مقاله تیترهای ساختاری کافی ندارد",
      insufficient_paragraphs: "مقاله پاراگراف‌های خوانای کافی ندارد",
      duplicate_adjacent_headings: "تیترهای تکراری پیاپی در مقاله شناسایی شد",
      missing_required_faq: "بخش پرسش‌های متداول درخواستی وجود ندارد",
      incomplete_required_faq: "بخش پرسش‌های متداول به پرسش‌های پاسخ‌داده‌شده بیشتری نیاز دارد",
    },
    ar: {
      word_count_below_minimum: "المقالة أقصر من الحد الأدنى المطلوب",
      word_count_above_maximum: "المقالة تتجاوز الحد الأقصى المسموح",
      insufficient_headings: "لا تحتوي المقالة على عناوين هيكلية كافية",
      insufficient_paragraphs: "لا تحتوي المقالة على فقرات مقروءة كافية",
      duplicate_adjacent_headings: "تم اكتشاف عناوين متتالية مكررة",
      missing_required_faq: "قسم الأسئلة الشائعة المطلوب غير موجود",
      incomplete_required_faq: "يحتاج قسم الأسئلة الشائعة إلى مزيد من الأسئلة المجابة",
    },
  } as const;
  return messages[locale][code as keyof typeof messages.en] ?? fallback ?? TASK_COPY[locale].notRecorded;
}

function formatPublishResult(value: unknown, fallback: string): string {
  if (typeof value === "string") {
    return value.trim() && value !== "[object Object]" ? value : fallback;
  }
  if (typeof value === "object" && value !== null) {
    const record = value as Record<string, unknown>;
    return formatPublishResult(record.message ?? record.label ?? record.detail, fallback);
  }
  return fallback;
}

function localizeRiskCategory(category: string, locale: TaskLocale): string {
  return category.trim().toLowerCase() === "seo" ? TASK_COPY[locale].seo : category;
}

function localizeRiskMessage(message: string, locale: TaskLocale): string {
  if (message.toLowerCase().includes("no faq section was detected")) {
    return TASK_COPY[locale].noFaq;
  }
  return message;
}

function resolveArticleDirection(language: string | undefined, content: string): "ltr" | "rtl" | "auto" {
  const normalized = language?.trim().toLowerCase();
  if (normalized && /^(fa|fa-ir|persian|farsi|ar|ar-)/.test(normalized)) return "rtl";
  if (normalized && /^(en|en-|english)/.test(normalized)) return "ltr";
  if (/[\u0600-\u06FF]/.test(content.slice(0, 240))) return "rtl";
  if (/[A-Za-z]/.test(content.slice(0, 240))) return "ltr";
  return "auto";
}

function formatDate(d: string | undefined, locale: TaskLocale): string {
  if (!d) return "—";
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  try {
    return new Intl.DateTimeFormat(localeName, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(d));
  }
  catch { return d; }
}

function formatPercentScore(value?: number): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const normalized = value <= 1 ? value * 100 : value;
  return `${Math.round(normalized)}%`;
}

function readFiniteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function qualityGrade(score: number | undefined, locale: TaskLocale): { label: string; color: string } {
  if (typeof score !== "number") return { label: "—", color: "text-ink-muted" };
  if (score >= 80) return { label: TASK_COPY[locale].gradeExcellent, color: "text-success" };
  if (score >= 65) return { label: TASK_COPY[locale].gradeGood, color: "text-brand text-brand" };
  if (score >= 50) return { label: TASK_COPY[locale].gradeFair, color: "text-warning" };
  return { label: TASK_COPY[locale].gradeNeedsWork, color: "text-danger" };
}

function humanizeMetricKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function toReaderText(content?: string): string {
  if (!content) return "";
  return content
    .replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?>[\s\S]*?<\/style>/gi, "")
    .replace(/<\/(p|div|h[1-6]|li|ul|ol|blockquote|br)>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, "\"")
    .replace(/&#39;/g, "'")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function downloadContent(article: ArticleDetail, format: "txt" | "html" | "markdown") {
  const content = format === "html" ? (article.html_content ?? "") : article.content;
  const extension = format === "markdown" ? "md" : format;
  const type = format === "html"
    ? "text/html;charset=utf-8"
    : format === "markdown"
      ? "text/markdown;charset=utf-8"
      : "text/plain;charset=utf-8";
  const blob = new Blob([content], { type });
  downloadBlob(blob, `${article.title || "article"}.${extension}`);
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}
