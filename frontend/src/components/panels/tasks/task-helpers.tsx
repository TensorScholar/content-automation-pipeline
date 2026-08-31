"use client";

import clsx from "clsx";
import { ArticleDetail, TaskStatusResponse, ProjectReadiness, ArticleReviewState, ArticleReviewAction, DraftRiskAssessment } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { StatusBadge as UiStatusBadge } from "@/components/ui/status-badge";
import { useToast } from "@/components/ui/toast";
import { ApiError } from "@/lib/api";
import type { TaskLocale } from "./task-constants";
import { TASK_COPY, RISK_COPY, REVIEW_COPY, PUBLISH_COPY } from "./task-constants";

type ReviewCopy = (typeof REVIEW_COPY)["en"];

export function isWordPressPublishReadinessItem(item: { id: string; label: string; message: string; remediation?: string | null }): boolean {
  const text = `${item.id} ${item.label} ${item.message} ${item.remediation ?? ""}`.toLowerCase();
  return /\bwordpress\b|وردپرس|ووردبريس/.test(text);
}

export function DiagnosticItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-b border-danger/60 pb-2 last:border-b-0 border-danger/15">
      <dt className="text-xs font-medium text-danger/80">{label}</dt>
      <dd className="mt-0.5 text-xs font-semibold text-danger" dir="auto">{value}</dd>
    </div>
  );
}

export function StatusBadge({ status, locale }: { status: string; locale: TaskLocale }) {
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

export function ReviewPanel({
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

export function reviewVariant(status: string) {
  if (status === "approved") return "success";
  if (status === "rejected") return "error";
  if (status === "changes_requested") return "warning";
  return "neutral";
}

export function reviewLabel(status: string, copy: ReviewCopy) {
  if (status === "approved") return copy.approved;
  if (status === "rejected") return copy.rejected;
  if (status === "changes_requested") return copy.changes_requested;
  return copy.pending_review;
}

/* ─── Helper Functions ─── */
export function localizeTaskStatus(status: string, locale: TaskLocale): string {
  const normalized = status.trim().toUpperCase() as keyof typeof TASK_COPY.en.statuses;
  return TASK_COPY[locale].statuses[normalized]
    ?? status.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function localizeTaskResult(message: string, locale: TaskLocale): string {
  const normalized = message.trim().toLowerCase().replace(/[.!]+$/, "");
  if (normalized === "task completed successfully") {
    return TASK_COPY[locale].completed;
  }
  if (normalized === "task failed") {
    return TASK_COPY[locale].statuses.FAILURE;
  }
  return message;
}

export function formatDiagnosticNumber(value: number | undefined, locale: TaskLocale): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return TASK_COPY[locale].notRecorded;
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  return new Intl.NumberFormat(localeName).format(value);
}

export function formatWordRange(
  diagnostics: NonNullable<TaskStatusResponse["quality_diagnostics"]>,
  locale: TaskLocale,
): string {
  const { min_word_count: minimum, max_word_count: maximum } = diagnostics;
  if (typeof minimum !== "number" || typeof maximum !== "number") return TASK_COPY[locale].notRecorded;
  return `${formatDiagnosticNumber(minimum, locale)}–${formatDiagnosticNumber(maximum, locale)}`;
}

export function formatDiagnosticLanguage(language: string | undefined, locale: TaskLocale): string {
  const normalized = language?.trim().toLowerCase();
  if (!normalized) return TASK_COPY[locale].notRecorded;
  const labels = {
    en: { fa: "Persian", ar: "Arabic", en: "English" },
    fa: { fa: "فارسی", ar: "عربی", en: "انگلیسی" },
    ar: { fa: "الفارسية", ar: "العربية", en: "الإنجليزية" },
  } as const;
  return labels[locale][normalized as keyof typeof labels.en] ?? language;
}

export function formatBoolean(value: boolean | undefined, locale: TaskLocale): string {
  if (typeof value !== "boolean") return TASK_COPY[locale].notRecorded;
  return locale === "fa" ? (value ? "بله" : "خیر") : locale === "ar" ? (value ? "نعم" : "لا") : value ? "Yes" : "No";
}

export function formatQualityFindingActual(
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

export function localizeQualityFinding(code: string | undefined, fallback: string | undefined, locale: TaskLocale): string {
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

export function formatPublishResult(value: unknown, fallback: string): string {
  if (typeof value === "string") {
    return value.trim() && value !== "[object Object]" ? value : fallback;
  }
  if (typeof value === "object" && value !== null) {
    const record = value as Record<string, unknown>;
    return formatPublishResult(record.message ?? record.label ?? record.detail, fallback);
  }
  return fallback;
}

export function localizeRiskCategory(category: string, locale: TaskLocale): string {
  return category.trim().toLowerCase() === "seo" ? TASK_COPY[locale].seo : category;
}

export function localizeRiskMessage(message: string, locale: TaskLocale): string {
  if (message.toLowerCase().includes("no faq section was detected")) {
    return TASK_COPY[locale].noFaq;
  }
  return message;
}

export function resolveArticleDirection(language: string | undefined, content: string): "ltr" | "rtl" | "auto" {
  const normalized = language?.trim().toLowerCase();
  if (normalized && /^(fa|fa-ir|persian|farsi|ar|ar-)/.test(normalized)) return "rtl";
  if (normalized && /^(en|en-|english)/.test(normalized)) return "ltr";
  if (/[\u0600-\u06FF]/.test(content.slice(0, 240))) return "rtl";
  if (/[A-Za-z]/.test(content.slice(0, 240))) return "ltr";
  return "auto";
}

export function formatDate(d: string | undefined, locale: TaskLocale): string {
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

export function formatPercentScore(value?: number): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const normalized = value <= 1 ? value * 100 : value;
  return `${Math.round(normalized)}%`;
}

export function readFiniteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function qualityGrade(score: number | undefined, locale: TaskLocale): { label: string; color: string } {
  if (typeof score !== "number") return { label: "—", color: "text-ink-muted" };
  if (score >= 80) return { label: TASK_COPY[locale].gradeExcellent, color: "text-success" };
  if (score >= 65) return { label: TASK_COPY[locale].gradeGood, color: "text-brand text-brand" };
  if (score >= 50) return { label: TASK_COPY[locale].gradeFair, color: "text-warning" };
  return { label: TASK_COPY[locale].gradeNeedsWork, color: "text-danger" };
}

export function humanizeMetricKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function toReaderText(content?: string): string {
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

export function downloadContent(article: ArticleDetail, format: "txt" | "html" | "markdown") {
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

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}
