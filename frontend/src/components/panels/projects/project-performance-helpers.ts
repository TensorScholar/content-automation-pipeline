"use client";

import { PERFORMANCE_COPY } from "./project-constants";
import type { ReadinessLocale } from "./project-constants";
import { PerformanceOpportunity } from "@/types/models";

export function performanceSeverityClasses(severity: string) {
  if (severity === "high") {
    return "border-danger/20 bg-danger/10 text-danger";
  }
  if (severity === "medium") {
    return "border-warning/20 bg-warning/10 text-warning";
  }
  return "border-info/20 bg-info-subtle text-info";
}

export function performanceTypeLabel(copy: typeof PERFORMANCE_COPY.en, type: string) {
  const key = type as keyof typeof PERFORMANCE_COPY.en.types;
  return copy.types[key] ?? type.replaceAll("_", " ");
}

export function formatCompactNumber(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function formatFixedNumber(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

export function formatCtr(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${formatFixedNumber(value * 100, 2)}%`;
}

export function formatShortDate(value: string | null | undefined, locale: ReadinessLocale) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  return date.toLocaleDateString(localeName, { month: "short", day: "numeric", year: "numeric" });
}

export function metricFromOpportunity(
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

