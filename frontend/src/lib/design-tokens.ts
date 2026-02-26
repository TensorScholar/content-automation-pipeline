/**
 * Smarlux Content OS — Design Token Constants
 *
 * Source of truth: globals.css (CSS custom properties) + tailwind.config.ts
 * Exports typed constants for JS/TS logic (charts, canvas, inline styles).
 */

/* ── Brand Colors (hex for JS contexts) ──────────────────────────── */

export const COLORS = {
  brand: "#0E6E6E",
  brandHover: "#0A5858",
  brandLight: "#E8F5F5",
  brandAccent: "#1ABC9C",

  textPrimary: "#111827",
  textSecondary: "#6B7280",
  textPlaceholder: "#9CA3AF",

  surface: "#FFFFFF",
  surfaceAlt: "#F7F9FB",
  border: "#E4E8ED",

  error: "#DC2626",
  warning: "#D97706",
  success: "#16A34A",
  info: "#2563EB",
} as const;

/* ── 4px Base Grid ───────────────────────────────────────────────── */

export const GRID = 4; // px

/* ── Breakpoints (matches Tailwind defaults) ─────────────────────── */

export const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  "2xl": 1536,
} as const;

/* ── Z-Index Semantic Scale ──────────────────────────────────────── */

export const Z_INDEX = {
  dropdown: 10,
  sticky: 20,
  overlay: 30,
  modal: 40,
  popover: 50,
  toast: 60,
  tooltip: 70,
} as const;

/* ── Motion ──────────────────────────────────────────────────────── */

export const DURATION = {
  fast: 100,
  base: 150,
  normal: 200,
  slow: 300,
  slower: 500,
} as const;

export const EASING = {
  default: "ease-in-out",
  apple: "cubic-bezier(0.25, 0.1, 0.25, 1)",
  spring: "cubic-bezier(0.175, 0.885, 0.32, 1.275)",
} as const;

/* ── Border Radius ───────────────────────────────────────────────── */

export const RADIUS = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  full: 9999,
} as const;

/* ── Layout ──────────────────────────────────────────────────────── */

export const LAYOUT = {
  sidebarWidth: 272,
  sidebarCollapsedWidth: 72,
  headerHeight: 64,
  maxContentWidth: 1152, // 72rem
} as const;

/* ── Type helpers ────────────────────────────────────────────────── */

export type Elevation = 1 | 2 | 3 | 4;
export type BreakpointKey = keyof typeof BREAKPOINTS;
