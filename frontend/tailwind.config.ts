import type { Config } from "tailwindcss";

/* ═══════════════════════════════════════════════════════════════════
   SMARLUX CONTENT OS — Design Token System (Tailwind)
   Spec: Smarlux_UI_Master_Redesign_Prompt.md
   Deep Teal brand · 4px grid · Plus Jakarta Sans + Vazirmatn
   ═══════════════════════════════════════════════════════════════════ */

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {

      /* ─── Color System ─────────────────────────────────────── */
      colors: {
        /* Brand */
        brand: {
          DEFAULT: "rgb(var(--color-brand) / <alpha-value>)",
          hover: "rgb(var(--color-brand-hover) / <alpha-value>)",
          light: "rgb(var(--color-brand-light) / <alpha-value>)",
          accent: "rgb(var(--color-brand-accent) / <alpha-value>)",
        },
        /* Legacy aliases — keep for backward compat during migration */
        accent: {
          DEFAULT: "rgb(var(--color-brand) / <alpha-value>)",
          hover: "rgb(var(--color-brand-hover) / <alpha-value>)",
          subtle: "rgb(var(--color-brand-light) / <alpha-value>)",
        },
        /* Surfaces */
        surface: {
          DEFAULT: "rgb(var(--color-surface) / <alpha-value>)",
          alt: "rgb(var(--color-surface-alt) / <alpha-value>)",
          secondary: "rgb(var(--color-surface-alt) / <alpha-value>)",
          tertiary: "rgb(var(--color-surface-tertiary) / <alpha-value>)",
          elevated: "rgb(var(--color-surface) / <alpha-value>)",
          sunken: "rgb(var(--color-surface-tertiary) / <alpha-value>)",
        },
        /* Text */
        ink: {
          DEFAULT: "rgb(var(--color-text-primary) / <alpha-value>)",
          secondary: "rgb(var(--color-text-secondary) / <alpha-value>)",
          tertiary: "rgb(var(--color-text-placeholder) / <alpha-value>)",
          inverse: "rgb(255 255 255 / <alpha-value>)",
        },
        /* Borders */
        border: {
          DEFAULT: "rgb(var(--color-border) / <alpha-value>)",
          secondary: "rgb(var(--color-border) / <alpha-value>)",
          focus: "rgb(var(--color-brand) / <alpha-value>)",
        },
        /* Status */
        danger: {
          DEFAULT: "rgb(var(--color-error) / <alpha-value>)",
          subtle: "rgb(var(--color-error-subtle) / <alpha-value>)",
        },
        success: {
          DEFAULT: "rgb(var(--color-success) / <alpha-value>)",
          subtle: "rgb(var(--color-success-subtle) / <alpha-value>)",
        },
        warning: {
          DEFAULT: "rgb(var(--color-warning) / <alpha-value>)",
          subtle: "rgb(var(--color-warning-subtle) / <alpha-value>)",
        },
        info: {
          DEFAULT: "rgb(var(--color-info) / <alpha-value>)",
          subtle: "rgb(var(--color-info-subtle) / <alpha-value>)",
        },
      },

      /* ─── Typography ───────────────────────────────────────── */
      fontFamily: {
        sans: ["var(--font-latin)", "Plus Jakarta Sans", "system-ui", "sans-serif"],
        persian: ["var(--font-persian)", "Vazirmatn", "Tahoma", "sans-serif"],
        mono: ["var(--font-mono)", "JetBrains Mono", "monospace"],
      },
      fontSize: {
        "display-2xl": ["3rem", { lineHeight: "3.5rem", letterSpacing: "-0.02em", fontWeight: "700" }],
        "display-xl": ["2.25rem", { lineHeight: "2.75rem", letterSpacing: "-0.02em", fontWeight: "700" }],
        "display-lg": ["1.875rem", { lineHeight: "2.375rem", letterSpacing: "-0.01em", fontWeight: "600" }],
        "heading-lg": ["1.5rem", { lineHeight: "2rem", letterSpacing: "-0.01em", fontWeight: "600" }],
        "heading-md": ["1.25rem", { lineHeight: "1.75rem", letterSpacing: "-0.005em", fontWeight: "600" }],
        "heading-sm": ["1rem", { lineHeight: "1.5rem", letterSpacing: "0em", fontWeight: "600" }],
        "body-lg": ["1rem", { lineHeight: "1.625rem", fontWeight: "400" }],
        "body-md": ["0.875rem", { lineHeight: "1.375rem", fontWeight: "400" }],
        "body-sm": ["0.75rem", { lineHeight: "1.125rem", fontWeight: "400" }],
      },

      /* ─── 4px Base Grid Spacing ────────────────────────────── */
      spacing: {
        "0.5": "2px",
        "1": "4px",
        "2": "8px",
        "3": "12px",
        "4": "16px",
        "5": "20px",
        "6": "24px",
        "8": "32px",
        "10": "40px",
        "12": "48px",
        "16": "64px",
        "20": "80px",
        "24": "96px",
      },

      /* ─── Border Radius (Spec) ─────────────────────────────── */
      borderRadius: {
        sm: "8px",
        md: "12px",
        lg: "16px",
        xl: "20px",
        full: "9999px",
      },

      /* ─── Elevation Shadows (Spec) ─────────────────────────── */
      boxShadow: {
        sm: "0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
        md: "0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.05)",
        lg: "0 14px 40px -4px rgba(0,0,0,0.08), 0 4px 12px -2px rgba(0,0,0,0.04)",
        xl: "0 24px 56px -6px rgba(0,0,0,0.12), 0 8px 20px -4px rgba(0,0,0,0.06)",
        "login-card": "0 20px 60px rgba(0,0,0,0.12)",
        "focus-ring": "0 0 0 3px rgba(14,110,110,0.12)",
        "focus-ring-error": "0 0 0 3px rgba(220,38,38,0.10)",
        "focus-ring-success": "0 0 0 3px rgba(22,163,74,0.10)",
        "toast": "0 8px 24px rgba(0,0,0,0.12)",
        /* Legacy aliases */
        "elevation-1": "0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
        "elevation-2": "0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.05)",
        "elevation-3": "0 14px 40px -4px rgba(0,0,0,0.08), 0 4px 12px -2px rgba(0,0,0,0.04)",
        "elevation-4": "0 24px 56px -6px rgba(0,0,0,0.12), 0 8px 20px -4px rgba(0,0,0,0.06)",
        glass: "0 8px 32px rgba(0,0,0,0.06)",
      },

      /* ─── Z-Index Scale ────────────────────────────────────── */
      zIndex: {
        dropdown: "10",
        sticky: "20",
        overlay: "30",
        modal: "40",
        popover: "50",
        toast: "60",
        tooltip: "70",
      },

      /* ─── Motion Tokens (Spec) ─────────────────────────────── */
      transitionDuration: {
        fast: "100ms",
        base: "150ms",
        normal: "200ms",
        slow: "300ms",
        slower: "500ms",
      },
      transitionTimingFunction: {
        apple: "cubic-bezier(0.25, 0.1, 0.25, 1)",
        spring: "cubic-bezier(0.175, 0.885, 0.32, 1.275)",
        smooth: "ease-in-out",
      },

      /* ─── Layout ───────────────────────────────────────────── */
      maxWidth: {
        content: "72rem",
        narrow: "40rem",
        reading: "48rem",
      },

      /* ─── Keyframes ────────────────────────────────────────── */
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "fade-out": {
          from: { opacity: "1", transform: "translateY(0)" },
          to: { opacity: "0", transform: "translateY(8px)" },
        },
        "slide-down": {
          from: { opacity: "0", transform: "translateY(-8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "slide-up": {
          from: { opacity: "0", transform: "translateY(12px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in-start": {
          from: { opacity: "0", transform: "translateX(calc(var(--direction-multiplier, 1) * -16px))" },
          to: { opacity: "1", transform: "translateX(0)" },
        },
        "scale-in": {
          from: { opacity: "0", transform: "scale(0.95)" },
          to: { opacity: "1", transform: "scale(1)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "0.45" },
          "50%": { opacity: "0.95" },
        },
        "pulse-status": {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.6", transform: "scale(1.15)" },
        },
      },
      animation: {
        "fade-in": "fade-in 300ms ease-in-out forwards",
        "fade-out": "fade-out 200ms ease-in-out forwards",
        "slide-down": "slide-down 200ms ease-in-out forwards",
        "slide-up": "slide-up 250ms ease-in-out forwards",
        "slide-in-start": "slide-in-start 300ms ease-in-out forwards",
        "scale-in": "scale-in 200ms cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards",
        shimmer: "shimmer 1.5s ease-in-out infinite",
        "pulse-soft": "pulse-soft 1.6s ease-in-out infinite",
        "pulse-status": "pulse-status 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
