import type { Config } from "tailwindcss";

/* ═══════════════════════════════════════════════════════════════════
   SMARLUX CONTENT OS — Design Token System (Tailwind)
   Spec: Smarlux_UI_Master_Redesign_Prompt.md
   Deep Teal brand · macOS system typography · 4px grid
   ═══════════════════════════════════════════════════════════════════ */

const config: Config = {
  darkMode: "class",
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
        /* macOS glassmorphism pivot tokens */
        macos: {
          app: "rgb(var(--macos-app-bg) / <alpha-value>)",
          glass: "rgb(var(--macos-glass-bg) / <alpha-value>)",
          "glass-border": "rgb(var(--macos-glass-border) / <alpha-value>)",
          "segment-bg": "rgb(var(--macos-segment-bg) / <alpha-value>)",
          "shadow-ring": "rgb(var(--macos-shadow-ring) / <alpha-value>)",
          light: {
            app: "#f5f5f7",
            glass: "rgb(255 255 255 / 0.7)",
            border: "rgb(0 0 0 / 0.05)",
            segment: "rgb(0 0 0 / 0.05)",
          },
          dark: {
            app: "#151515",
            glass: "rgb(30 30 30 / 0.8)",
            border: "rgb(255 255 255 / 0.1)",
            segment: "rgb(255 255 255 / 0.1)",
          },
        },
      },

      /* ─── Typography ───────────────────────────────────────── */
      fontFamily: {
        sans: ["var(--font-family-system)"],
        persian: ["var(--font-family-farsi)"],
        mono: ["\"SF Mono\"", "ui-monospace", "Menlo", "Consolas", "monospace"],
      },
      fontSize: {
        "display-2xl": ["var(--text-2xl-size)", { lineHeight: "var(--text-2xl-line)", letterSpacing: "var(--letter-spacing-tight)", fontWeight: "700" }],
        "display-xl": ["var(--text-xl-size)", { lineHeight: "var(--text-xl-line)", letterSpacing: "var(--letter-spacing-tight)", fontWeight: "600" }],
        "display-lg": ["var(--text-xl-size)", { lineHeight: "var(--text-xl-line)", letterSpacing: "var(--letter-spacing-tight)", fontWeight: "600" }],
        "heading-lg": ["var(--text-xl-size)", { lineHeight: "var(--text-xl-line)", letterSpacing: "var(--letter-spacing-tight)", fontWeight: "600" }],
        "heading-md": ["var(--text-lg-size)", { lineHeight: "var(--text-lg-line)", letterSpacing: "var(--letter-spacing-normal)", fontWeight: "500" }],
        "heading-sm": ["var(--text-base-size)", { lineHeight: "var(--text-base-line)", letterSpacing: "var(--letter-spacing-normal)", fontWeight: "500" }],
        "body-lg": ["var(--text-lg-size)", { lineHeight: "var(--text-lg-line)", letterSpacing: "var(--letter-spacing-normal)", fontWeight: "500" }],
        "body-md": ["var(--text-base-size)", { lineHeight: "var(--text-base-line)", letterSpacing: "var(--letter-spacing-normal)", fontWeight: "400" }],
        "body-sm": ["var(--text-xs-size)", { lineHeight: "var(--text-xs-line)", letterSpacing: "var(--letter-spacing-normal)", fontWeight: "400" }],
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
        sm: "4px",
        md: "8px",
        lg: "10px",
        xl: "12px",
        full: "9999px",
      },

      /* ─── Elevation Shadows (Spec) ─────────────────────────── */
      boxShadow: {
        sm: "0 1px 2px rgba(0,0,0,0.05)",
        md: "0 4px 6px rgba(0,0,0,0.1)",
        lg: "0 10px 25px rgba(0,0,0,0.1)",
        xl: "0 10px 25px rgba(0,0,0,0.1)",
        "login-card": "0 20px 60px rgba(0,0,0,0.12)",
        "focus-ring": "0 0 0 2px rgb(255 255 255), 0 0 0 4px rgba(15,148,136,0.95)",
        "focus-ring-error": "0 0 0 3px rgba(220,38,38,0.10)",
        "focus-ring-success": "0 0 0 3px rgba(22,163,74,0.10)",
        "toast": "0 8px 24px rgba(0,0,0,0.12)",
        /* Legacy aliases */
        "elevation-1": "0 1px 2px rgba(0,0,0,0.05)",
        "elevation-2": "0 4px 6px rgba(0,0,0,0.1)",
        "elevation-3": "0 10px 25px rgba(0,0,0,0.1)",
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
        fast: "150ms",
        base: "250ms",
        normal: "250ms",
        slow: "350ms",
        slower: "500ms",
      },
      transitionTimingFunction: {
        apple: "cubic-bezier(0.16, 1, 0.3, 1)",
        spring: "cubic-bezier(0.34, 1.56, 0.64, 1)",
        smooth: "cubic-bezier(0.16, 1, 0.3, 1)",
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
