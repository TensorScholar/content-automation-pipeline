import type { Config } from "tailwindcss";

/**
 * Smarlux Composition Desk design tokens.
 * CSS variables in app/globals.css are canonical; Tailwind exposes semantic aliases only.
 */
const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: "rgb(var(--color-brand) / <alpha-value>)",
          hover: "rgb(var(--color-brand-hover) / <alpha-value>)",
          light: "rgb(var(--color-brand-light) / <alpha-value>)",
          accent: "rgb(var(--color-brand-accent) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "rgb(var(--color-brand) / <alpha-value>)",
          hover: "rgb(var(--color-brand-hover) / <alpha-value>)",
          subtle: "rgb(var(--color-brand-light) / <alpha-value>)",
        },
        surface: {
          DEFAULT: "rgb(var(--color-surface) / <alpha-value>)",
          alt: "rgb(var(--color-surface-alt) / <alpha-value>)",
          secondary: "rgb(var(--color-surface-alt) / <alpha-value>)",
          tertiary: "rgb(var(--color-surface-tertiary) / <alpha-value>)",
          elevated: "rgb(var(--color-surface) / <alpha-value>)",
          sunken: "rgb(var(--color-surface-tertiary) / <alpha-value>)",
        },
        ink: {
          DEFAULT: "rgb(var(--color-text-primary) / <alpha-value>)",
          secondary: "rgb(var(--color-text-secondary) / <alpha-value>)",
          muted: "rgb(var(--color-text-muted) / <alpha-value>)",
          tertiary: "rgb(var(--color-text-muted) / <alpha-value>)",
          inverse: "rgb(255 255 255 / <alpha-value>)",
        },
        line: "rgb(var(--color-border) / var(--color-border-alpha))",
        border: {
          DEFAULT: "rgb(var(--color-border) / var(--color-border-alpha))",
          strong: "rgb(var(--color-border) / var(--color-border-strong-alpha))",
          secondary: "rgb(var(--color-border) / var(--color-border-alpha))",
          focus: "rgb(var(--color-brand) / <alpha-value>)",
        },
        danger: {
          DEFAULT: "rgb(var(--color-error) / <alpha-value>)",
          subtle: "rgb(var(--color-error-subtle) / <alpha-value>)",
        },
        error: {
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
      fontFamily: {
        sans: ["var(--font-family-system)"],
        rtl: ["var(--font-family-rtl)"],
        mono: ["SFMono-Regular", "ui-monospace", "Menlo", "Consolas", "monospace"],
      },
      fontSize: {
        "display-hero": ["var(--text-hero-size)", { lineHeight: "var(--text-hero-line)", letterSpacing: "var(--letter-spacing-tight)", fontWeight: "500" }],
        "display-lg": ["var(--text-display-size)", { lineHeight: "var(--text-display-line)", letterSpacing: "var(--letter-spacing-tight)", fontWeight: "600" }],
        "display-2xl": ["var(--text-2xl-size)", { lineHeight: "var(--text-2xl-line)", letterSpacing: "var(--letter-spacing-tight)", fontWeight: "600" }],
        metric: ["var(--text-metric-size)", { lineHeight: "var(--text-metric-line)", fontWeight: "600" }],
        "metric-lead": ["var(--text-metric-lead-size)", { lineHeight: "var(--text-metric-lead-line)", fontWeight: "600" }],
        "metric-support": ["var(--text-metric-support-size)", { lineHeight: "var(--text-metric-support-line)", fontWeight: "600" }],
        "display-xl": ["var(--text-xl-size)", { lineHeight: "var(--text-xl-line)", fontWeight: "600" }],
        "heading-lg": ["var(--text-xl-size)", { lineHeight: "var(--text-xl-line)", fontWeight: "600" }],
        "heading-md": ["var(--text-lg-size)", { lineHeight: "var(--text-lg-line)", fontWeight: "600" }],
        "heading-sm": ["var(--text-base-size)", { lineHeight: "var(--text-base-line)", fontWeight: "600" }],
        "page-title": ["var(--text-2xl-size)", { lineHeight: "var(--text-2xl-line)", fontWeight: "600" }],
        "object-title": ["var(--text-xl-size)", { lineHeight: "var(--text-xl-line)", fontWeight: "600" }],
        "section-title": ["var(--text-section-size)", { lineHeight: "var(--text-section-line)", fontWeight: "600" }],
        "doc-body": ["var(--text-doc-size)", { lineHeight: "var(--text-doc-line)", fontWeight: "400" }],
        "body-lg": ["var(--text-doc-size)", { lineHeight: "var(--text-doc-line)", fontWeight: "400" }],
        "body-md": ["var(--text-base-size)", { lineHeight: "var(--text-base-line)", fontWeight: "400" }],
        "body-sm": ["var(--text-sm-size)", { lineHeight: "var(--text-sm-line)", fontWeight: "400" }],
        controls: ["var(--text-sm-size)", { lineHeight: "var(--text-sm-line)", fontWeight: "500" }],
        metadata: ["var(--text-xs-size)", { lineHeight: "var(--text-xs-line)", fontWeight: "400" }],
        xs: ["var(--text-xs-size)", { lineHeight: "var(--text-xs-line)" }],
        sm: ["var(--text-sm-size)", { lineHeight: "var(--text-sm-line)" }],
        base: ["var(--text-base-size)", { lineHeight: "var(--text-base-line)" }],
        lg: ["var(--text-lg-size)", { lineHeight: "var(--text-lg-line)" }],
        xl: ["var(--text-xl-size)", { lineHeight: "var(--text-xl-line)" }],
        "2xl": ["var(--text-2xl-size)", { lineHeight: "var(--text-2xl-line)" }],
      },
      borderRadius: {
        xs: "var(--radius-xs)",
        sm: "var(--radius-sm)",
        md: "var(--radius-md)",
        lg: "var(--radius-md)",
        xl: "var(--radius-md)",
        full: "var(--radius-full)",
        DEFAULT: "var(--radius-sm)",
      },
      boxShadow: {
        dropdown: "0 4px 16px -4px rgba(0, 0, 0, 0.18), 0 1px 3px rgba(0, 0, 0, 0.08)",
        modal: "0 16px 36px -12px rgba(0, 0, 0, 0.28), 0 2px 8px rgba(0, 0, 0, 0.08)",
        popover: "0 8px 24px -6px rgba(0, 0, 0, 0.22), 0 2px 6px rgba(0, 0, 0, 0.06)",
        toast: "0 12px 28px -8px rgba(0, 0, 0, 0.24), 0 2px 6px rgba(0, 0, 0, 0.06)",
        sm: "0 1px 2px rgba(0, 0, 0, 0.04)",
        md: "0 8px 24px -8px rgba(0, 0, 0, 0.20)",
        lg: "0 16px 36px -12px rgba(0, 0, 0, 0.24)",
        xl: "0 24px 48px -16px rgba(0, 0, 0, 0.28)",
      },
      zIndex: {
        dropdown: "10",
        sticky: "20",
        overlay: "30",
        modal: "40",
        popover: "50",
        toast: "60",
        tooltip: "70",
      },
      transitionDuration: {
        fast: "120ms",
        base: "160ms",
        normal: "160ms",
        slow: "220ms",
      },
      transitionTimingFunction: {
        precise: "cubic-bezier(0.2,0,0,1)",
      },
      maxWidth: {
        content: "72rem",
        narrow: "40rem",
        reading: "48rem",
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "translateY(3px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "fade-out": { from: { opacity: "1" }, to: { opacity: "0" } },
        "slide-down": {
          from: { opacity: "0", transform: "translateY(-4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "slide-up": {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in-start": {
          from: { opacity: "0", transform: "translateX(calc(var(--direction-multiplier) * -6px))" },
          to: { opacity: "1", transform: "translateX(0)" },
        },
        "scale-in": {
          from: { opacity: "0", transform: "scale(.985)" },
          to: { opacity: "1", transform: "scale(1)" },
        },
        "pulse-soft": { "0%,100%": { opacity: ".45" }, "50%": { opacity: ".85" } },
        "pulse-status": { "0%,100%": { opacity: "1" }, "50%": { opacity: ".48" } },
      },
      animation: {
        "fade-in": "fade-in 160ms cubic-bezier(0.2,0,0,1) forwards",
        "fade-out": "fade-out 120ms ease-out forwards",
        "slide-down": "slide-down 160ms cubic-bezier(0.2,0,0,1) forwards",
        "slide-up": "slide-up 160ms cubic-bezier(0.2,0,0,1) forwards",
        "slide-in-start": "slide-in-start 160ms cubic-bezier(0.2,0,0,1) forwards",
        "scale-in": "scale-in 140ms cubic-bezier(0.2,0,0,1) forwards",
        "pulse-soft": "pulse-soft 1.6s ease-in-out infinite",
        "pulse-status": "pulse-status 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
