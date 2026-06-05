"use client";

import { useState } from "react";

/* ═══════════════════════════════════════════════════════════════════
   DESIGN SYSTEM SHOWCASE — Visual Verification Page
   Temporary page for CTO review of all design tokens.
   Route: /design-system
   ═══════════════════════════════════════════════════════════════════ */

/* ─── Data ────────────────────────────────────────────────────────── */

const SURFACE_COLORS = [
  { name: "surface",           css: "bg-surface",           hex: "#FFFFFF" },
  { name: "surface-secondary", css: "bg-surface-secondary", hex: "#F9FAFB" },
  { name: "surface-tertiary",  css: "bg-surface-tertiary",  hex: "#F3F4F6" },
  { name: "surface-elevated",  css: "bg-surface-elevated",  hex: "#FFFFFF" },
  { name: "surface-sunken",    css: "bg-surface-sunken",    hex: "#F0F1F3" },
];

const INK_COLORS = [
  { name: "ink",           css: "bg-ink",           hex: "#111827", light: true },
  { name: "ink-secondary", css: "bg-ink-secondary", hex: "#6B7280", light: true },
  { name: "ink-tertiary",  css: "bg-ink-tertiary",  hex: "#9CA3AF", light: true },
  { name: "ink-inverse",   css: "bg-ink-inverse",   hex: "#FFFFFF" },
];

const ACCENT_COLORS = [
  { name: "accent",       css: "bg-accent",       hex: "#0F9488", light: true },
  { name: "accent-hover", css: "bg-accent-hover", hex: "#0D8476", light: true },
  { name: "accent-subtle",css: "bg-accent-subtle", hex: "#E0F7F4" },
];

const STATUS_COLORS = [
  { name: "danger",         css: "bg-danger",         hex: "#EF4444", light: true },
  { name: "danger-subtle",  css: "bg-danger-subtle",  hex: "#FEF2F2" },
  { name: "success",        css: "bg-success",        hex: "#10B981", light: true },
  { name: "success-subtle", css: "bg-success-subtle", hex: "#ECFDF5" },
  { name: "warning",        css: "bg-warning",        hex: "#F59E0B", light: true },
  { name: "warning-subtle", css: "bg-warning-subtle", hex: "#FFF7ED" },
  { name: "info",           css: "bg-info",           hex: "#3B82F6", light: true },
  { name: "info-subtle",    css: "bg-info-subtle",    hex: "#EFF6FF" },
];

const BORDER_COLORS = [
  { name: "border",           css: "bg-border",           hex: "rgba(0,0,0,0.08)" },
  { name: "border-secondary", css: "bg-border-secondary", hex: "#E5E7EB" },
];

const TYPE_SCALE = [
  { token: "text-2xl", css: "text-display-lg", size: "24 / 32", weight: "700", en: "Display Title",      fa: "عنوان اصلی" },
  { token: "text-xl",  css: "text-heading-lg", size: "18 / 28", weight: "600", en: "Section Heading",    fa: "عنوان بخش" },
  { token: "text-lg",  css: "text-heading-md", size: "16 / 24", weight: "500", en: "Emphasized UI",      fa: "رابط تاکیدشده" },
  { token: "text-base",css: "text-body-md",    size: "14 / 22", weight: "400", en: "Body",               fa: "متن بدنه" },
  { token: "text-sm",  css: "text-body-md",    size: "13 / 20", weight: "400", en: "Compact Body",       fa: "متن فشرده" },
  { token: "text-xs",  css: "text-body-sm",    size: "12 / 16", weight: "400", en: "Label / Caption",    fa: "برچسب / کپشن" },
];

const ELEVATIONS = [
  { name: "elevation-1", css: "shadow-elevation-1", desc: "0 1px 2px rgba(0,0,0,0.05)" },
  { name: "elevation-2", css: "shadow-elevation-2", desc: "0 4px 6px rgba(0,0,0,0.10)" },
  { name: "elevation-3", css: "shadow-elevation-3", desc: "0 10px 25px rgba(0,0,0,0.10)" },
];

const SPACING_GRID = [
  { token: "spacing-1",  px: 4 },
  { token: "spacing-2",  px: 8 },
  { token: "spacing-3",  px: 12 },
  { token: "spacing-4",  px: 16 },
  { token: "spacing-5",  px: 20 },
  { token: "spacing-6",  px: 24 },
  { token: "spacing-8",  px: 32 },
  { token: "spacing-10", px: 40 },
  { token: "spacing-12", px: 48 },
];

/* ─── Components ──────────────────────────────────────────────────── */

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-display-lg text-ink mb-2">{children}</h2>
  );
}

function SectionDesc({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-body-md text-ink-secondary mb-6">{children}</p>
  );
}

function ColorSwatch({ name, css, hex, light }: { name: string; css: string; hex: string; light?: boolean }) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className={`${css} w-16 h-16 rounded-xl border border-border shadow-elevation-1 transition-transform duration-normal ease-apple hover:scale-110`}
      />
      <span className={`text-body-sm font-semibold ${light ? "text-ink" : "text-ink-secondary"}`}>{name}</span>
      <span className="text-body-sm text-ink-tertiary font-mono">{hex}</span>
    </div>
  );
}

/* ─── Page ────────────────────────────────────────────────────────── */

export default function DesignSystemPage() {
  const [dir, setDir] = useState<"ltr" | "rtl">("rtl");

  return (
    <div dir={dir} className="min-h-screen p-grid-4 md:p-grid-6 max-w-content mx-auto">

      {/* ── Header ── */}
      <header className="mb-grid-6">
        <p className="text-body-sm text-accent font-semibold tracking-wider uppercase mb-2">
          Smarlux Design System
        </p>
        <h1 className="text-display-2xl text-ink mb-grid-1">
          Visual Token Showcase
        </h1>
        <p className="text-body-lg text-ink-secondary max-w-reading">
          CTO verification page — every primitive rendered for browser inspection.
          Toggle direction below to verify RTL/LTR behavior.
        </p>

        {/* Direction Toggle */}
        <div className="mt-grid-3 flex items-center gap-3">
          <button
            onClick={() => setDir("ltr")}
            className={`px-grid-2 py-2 rounded-lg text-body-md font-semibold transition-all duration-normal ease-apple ${
              dir === "ltr"
                ? "bg-accent text-ink-inverse shadow-elevation-2"
                : "bg-surface-tertiary text-ink-secondary hover:bg-surface-sunken"
            }`}
          >
            LTR — English
          </button>
          <button
            onClick={() => setDir("rtl")}
            className={`px-grid-2 py-2 rounded-lg text-body-md font-semibold transition-all duration-normal ease-apple ${
              dir === "rtl"
                ? "bg-accent text-ink-inverse shadow-elevation-2"
                : "bg-surface-tertiary text-ink-secondary hover:bg-surface-sunken"
            }`}
          >
            RTL — فارسی
          </button>
          <span className="text-body-sm text-ink-tertiary ms-2">
            Current: <code className="font-mono text-accent">{dir.toUpperCase()}</code>
          </span>
        </div>
      </header>

      {/* ═══════════════ SECTION 1: COLORS ═══════════════ */}
      <section className="mb-grid-8">
        <SectionTitle>Semantic Color System</SectionTitle>
        <SectionDesc>28 tokens across 6 semantic groups. All support Tailwind opacity modifiers.</SectionDesc>

        {/* Surface */}
        <div className="elevated-card p-grid-3 mb-grid-3">
          <h3 className="text-heading-sm text-ink mb-grid-2">Surface Palette</h3>
          <div className="flex flex-wrap gap-grid-3">
            {SURFACE_COLORS.map((c) => <ColorSwatch key={c.name} {...c} />)}
          </div>
        </div>

        {/* Ink */}
        <div className="elevated-card p-grid-3 mb-grid-3">
          <h3 className="text-heading-sm text-ink mb-grid-2">Ink (Text) Palette</h3>
          <div className="flex flex-wrap gap-grid-3">
            {INK_COLORS.map((c) => <ColorSwatch key={c.name} {...c} />)}
          </div>
        </div>

        {/* Accent */}
        <div className="elevated-card p-grid-3 mb-grid-3">
          <h3 className="text-heading-sm text-ink mb-grid-2">Brand Accent</h3>
          <div className="flex flex-wrap gap-grid-3">
            {ACCENT_COLORS.map((c) => <ColorSwatch key={c.name} {...c} />)}
          </div>
        </div>

        {/* Status */}
        <div className="elevated-card p-grid-3 mb-grid-3">
          <h3 className="text-heading-sm text-ink mb-grid-2">Semantic Status</h3>
          <div className="flex flex-wrap gap-grid-3">
            {STATUS_COLORS.map((c) => <ColorSwatch key={c.name} {...c} />)}
          </div>
        </div>

        {/* Border */}
        <div className="elevated-card p-grid-3">
          <h3 className="text-heading-sm text-ink mb-grid-2">Border</h3>
          <div className="flex flex-wrap gap-grid-3">
            {BORDER_COLORS.map((c) => <ColorSwatch key={c.name} {...c} />)}
          </div>
        </div>
      </section>

      {/* ═══════════════ SECTION 2: TYPOGRAPHY ═══════════════ */}
      <section className="mb-grid-8">
        <SectionTitle>9-Level Typography Scale</SectionTitle>
        <SectionDesc>
          Each level shown in Inter (English) and Vazirmatn (Persian). Toggle LTR/RTL above to see font switching.
        </SectionDesc>

        <div className="elevated-card overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border bg-surface-tertiary/50">
                <th className="text-start text-body-sm text-ink-secondary font-semibold px-grid-2 py-3">Token</th>
                <th className="text-start text-body-sm text-ink-secondary font-semibold px-grid-2 py-3">Size / Line</th>
                <th className="text-start text-body-sm text-ink-secondary font-semibold px-grid-2 py-3">English (Inter)</th>
                <th className="text-start text-body-sm text-ink-secondary font-semibold px-grid-2 py-3">فارسی (Vazirmatn)</th>
              </tr>
            </thead>
            <tbody>
              {TYPE_SCALE.map((row) => (
                <tr key={row.token} className="border-b border-border-secondary last:border-b-0 hover:bg-surface-secondary/50 transition-colors duration-fast">
                  <td className="px-grid-2 py-grid-2 align-baseline">
                    <code className="text-body-sm font-mono text-accent bg-accent-subtle px-2 py-0.5 rounded">
                      {row.token}
                    </code>
                    <span className="block text-body-sm text-ink-tertiary mt-1">{row.size}px · w{row.weight}</span>
                  </td>
                  <td className="px-grid-2 py-grid-2 align-baseline text-body-sm text-ink-secondary font-mono">
                    {row.size}
                  </td>
                  <td className="px-grid-2 py-grid-2 align-baseline" dir="ltr">
                    <span className={`${row.css} text-ink`} style={{ fontFamily: "var(--font-ui), Inter, system-ui, sans-serif" }}>
                      {row.en}
                    </span>
                  </td>
                  <td className="px-grid-2 py-grid-2 align-baseline" dir="rtl">
                    <span className={`${row.css} text-ink`} style={{ fontFamily: "var(--font-persian), Vazirmatn, Tahoma, sans-serif" }}>
                      {row.fa}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ═══════════════ SECTION 3: ELEVATION & PRIMITIVES ═══════════════ */}
      <section className="mb-grid-8">
        <SectionTitle>Elevation System &amp; Component Primitives</SectionTitle>
        <SectionDesc>4-level shadow scale + glass-card, elevated-card, and skeleton primitives.</SectionDesc>

        {/* Elevation Levels */}
        <h3 className="text-heading-sm text-ink mb-grid-2">Shadow Elevations</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-grid-3 mb-grid-4">
          {ELEVATIONS.map((e) => (
            <div
              key={e.name}
              className={`${e.css} bg-surface rounded-xl p-grid-3 border border-border-secondary transition-shadow duration-slow ease-apple hover:shadow-elevation-4`}
            >
              <p className="text-heading-sm text-ink mb-1">{e.name}</p>
              <p className="text-body-sm text-ink-secondary">{e.desc}</p>
            </div>
          ))}
        </div>

        {/* Card Primitives */}
        <h3 className="text-heading-sm text-ink mb-grid-2">Card Primitives</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-grid-3 mb-grid-4">
          {/* Glass Card */}
          <div className="glass-card p-grid-3">
            <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center mb-3">
              <span className="text-accent text-heading-md">◆</span>
            </div>
            <h4 className="text-heading-sm text-ink mb-1">.glass-card</h4>
            <p className="text-body-md text-ink-secondary">
              Frosted glass with backdrop-filter blur. Premium overlay feel.
            </p>
          </div>

          {/* Elevated Card */}
          <div className="elevated-card p-grid-3">
            <div className="w-10 h-10 rounded-lg bg-success/10 flex items-center justify-center mb-3">
              <span className="text-success text-heading-md">■</span>
            </div>
            <h4 className="text-heading-sm text-ink mb-1">.elevated-card</h4>
            <p className="text-body-md text-ink-secondary">
              Solid surface with subtle elevation shadow. Default card style.
            </p>
          </div>

          {/* Skeleton */}
          <div className="elevated-card p-grid-3">
            <div className="flex flex-col gap-3">
              <div className="skeleton h-4 w-3/4" />
              <div className="skeleton h-4 w-full" />
              <div className="skeleton h-4 w-5/6" />
              <div className="skeleton h-10 w-1/3 mt-2" />
            </div>
            <h4 className="text-heading-sm text-ink mt-4 mb-1">.skeleton</h4>
            <p className="text-body-md text-ink-secondary">
              Shimmer loading placeholder. Infinite animation.
            </p>
          </div>
        </div>
      </section>

      {/* ═══════════════ SECTION 4: 8px GRID ═══════════════ */}
      <section className="mb-grid-8">
        <SectionTitle>8px Spacing Grid</SectionTitle>
        <SectionDesc>All spacing tokens are multiples of the 8px base unit.</SectionDesc>

        <div className="elevated-card p-grid-3">
          <div className="flex flex-wrap items-end gap-grid-2">
            {SPACING_GRID.map((s) => (
              <div key={s.token} className="flex flex-col items-center gap-2">
                <div
                  className="bg-accent/20 border border-accent/30 rounded"
                  style={{ width: s.px, height: s.px }}
                />
                <code className="text-body-sm font-mono text-accent">{s.token}</code>
                <span className="text-body-sm text-ink-tertiary">{s.px}px</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════ SECTION 5: MOTION & ANIMATION ═══════════════ */}
      <section className="mb-grid-8">
        <SectionTitle>Motion Tokens &amp; Animations</SectionTitle>
        <SectionDesc>Apple-like easing curves and direction-aware animations. Hover cards to see transitions.</SectionDesc>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-grid-3 mb-grid-4">
          {/* Easing: Apple */}
          <div className="elevated-card p-grid-3 group cursor-pointer">
            <div className="w-12 h-12 rounded-xl bg-accent flex items-center justify-center mb-3 transition-transform duration-slow ease-apple group-hover:translate-x-4 group-hover:scale-110">
              <span className="text-ink-inverse text-heading-sm">→</span>
            </div>
            <h4 className="text-heading-sm text-ink mb-1">ease-apple</h4>
            <p className="text-body-sm text-ink-secondary">cubic-bezier(0.25, 0.1, 0.25, 1)</p>
            <p className="text-body-sm text-ink-tertiary mt-1">Hover to see motion</p>
          </div>

          {/* Easing: Spring */}
          <div className="elevated-card p-grid-3 group cursor-pointer">
            <div className="w-12 h-12 rounded-xl bg-success flex items-center justify-center mb-3 transition-transform duration-slow ease-spring group-hover:scale-125">
              <span className="text-ink-inverse text-heading-sm">⟳</span>
            </div>
            <h4 className="text-heading-sm text-ink mb-1">ease-spring</h4>
            <p className="text-body-sm text-ink-secondary">cubic-bezier(0.175, 0.885, 0.32, 1.275)</p>
            <p className="text-body-sm text-ink-tertiary mt-1">Hover to see bounce</p>
          </div>

          {/* Easing: Smooth */}
          <div className="elevated-card p-grid-3 group cursor-pointer">
            <div className="w-12 h-12 rounded-xl bg-warning flex items-center justify-center mb-3 transition-all duration-slower ease-smooth group-hover:rounded-full group-hover:rotate-180">
              <span className="text-ink-inverse text-heading-sm">◇</span>
            </div>
            <h4 className="text-heading-sm text-ink mb-1">ease-smooth</h4>
            <p className="text-body-sm text-ink-secondary">cubic-bezier(0.4, 0, 0.2, 1)</p>
            <p className="text-body-sm text-ink-tertiary mt-1">Hover to see morph</p>
          </div>
        </div>

        {/* Keyframe Animations */}
        <h3 className="text-heading-sm text-ink mb-grid-2">Keyframe Animations</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-grid-3">
          <AnimationCard name="fade-in" css="animate-fade-in" />
          <AnimationCard name="slide-in-start" css="animate-slide-in-start" />
          <AnimationCard name="scale-in" css="animate-scale-in" />
          <div className="elevated-card p-grid-3 flex flex-col items-center justify-center">
            <div className="w-full h-4 skeleton mb-2" />
            <div className="w-3/4 h-4 skeleton mb-2" />
            <div className="w-1/2 h-4 skeleton" />
            <p className="text-body-sm text-ink-secondary mt-3">shimmer</p>
          </div>
        </div>
      </section>

      {/* ═══════════════ SECTION 6: RTL/LTR PROOF ═══════════════ */}
      <section className="mb-grid-8">
        <SectionTitle>RTL / LTR Logical Properties Proof</SectionTitle>
        <SectionDesc>
          These elements use logical properties (ms-*, me-*, ps-*, pe-*, start-*, end-*).
          Toggle direction above — layout flips automatically without any JS class swapping.
        </SectionDesc>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-grid-3">
          {/* Logical Margin */}
          <div className="elevated-card p-grid-3">
            <h4 className="text-heading-sm text-ink mb-grid-2">Logical Margin (ms-*)</h4>
            <div className="space-y-2">
              {[2, 4, 6, 8, 12].map((v) => (
                <div key={v} className="flex items-center">
                  <div
                    className={`ms-${v} h-8 bg-accent/80 rounded flex items-center justify-center px-3`}
                    style={{ marginInlineStart: `${v * 4}px` }}
                  >
                    <span className="text-body-sm text-ink-inverse font-mono">ms-{v}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Logical Padding */}
          <div className="elevated-card p-grid-3">
            <h4 className="text-heading-sm text-ink mb-grid-2">Logical Padding (ps-*)</h4>
            <div className="space-y-2">
              {[2, 4, 6, 8, 12].map((v) => (
                <div
                  key={v}
                  className="bg-surface-tertiary rounded border border-border-secondary"
                  style={{ paddingInlineStart: `${v * 4}px` }}
                >
                  <div className="h-8 bg-info/20 rounded flex items-center px-3">
                    <span className="text-body-sm text-info font-mono">ps-{v}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Bidi Text Alignment */}
          <div className="elevated-card p-grid-3">
            <h4 className="text-heading-sm text-ink mb-grid-2">Text Alignment (text-start / text-end)</h4>
            <div className="space-y-3">
              <div className="bg-surface-tertiary rounded p-3 text-start">
                <span className="text-body-md text-ink">text-start →</span>
                <span className="text-body-sm text-ink-tertiary block">Aligns to inline-start edge</span>
              </div>
              <div className="bg-surface-tertiary rounded p-3 text-end">
                <span className="text-body-md text-ink">← text-end</span>
                <span className="text-body-sm text-ink-tertiary block">Aligns to inline-end edge</span>
              </div>
            </div>
          </div>

          {/* Bidi Border */}
          <div className="elevated-card p-grid-3">
            <h4 className="text-heading-sm text-ink mb-grid-2">Logical Borders (border-s-*, border-e-*)</h4>
            <div className="space-y-3">
              <div className="border-s-4 border-s-accent bg-surface-tertiary rounded p-3">
                <span className="text-body-md text-ink">border-s-4 border-s-accent</span>
                <span className="text-body-sm text-ink-tertiary block">Start border flips with direction</span>
              </div>
              <div className="border-e-4 border-e-danger bg-surface-tertiary rounded p-3">
                <span className="text-body-md text-ink">border-e-4 border-e-danger</span>
                <span className="text-body-sm text-ink-tertiary block">End border flips with direction</span>
              </div>
              <div className="rounded-s-xl bg-accent/10 border border-accent/20 p-3">
                <span className="text-body-md text-accent">rounded-s-xl</span>
                <span className="text-body-sm text-ink-tertiary block">Start radius flips with direction</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Z-Index Scale ── */}
      <section className="mb-grid-8">
        <SectionTitle>Z-Index Semantic Scale</SectionTitle>
        <SectionDesc>Named z-index layers prevent magic numbers across the codebase.</SectionDesc>
        <div className="elevated-card p-grid-3">
          <div className="flex flex-wrap gap-grid-2">
            {["dropdown:10", "sticky:20", "overlay:30", "modal:40", "popover:50", "toast:60", "tooltip:70"].map((z) => {
              const [name, val] = z.split(":");
              return (
                <div key={name} className="flex items-center gap-2 bg-surface-tertiary rounded-lg px-3 py-2">
                  <code className="text-body-sm font-mono text-accent">z-{name}</code>
                  <span className="text-body-sm text-ink-tertiary">{val}</span>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-border pt-grid-3 pb-grid-6">
        <p className="text-body-sm text-ink-tertiary text-center">
          Smarlux Design System · Token Showcase · Temporary verification page
        </p>
      </footer>
    </div>
  );
}

/* ─── Animation Demo Card (re-triggers on click) ─────────────────── */

function AnimationCard({ name, css }: { name: string; css: string }) {
  const [key, setKey] = useState(0);

  return (
    <div
      className="elevated-card p-grid-3 flex flex-col items-center justify-center cursor-pointer"
      onClick={() => setKey((k) => k + 1)}
    >
      <div
        key={key}
        className={`w-16 h-16 rounded-xl bg-accent/20 border border-accent/30 ${css} mb-3`}
      />
      <p className="text-body-sm text-ink-secondary">{name}</p>
      <p className="text-body-sm text-ink-tertiary mt-0.5">Click to replay</p>
    </div>
  );
}
