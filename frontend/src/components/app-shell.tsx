"use client";

import Image from "next/image";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTheme } from "next-themes";
import clsx from "clsx";
import { apiRequest } from "@/lib/api";
import { User, Project } from "@/types/models";
import { useAuth } from "@/providers/auth-provider";
import { useI18n } from "@/i18n/provider";
import { LanguageToggle } from "./language-toggle";
import { ErrorBoundary } from "./ui/error-boundary";
import { DashboardPanel } from "./panels/dashboard-panel";
import { ProjectsPanel } from "./panels/projects-panel";
import { ContentStudioPanel } from "./panels/content-studio-panel";
import { TasksPanel } from "./panels/tasks-panel";
import { UsersPanel } from "./panels/users-panel";
import { MonitoringPanel } from "./panels/monitoring-panel";
import {
  IconDashboard, IconProjects, IconStudio, IconTasks, IconUsers,
  IconMonitoring, IconChevron, IconMenu, IconLogout,
} from "./shell/nav-icons";

type AppPage = "dashboard" | "projects" | "studio" | "tasks" | "users" | "monitoring";
interface AppShellProps { token: string; user: User; }

const PAGE_ICONS: Record<AppPage, (props: { className?: string }) => React.ReactElement> = {
  dashboard: IconDashboard,
  projects: IconProjects,
  studio: IconStudio,
  tasks: IconTasks,
  users: IconUsers,
  monitoring: IconMonitoring,
};

// Pages whose label is shown in the utility strip.
// Dashboard, Projects, Tasks, Users, Monitoring suppress their page title there.
const STRIP_LABEL_PAGES = new Set<AppPage>(["studio"]);

function ThemeControls() {
  const { theme, setTheme } = useTheme();
  const { t } = useI18n();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const current = mounted ? (theme ?? "system") : "system";
  const modes = [
    { id: "system", label: t("theme.system") },
    { id: "light", label: t("theme.light") },
    { id: "dark", label: t("theme.dark") },
  ] as const;

  return (
    // Container: 4px (rounded-sm). Segment buttons: 4px (rounded-sm).
    <div className="grid grid-cols-3 gap-1 rounded-sm bg-ink/[0.045] p-1" dir="ltr">
      {modes.map((mode) => (
        <button
          key={mode.id}
          type="button"
          aria-pressed={current === mode.id}
          onClick={() => setTheme(mode.id)}
          className={clsx(
            "min-h-7 rounded-sm px-2 text-xs font-medium transition-colors",
            current === mode.id ? "bg-surface text-ink ring-1 ring-line" : "text-ink-tertiary hover:text-ink",
          )}
        >
          {mode.label}
        </button>
      ))}
    </div>
  );
}

export function AppShell({ token, user }: AppShellProps) {
  const { logout, isAdmin } = useAuth();
  const { t, direction } = useI18n();
  const [page, setPage] = useState<AppPage>("dashboard");
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [preferencesOpen, setPreferencesOpen] = useState(false);
  const preferencesTriggerRef = useRef<HTMLButtonElement>(null);
  const preferencesRef = useRef<HTMLDivElement>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [projectsLoading, setProjectsLoading] = useState(true);
  const [health, setHealth] = useState<{ version?: string } | null>(null);

  const navItems = useMemo(() => {
    const items: Array<{ key: AppPage; label: string; adminOnly?: boolean }> = [
      { key: "dashboard", label: t("nav.dashboard") },
      { key: "projects", label: t("nav.projects") },
      { key: "studio", label: t("nav.studio") },
      { key: "tasks", label: t("nav.tasks") },
      { key: "users", label: t("nav.users"), adminOnly: true },
      { key: "monitoring", label: t("nav.monitoring"), adminOnly: true },
    ];
    return items.filter((item) => !item.adminOnly || isAdmin);
  }, [isAdmin, t]);

  const activeNavItem = navItems.find((item) => item.key === page);
  const selectedProject = projects.find((project) => project.id === selectedProjectId);
  const pageUsesProjectContext = page === "projects" || page === "studio";
  const showStripLabel = STRIP_LABEL_PAGES.has(page);

  useEffect(() => {
    if (!isAdmin && (page === "users" || page === "monitoring")) setPage("dashboard");
  }, [isAdmin, page]);

  const refreshProjects = useCallback(async (signal?: AbortSignal) => {
    setProjectsLoading(true);
    try {
      const payload = await apiRequest<Project[]>("/projects", { token, signal });
      if (signal?.aborted) return;
      setProjects(payload);
      if (!payload.length) setSelectedProjectId(null);
      else setSelectedProjectId((current) => payload.some((project) => project.id === current) ? current : payload[0].id);
    } catch {
      if (signal?.aborted) return;
      setProjects([]);
      setSelectedProjectId(null);
    } finally {
      if (!signal?.aborted) setProjectsLoading(false);
    }
  }, [token]);

  useEffect(() => {
    const controller = new AbortController();
    void refreshProjects(controller.signal);
    return () => controller.abort();
  }, [refreshProjects]);

  useEffect(() => {
    const controller = new AbortController();
    apiRequest<{ version?: string }>("/system/health", { token, signal: controller.signal })
      .then((payload) => { if (!controller.signal.aborted) setHealth(payload); })
      .catch(() => undefined);
    return () => controller.abort();
  }, [token]);

  useEffect(() => {
    if (!mobileOpen) return;
    const close = (event: KeyboardEvent) => { if (event.key === "Escape") setMobileOpen(false); };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [mobileOpen]);

  useEffect(() => {
    if (!preferencesOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPreferencesOpen(false);
        preferencesTriggerRef.current?.focus();
      }
    };
    const handleClickOutside = (event: MouseEvent) => {
      if (
        !preferencesRef.current?.contains(event.target as Node) &&
        !preferencesTriggerRef.current?.contains(event.target as Node)
      ) {
        setPreferencesOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [preferencesOpen]);

  const navigate = (next: AppPage) => { setPage(next); setMobileOpen(false); };
  // RTL: drawer slides in from the right (logical start = right), so offset is positive
  const mobileDrawerOffset = mobileOpen ? "0px" : direction === "rtl" ? "calc(100% + 1rem)" : "calc(-100% - 1rem)";

  return (
    <div className="quiet-ui-scope flex h-dvh min-h-0 w-full overflow-hidden bg-[rgb(var(--bg-primary))] pt-[34px] text-ink" dir={direction}>
      {/* Tauri drag region — 34px, unchanged */}
      <div data-tauri-drag-region aria-hidden="true" className="fixed inset-x-0 top-0 z-[80] h-[34px] border-b border-line bg-[rgb(var(--bg-primary))]" />

      {/* Mobile backdrop */}
      {mobileOpen ? <button type="button" aria-label={t("common.close")} className="fixed inset-0 z-overlay bg-black/35 lg:hidden" onClick={() => setMobileOpen(false)} /> : null}

      {/* Sidebar — 200px expanded, 48px collapsed */}
      <aside
        id="primary-navigation"
        className={clsx(
          "fixed inset-y-[34px] start-0 z-modal flex flex-col border-e border-line bg-[rgb(var(--bg-secondary))] transition-[width,transform] duration-base translate-x-[var(--mobile-drawer-x)] lg:relative lg:inset-y-auto lg:z-auto lg:shrink-0 lg:translate-x-0",
          mobileOpen ? "pointer-events-auto" : "pointer-events-none lg:pointer-events-auto",
          collapsed ? "w-[48px]" : "w-[200px]",
        )}
        style={{ "--mobile-drawer-x": mobileDrawerOffset } as React.CSSProperties}
      >
        {/* Logo header — 40px to match utility strip height */}
        <div className={clsx("flex h-10 shrink-0 items-center border-b border-line px-2", collapsed ? "justify-center" : "justify-between")}>
          <button type="button" onClick={() => navigate("dashboard")} aria-label={t("nav.dashboard")} className="flex min-w-0 items-center gap-2 rounded-sm text-start">
            <Image src="/logo.png" alt="" width={24} height={24} className="h-6 w-6 shrink-0 rounded-sm object-cover" />
            {!collapsed ? <span className="truncate text-sm font-semibold tracking-[-0.01em] rtl:tracking-normal">{t("app.name")}</span> : null}
          </button>
          {!collapsed ? (
            // Collapse toggle — 32×32 icon button
            <button
              type="button"
              onClick={() => setCollapsed(true)}
              className="hidden h-8 w-8 shrink-0 items-center justify-center rounded-sm text-ink-tertiary hover:bg-ink/[0.03] hover:text-ink lg:flex"
              aria-label={t("shell.collapseNavigation")}
            >
              <IconChevron className={clsx("h-4 w-4", direction === "rtl" && "rotate-180")} />
            </button>
          ) : null}
        </div>

        {/* Expand button when collapsed */}
        {collapsed ? (
          <button
            type="button"
            onClick={() => setCollapsed(false)}
            className="mx-auto mt-1 hidden h-8 w-8 shrink-0 items-center justify-center rounded-sm text-ink-tertiary hover:bg-ink/[0.03] hover:text-ink lg:flex"
            aria-label={t("shell.expandNavigation")}
          >
            <IconChevron className={clsx("h-4 w-4 rotate-180", direction === "rtl" && "!rotate-0")} />
          </button>
        ) : null}

        {/* Navigation list — 32px nav items, no brand tint */}
        <nav className="flex-1 overflow-y-auto px-1 py-2" aria-label={t("app.name")}>
          {navItems.map((item, index) => {
            const Icon = PAGE_ICONS[item.key];
            const active = page === item.key;
            const dividerBefore = isAdmin && index > 0 && item.key === "users";
            return (
              <div key={item.key}>
                {dividerBefore ? <div className="mx-2 my-2 border-t border-line" /> : null}
                <button
                  type="button"
                  title={collapsed ? item.label : undefined}
                  onClick={() => navigate(item.key)}
                  aria-current={active ? "page" : undefined}
                  className={clsx(
                    // h-8 = 32px nav item height; no pill/brand tint on active
                    "mb-px flex h-8 w-full items-center gap-2 rounded-sm text-start text-sm font-medium transition-colors duration-fast",
                    active
                      // Active: start border (2px ink), ~3% ink wash, ink text
                      ? "border-s-2 border-ink bg-ink/[0.035] ps-[6px] text-ink"
                      // Inactive: ~3% ink wash on hover, no start mark
                      : "text-ink-secondary hover:bg-ink/[0.03] hover:text-ink",
                    // Collapsed: center icon, no text
                    collapsed ? "justify-center px-0 ps-0" : active ? "" : "px-2",
                  )}
                >
                  {/* Icon: always ink-tertiary when inactive, ink when active — no brand/green */}
                  <Icon className={clsx("h-[17px] w-[17px] shrink-0", active ? "text-ink" : "text-ink-tertiary")} />
                  {!collapsed ? <span className="truncate">{item.label}</span> : null}
                </button>
              </div>
            );
          })}
        </nav>

        {/* Bottom: language toggle + preferences popover + user trigger */}
        <div className="relative mt-auto border-t border-line p-1">
          {!collapsed ? (
            <div className="mb-1.5 px-1">
              <LanguageToggle />
            </div>
          ) : null}

          {/* Preferences popover — non-modal listbox-style popup, 6px radius, ~220px wide */}
          {preferencesOpen ? (
            <div
              ref={preferencesRef}
              // Non-modal popup: no role=dialog (which implies modality); just a region
              id="preferences-popup"
              className={clsx(
                "absolute bottom-[calc(100%-2px)] z-popover w-[220px] rounded-md border border-line bg-surface p-2 shadow-lg",
                "start-1",
              )}
            >
              {collapsed ? (
                <div className="mb-2 border-b border-line px-1 pb-2">
                  <LanguageToggle />
                </div>
              ) : null}
              <p className="px-2 pb-1.5 pt-0.5 text-xs font-medium text-ink-tertiary">{t("shell.appearance")}</p>
              {/* ThemeControls: 4px segment buttons */}
              <ThemeControls />
              {health?.version ? <p className="px-2 pt-1.5 text-xs text-ink-tertiary">Smarlux v{health.version}</p> : null}
              <div className="my-2 border-t border-line" />
              <button
                type="button"
                onClick={() => void logout()}
                className="flex h-8 w-full items-center gap-2 rounded-sm px-2 text-start text-xs font-medium text-danger hover:bg-danger-subtle"
              >
                <IconLogout className="h-4 w-4" />{t("nav.logout")}
              </button>
            </div>
          ) : null}

          {/* User/preferences trigger — 32px min-height */}
          <button
            ref={preferencesTriggerRef}
            type="button"
            onClick={() => setPreferencesOpen((value) => !value)}
            aria-expanded={preferencesOpen}
            aria-controls="preferences-popup"
            aria-haspopup="true"
            aria-label={user.full_name ?? user.email}
            className={clsx(
              "flex min-h-8 w-full items-center rounded-sm px-1 text-start hover:bg-ink/[0.03]",
              collapsed ? "justify-center" : "gap-2",
            )}
          >
            <span className="grid h-6 w-6 shrink-0 place-items-center rounded-sm bg-ink/[0.06] text-xs font-semibold uppercase text-ink">
              {user.email.slice(0, 2)}
            </span>
            {!collapsed ? (
              <>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-xs font-medium text-ink">{user.full_name ?? user.email}</span>
                  <span dir="ltr" className="block truncate text-left text-xs text-ink-tertiary">{user.email}</span>
                </span>
                <span className="text-xs text-ink-tertiary">•••</span>
              </>
            ) : null}
          </button>
        </div>
      </aside>

      {/* Main content area */}
      <section className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-[rgb(var(--bg-primary))]">
        {/* Utility strip — 40px, no duplicate page title for dashboard/projects/tasks/users/monitoring */}
        <header className="flex h-10 shrink-0 items-center gap-2 border-b border-line px-4 lg:px-5">
          {/* Mobile hamburger — 32×32 */}
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            aria-label={t("shell.openNavigation")}
            aria-controls="primary-navigation"
            aria-expanded={mobileOpen}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-sm text-ink-secondary hover:bg-ink/[0.03] lg:hidden"
          >
            <IconMenu className="h-[18px] w-[18px]" />
          </button>

          {/* Page identity label — only for studio in this spec */}
          <div className="min-w-0 flex-1">
            {showStripLabel ? (
              <p className="truncate text-sm font-medium text-ink-secondary">{activeNavItem?.label}</p>
            ) : null}
          </div>

          {/* Project selector — max 220px, 32px height */}
          {pageUsesProjectContext ? (
            <label className="relative flex min-w-0 max-w-[220px] items-center gap-2 text-xs text-ink-tertiary">
              <span className="hidden sm:inline">{t("shell.activeProject")}</span>
              <span className="relative inline-flex h-8 min-w-[120px] items-center justify-between gap-1.5 rounded-sm border border-line bg-surface px-2 text-xs font-medium text-ink">
                <select
                  aria-label={t("shell.selectProject")}
                  disabled={projectsLoading || !projects.length}
                  className="absolute inset-0 z-10 cursor-pointer opacity-0 disabled:cursor-not-allowed"
                  value={selectedProjectId ?? ""}
                  onChange={(event) => setSelectedProjectId(event.target.value || null)}
                >
                  {!projects.length ? <option value="">{t("shell.noProject")}</option> : null}
                  {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
                </select>
                <span className="truncate">{selectedProject?.name ?? t("shell.noProject")}</span>
                <svg className="h-3 w-3 shrink-0 text-ink-tertiary" viewBox="0 0 20 20" fill="currentColor" aria-hidden>
                  <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
                </svg>
              </span>
            </label>
          ) : null}
        </header>

        <main className={clsx("min-h-0 flex-1 overflow-x-hidden overflow-y-auto", page !== "studio" && "px-4 lg:px-6")}>
          <ErrorBoundary resetKey={page} fallbackTitle={t("common.unexpectedError")} fallbackMessage={t("common.errorBoundaryMessage")} retryLabel={t("common.retry")}>
            {page === "dashboard" && <DashboardPanel token={token} projects={projects} isAdmin={isAdmin} onNavigate={(next) => navigate(next)} />}
            {page === "projects" && <ProjectsPanel token={token} projects={projects} selectedProjectId={selectedProjectId} canManageProjects={isAdmin} onSelectProject={setSelectedProjectId} onProjectsRefresh={refreshProjects} />}
            {page === "studio" && <ContentStudioPanel token={token} selectedProjectId={selectedProjectId} />}
            {page === "tasks" && <TasksPanel token={token} canReview={isAdmin} />}
            {page === "users" && isAdmin && <UsersPanel token={token} isAdmin={isAdmin} currentUserId={user.id} />}
            {page === "monitoring" && isAdmin && <MonitoringPanel token={token} />}
          </ErrorBoundary>
        </main>
      </section>
    </div>
  );
}
