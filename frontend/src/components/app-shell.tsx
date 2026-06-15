"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
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
  IconDashboard, IconProjects, IconStudio, IconTasks,
  IconUsers, IconMonitoring, IconChevron, IconMenu, IconLogout,
} from "./shell/nav-icons";

type AppPage = "dashboard" | "projects" | "studio" | "tasks" | "users" | "monitoring";

interface AppShellProps { token: string; user: User }

const PAGE_ICONS: Record<AppPage, (p: { className?: string }) => React.ReactElement> = {
  dashboard: IconDashboard,
  projects: IconProjects,
  studio: IconStudio,
  tasks: IconTasks,
  users: IconUsers,
  monitoring: IconMonitoring,
};

/* Globe icon for language toggle */
function IconGlobe({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className} stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10" cy="10" r="8" /><path d="M2 10h16" /><path d="M10 2a13 13 0 0 1 0 16 13 13 0 0 1 0-16z" />
    </svg>
  );
}

function ThemeSwitcher() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  const currentTheme = mounted ? theme ?? "system" : "system";

  return (
    <div className="macos-segmented relative z-40 flex h-9 items-center gap-0.5 rounded-[10px] p-0.5">
      {(["system", "dark", "light"] as const).map((mode) => (
        <button
          key={mode}
          type="button"
          onClick={() => setTheme(mode)}
          className={clsx(
            "min-h-[32px] rounded-[8px] px-3 text-[13px] font-medium tracking-normal transition-[background-color,color,box-shadow,transform] duration-150 [transition-timing-function:cubic-bezier(0.16,1,0.3,1)] focus-visible:outline-none",
            currentTheme === mode
              ? "bg-white text-slate-950 shadow-[0_1px_2px_rgb(0_0_0/0.06)] dark:bg-white/10 dark:text-white"
              : "text-slate-500 hover:bg-black/[0.03] hover:text-slate-900 dark:text-gray-400 dark:hover:bg-white/[0.05] dark:hover:text-gray-100"
          )}
          aria-pressed={currentTheme === mode}
        >
          {mode === "system" ? "System" : mode === "dark" ? "Dark" : "Light"}
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
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [projectsLoading, setProjectsLoading] = useState(true);
  const [health, setHealth] = useState<{ version?: string } | null>(null);

  const navItems = useMemo(() => {
    const base: Array<{ key: AppPage; label: string; adminOnly?: boolean }> = [
      { key: "dashboard", label: t("nav.dashboard") },
      { key: "projects", label: t("nav.projects") },
      { key: "studio", label: t("nav.studio") },
      { key: "tasks", label: t("nav.tasks") },
      { key: "users", label: t("nav.users"), adminOnly: true },
      { key: "monitoring", label: t("nav.monitoring"), adminOnly: true },
    ];
    return base.filter((item) => !item.adminOnly || isAdmin);
  }, [isAdmin, t]);

  useEffect(() => {
    if (!isAdmin && (page === "users" || page === "monitoring")) {
      setPage("dashboard");
    }
  }, [isAdmin, page]);

  const refreshProjects = useCallback(async (signal?: AbortSignal) => {
    setProjectsLoading(true);
    try {
      const payload = await apiRequest<Project[]>("/projects", { token, signal });
      if (signal?.aborted) return;
      setProjects(payload);
      if (payload.length === 0) {
        setSelectedProjectId(null);
      } else {
        setSelectedProjectId((currentProjectId) =>
          payload.some((p) => p.id === currentProjectId) ? currentProjectId : payload[0].id
        );
      }
    } catch {
      if (signal?.aborted) return;
      setProjects([]);
      setSelectedProjectId(null);
    } finally {
      if (signal?.aborted) return;
      setProjectsLoading(false);
    }
  }, [token]);

  useEffect(() => {
    const controller = new AbortController();
    void refreshProjects(controller.signal);
    return () => controller.abort();
  }, [refreshProjects]);

  // Fetch health for version in sidebar
  useEffect(() => {
    const controller = new AbortController();
    apiRequest<{ version?: string }>("/system/health", { token, signal: controller.signal })
      .then((payload) => {
        if (!controller.signal.aborted) setHealth(payload);
      })
      .catch(() => { });
    return () => controller.abort();
  }, [token]);

  const navigate = (next: AppPage) => { setPage(next); setMobileOpen(false); };

  const drawerTransform = mobileOpen
    ? "translateX(0)"
    : direction === "rtl" ? "translateX(100%)" : "translateX(-100%)";

  return (
    <div className="macos-content-scope macos-app-bg flex h-dvh min-h-0 w-full overflow-hidden pt-10 text-sm tracking-normal text-ink" dir={direction}>
      <div
        data-tauri-drag-region
        aria-hidden="true"
        className="macos-titlebar fixed inset-x-0 top-0 z-[80] h-10"
      />

      {/* ── Mobile overlay ── */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-overlay animate-fade-in bg-ink/40 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={clsx(
          "fixed start-3 top-12 bottom-3 z-modal flex flex-col",
          "lg:relative lg:start-auto lg:top-auto lg:bottom-auto lg:m-3 lg:me-0 lg:mt-2 lg:h-[calc(100dvh-60px)] lg:shrink-0",
          "macos-sidebar rounded-[18px] border",
          "transition-all duration-300 overflow-hidden",
          collapsed ? "w-[64px]" : "w-[248px]",
          "lg:translate-x-0",
        )}
        style={{
          transform: typeof window !== "undefined" && window.innerWidth < 1024 ? drawerTransform : undefined,
          transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)"
        }}
      >
        {/* Sidebar Header */}
        <div className={clsx(
          "relative flex shrink-0 items-center border-b border-black/5 px-4 py-3 dark:border-white/10",
          "justify-center flex-col", // Always center the contents
        )}>
          {!collapsed && (
            <div className="flex items-center gap-3">
              {/* Refined Typographic Logo — Centered */}
              <span className="truncate text-[20px] font-semibold tracking-tight text-gray-950 dark:text-gray-100">Smarlux</span>
            </div>
          )}
          {collapsed && (
            <span className="pt-0.5 text-[18px] font-semibold tracking-tight text-gray-950 dark:text-gray-100">S</span>
          )}

          {/* Toggle Button Clean Integration (Absolutely Positioned when open, centered below when closed) */}
          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            className={clsx(
              "hidden h-7 w-7 items-center justify-center rounded-md text-gray-500 transition-colors duration-150 hover:bg-black/5 hover:text-gray-950 dark:text-gray-400 dark:hover:bg-white/10 dark:hover:text-white lg:flex",
              collapsed ? "absolute inset-x-0 mx-auto mt-16" : "absolute end-3 top-1/2 -translate-y-1/2"
            )}
            aria-label="Toggle sidebar"
          >
            <IconMenu className={clsx("h-4 w-4 transition-transform duration-500", collapsed && "rotate-180")} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto p-2">
          {navItems.map((item) => {
            const Icon = PAGE_ICONS[item.key];
            const active = page === item.key;
            return (
              <button
                key={item.key}
                type="button"
                title={collapsed ? item.label : undefined}
                onClick={() => navigate(item.key)}
                className={clsx(
                  "group relative mb-1 flex h-8 w-full items-center gap-2.5 rounded-md px-2.5 text-start transition-colors duration-150",
                  active
                    ? "bg-black/[0.08] text-gray-950 font-medium dark:bg-white/10 dark:text-gray-100"
                    : "text-gray-600 hover:bg-black/5 hover:text-gray-950 dark:text-gray-300 dark:hover:bg-white/10 dark:hover:text-white",
                  collapsed && "justify-center px-0"
                )}
                style={{ transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)" }}
              >
                <div className="flex h-4 w-4 shrink-0 items-center justify-center">
                  <Icon className={clsx("h-4 w-4 transition-colors duration-150", active ? "text-gray-950 dark:text-gray-100" : "text-gray-500 dark:text-gray-400")} />
                </div>
                {!collapsed && (
                  <span className="translate-y-[0.5px] text-[13px] font-medium">{item.label}</span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="mt-auto flex flex-col gap-2 border-t border-black/5 px-0 pb-3 pt-3 dark:border-white/10">

          {/* Project Selector */}
          <div className={clsx("relative mx-3 mb-2 w-auto cursor-pointer rounded-[14px] border border-black/5 bg-white p-2 shadow-[inset_0_1px_0_rgb(255_255_255/0.75),0_10px_22px_-20px_rgb(0_0_0/0.55)] transition-colors hover:bg-white dark:border-white/10 dark:bg-surface dark:shadow-none dark:hover:bg-surface-alt", collapsed && "hidden")}>
            <div className="pointer-events-none mb-1 px-1 text-[10px] font-medium uppercase tracking-normal text-gray-500 dark:text-gray-400">
              {t("shell.activeProject")}
            </div>
            <div className="relative flex justify-between items-center w-full px-1">
              <select
                disabled={projectsLoading || projects.length === 0}
                className="w-full absolute inset-0 opacity-0 cursor-pointer z-10"
                value={selectedProjectId ?? ""}
                onChange={(e) => setSelectedProjectId(e.target.value || null)}
              >
                {projects.length === 0 && <option value="" className="bg-black/80">{t("shell.noProject")}</option>}
                {projects.map((p) => (
                  <option key={p.id} value={p.id} className="bg-black/80 font-sans">{p.name}</option>
                ))}
              </select>
              <span className="truncate pr-6 text-[12px] font-medium text-gray-950 dark:text-gray-100">
                {projects.find(p => p.id === selectedProjectId)?.name || t("shell.noProject")}
              </span>
              <svg className="w-4 h-4 text-gray-500 dark:text-gray-400 shrink-0 opacity-80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6" /></svg>
            </div>
          </div>

          <div className="my-1 h-px w-full bg-black/5 dark:bg-white/10" />

          {/* User Profile & Logout - Clean Row */}
          <div className={clsx("flex items-center justify-between px-3", collapsed ? "flex-col gap-3 px-0" : "gap-2")}>
            {!collapsed && (
              <div className="flex items-center gap-3 min-w-0 flex-1">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-black/5 bg-black/5 text-[11px] font-semibold uppercase text-gray-950 dark:border-white/10 dark:bg-white/10 dark:text-gray-100">
                  {user.email.substring(0, 2)}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[12px] font-medium text-gray-950 dark:text-gray-100">{user.full_name ?? "Manager"}</p>
                  <p className="truncate text-[11px] font-medium text-gray-500 dark:text-gray-400">{user.email}</p>
                </div>
              </div>
            )}

            <button
              type="button"
              title={t("nav.logout")}
              onClick={() => void logout()}
              className={clsx(
                "flex items-center justify-center rounded-full transition-all duration-500",
                "text-gray-500 hover:text-red-500 hover:bg-black/5 dark:text-gray-400 dark:hover:text-red-400 dark:hover:bg-white/10",
                collapsed ? "mx-auto h-8 w-8" : "h-7 w-7 shrink-0"
              )}
            >
              <IconLogout className="h-4 w-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* Responsive toggle overlay logic handled natively */}

      {/* ═══ MAIN CONTENT AREA (Offset by Sidebar width + Margin Geometry) ═══ */}
      <div className="macos-main-material relative m-0 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden border-s border-black/5 transition-all duration-300 dark:border-white/10 lg:m-3 lg:ms-0 lg:mt-2 lg:rounded-[18px]">
        {/* ── Header Utilities ─ */}
        <header className="relative z-50 flex h-12 shrink-0 items-center justify-end border-b border-black/5 bg-white px-3 dark:border-white/10 dark:bg-surface lg:px-4">
          <div className="flex items-center gap-2">
            <ThemeSwitcher />
            <div className="relative z-50 flex h-9 items-center gap-1.5 rounded-[10px] border border-black/5 bg-white px-2 shadow-none dark:border-white/10 dark:bg-white/[0.06]">
              <IconGlobe className="h-3.5 w-3.5 text-gray-500 dark:text-gray-400" />
              <LanguageToggle />
            </div>
          </div>
        </header>

        {/* ── Panel Content ── */}
        <main className={clsx("min-h-0 flex-1 overflow-x-hidden overflow-y-auto", page !== "studio" && "px-3 pb-4 pt-3 lg:px-4 lg:pb-4 lg:pt-3")}>
          <ErrorBoundary resetKey={page}>
            {page === "dashboard" && <DashboardPanel token={token} projects={projects} isAdmin={isAdmin} onNavigate={navigate as unknown as (page: string) => void} />}
            {page === "projects" && (
              <ProjectsPanel
                token={token}
                projects={projects}
                selectedProjectId={selectedProjectId}
                canManageProjects={isAdmin}
                onSelectProject={setSelectedProjectId}
                onProjectsRefresh={refreshProjects}
              />
            )}
            {page === "studio" && <ContentStudioPanel token={token} selectedProjectId={selectedProjectId} />}
            {page === "tasks" && <TasksPanel token={token} canReview={isAdmin} />}
            {page === "users" && isAdmin && <UsersPanel token={token} isAdmin={isAdmin} currentUserId={user.id} />}
            {page === "monitoring" && isAdmin && <MonitoringPanel token={token} />}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
