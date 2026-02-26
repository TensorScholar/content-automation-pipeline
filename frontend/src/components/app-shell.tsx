"use client";

import { useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { apiRequest } from "@/lib/api";
import { User, Project } from "@/types/models";
import { useAuth } from "@/providers/auth-provider";
import { useI18n } from "@/i18n/provider";
import { LanguageToggle } from "./language-toggle";
import { ErrorBoundary } from "./ui/error-boundary";
import { useHasRole } from "./ui/role-guard";
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

  const refreshProjects = async () => {
    setProjectsLoading(true);
    try {
      const payload = await apiRequest<Project[]>("/projects", { token });
      setProjects(payload);
      if (payload.length === 0) {
        setSelectedProjectId(null);
      } else if (!payload.some((p) => p.id === selectedProjectId)) {
        setSelectedProjectId(payload[0].id);
      }
    } catch {
      setProjects([]);
      setSelectedProjectId(null);
    } finally {
      setProjectsLoading(false);
    }
  };

  useEffect(() => { void refreshProjects(); }, [token]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch health for version in sidebar
  useEffect(() => {
    apiRequest<{ version?: string }>("/system/health", { token }).then(setHealth).catch(() => { });
  }, [token]);

  const navigate = (next: AppPage) => { setPage(next); setMobileOpen(false); };

  const drawerTransform = mobileOpen
    ? "translateX(0)"
    : direction === "rtl" ? "translateX(100%)" : "translateX(-100%)";

  const sidebarW = collapsed ? 72 : 272;

  return (
    <div className="min-h-screen bg-surface-secondary" dir={direction}>

      {/* ── Mobile overlay ── */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-overlay animate-fade-in bg-ink/40 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* ═══ SIDEBAR — distinct style with subtle gradient bg ═══ */}
      <aside
        className={clsx(
          "fixed inset-y-0 start-0 z-modal flex flex-col border-e border-border",
          "transition-all duration-slow ease-apple overflow-hidden",
          collapsed ? "w-[72px]" : "w-[272px]",
          "lg:translate-x-0",
        )}
        style={{
          background: "linear-gradient(180deg, #FAFCFC 0%, #F4F7F7 100%)",
          transform: typeof window !== "undefined" && window.innerWidth < 1024 ? drawerTransform : undefined,
        }}
      >
        {/* Sidebar Header — brand + user avatar */}
        <div className={clsx(
          "flex shrink-0 items-center border-b border-border/60 px-5",
          collapsed ? "h-14 justify-center" : "h-16 justify-between",
        )}>
          {!collapsed && (
            <div className="flex items-center gap-3 min-w-0">
              <div className="h-8 w-8 shrink-0 rounded-lg bg-brand flex items-center justify-center">
                <span className="text-body-sm font-bold text-white">
                  {(user.full_name ?? user.email).charAt(0).toUpperCase()}
                </span>
              </div>
              <div className="min-w-0">
                <p className="text-body-sm font-bold text-ink truncate">{user.full_name ?? user.email}</p>
                <p className="text-[11px] text-ink-tertiary truncate">{user.email}</p>
              </div>
            </div>
          )}
          {collapsed && (
            <div className="h-8 w-8 rounded-lg bg-brand flex items-center justify-center">
              <span className="text-body-sm font-bold text-white">
                {(user.full_name ?? user.email).charAt(0).toUpperCase()}
              </span>
            </div>
          )}
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto px-3 py-3 space-y-0.5">
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
                  "w-full flex items-center gap-3 rounded-xl px-3 py-2.5 text-start",
                  "transition-all duration-fast",
                  active
                    ? "bg-brand text-white shadow-sm"
                    : "text-ink-secondary hover:bg-white/70 hover:text-ink",
                )}
              >
                <Icon className="h-5 w-5 shrink-0" />
                {!collapsed && (
                  <span className="text-[14px] font-medium truncate">{item.label}</span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Project Selector — moved up, compact */}
        <div className={clsx("border-t border-border/60 px-3 py-3", collapsed && "hidden")}>
          <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-tertiary mb-1.5">
            {t("shell.activeProject")}
          </p>
          <select
            disabled={projectsLoading || projects.length === 0}
            className="w-full rounded-lg border border-border bg-white px-3 py-1.5 text-[13px] text-ink outline-none focus:border-brand transition-colors duration-fast disabled:opacity-50"
            value={selectedProjectId ?? ""}
            onChange={(e) => setSelectedProjectId(e.target.value || null)}
          >
            {projects.length === 0 && <option value="">{t("shell.noProject")}</option>}
            {projects.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </div>

        {/* Logout — right after project selector */}
        <div className={clsx(
          "border-t border-border/60 px-3 py-2 flex items-center",
          collapsed ? "justify-center" : "gap-3",
        )}>
          <button
            type="button"
            title={t("nav.logout")}
            onClick={() => void logout()}
            className={clsx(
              "flex items-center gap-2 rounded-lg transition-colors duration-fast hover:bg-danger/8 hover:text-danger",
              collapsed ? "h-9 w-9 justify-center text-ink-tertiary" : "w-full px-3 py-2 text-ink-secondary text-[13px]",
            )}
          >
            <IconLogout className="h-4 w-4 shrink-0" />
            {!collapsed && <span className="font-medium">{t("nav.logout")}</span>}
          </button>
        </div>

        {/* API Version — at very bottom of sidebar */}
        <div className={clsx("py-2 text-center", collapsed && "hidden")}>
          <p className="text-[10px] text-ink-tertiary/60 font-mono">
            {t("dashboard.apiVersion")} · {health?.version ?? "1.0.0"}
          </p>
        </div>
      </aside>

      {/* ═══ MAIN CONTENT AREA ═══ */}
      <div
        className="flex min-h-screen flex-col transition-all duration-slow ease-apple"
        style={{ marginInlineStart: `${sidebarW}px` }}
      >
        {/* ── Top Header — no duplicate page title, just language + collapse ── */}
        <header className="sticky top-0 z-sticky flex h-14 shrink-0 items-center gap-4 border-b border-border bg-surface/80 backdrop-blur-md px-4 lg:px-6">
          {/* Mobile menu */}
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            className="lg:hidden h-9 w-9 flex items-center justify-center rounded-lg text-ink-secondary hover:bg-surface-tertiary transition-colors duration-fast"
          >
            <IconMenu className="h-5 w-5" />
          </button>

          {/* Collapse toggle for desktop — also in header for quick access */}
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            className="hidden lg:flex h-8 w-8 items-center justify-center rounded-lg text-ink-tertiary hover:bg-surface-tertiary hover:text-ink transition-colors duration-fast"
            aria-label="Toggle sidebar"
          >
            <IconChevron className={clsx("h-4 w-4 transition-transform duration-normal ease-apple", collapsed ? "rotate-180" : "")} />
          </button>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Language Toggle with globe icon */}
          <div className="flex items-center gap-2">
            <IconGlobe className="h-4 w-4 text-ink-tertiary" />
            <LanguageToggle />
          </div>
        </header>

        {/* ── Panel Content ── */}
        <main className="flex-1 px-4 py-5 lg:px-6 lg:py-6">
          <ErrorBoundary>
            {page === "dashboard" && <DashboardPanel token={token} projects={projects} onNavigate={navigate as unknown as (page: string) => void} />}
            {page === "projects" && (
              <ProjectsPanel
                token={token}
                projects={projects}
                selectedProjectId={selectedProjectId}
                onSelectProject={setSelectedProjectId}
                onProjectsRefresh={refreshProjects}
              />
            )}
            {page === "studio" && <ContentStudioPanel token={token} selectedProjectId={selectedProjectId} />}
            {page === "tasks" && <TasksPanel token={token} />}
            {page === "users" && <UsersPanel token={token} isAdmin={isAdmin} />}
            {page === "monitoring" && <MonitoringPanel token={token} />}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
