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

export function AppShell({ token, user }: AppShellProps) {
  const { logout, isAdmin } = useAuth();
  const { t, direction } = useI18n();

  const [page, setPage] = useState<AppPage>("dashboard");
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [projectsLoading, setProjectsLoading] = useState(true);

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

  const navigate = (next: AppPage) => { setPage(next); setMobileOpen(false); };

  /* Mobile drawer transform — direction-aware to avoid Tailwind class conflicts */
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

      {/* ═══ SIDEBAR ═══ */}
      <aside
        className={clsx(
          "fixed inset-y-0 start-0 z-modal flex flex-col bg-surface border-e border-border",
          "transition-all duration-slow ease-apple overflow-hidden",
          collapsed ? "w-[72px]" : "w-[272px]",
          "lg:translate-x-0",
        )}
        style={{ transform: typeof window !== "undefined" && window.innerWidth < 1024 ? drawerTransform : undefined }}
      >
        {/* Sidebar Header */}
        <div className={clsx(
          "flex h-16 shrink-0 items-center border-b border-border px-4",
          collapsed ? "justify-center" : "justify-between",
        )}>
          {!collapsed && (
            <div className="min-w-0">
              <p className="text-heading-sm text-ink truncate">{t("app.name")}</p>
              <p className="text-body-sm text-ink-tertiary truncate">{user.email}</p>
            </div>
          )}
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            className="hidden lg:flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-ink-tertiary transition-colors duration-fast hover:bg-surface-tertiary hover:text-ink"
            aria-label="Toggle sidebar"
          >
            <IconChevron className={clsx("h-4 w-4 transition-transform duration-normal ease-apple", collapsed ? "rotate-180" : "")} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
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
                  "transition-colors duration-fast",
                  active
                    ? "bg-brand text-white"
                    : "text-ink-secondary hover:bg-surface-alt hover:text-ink",
                )}
              >
                <Icon className="h-5 w-5 shrink-0" />
                {!collapsed && (
                  <span className="text-body-md font-medium truncate">{item.label}</span>
                )}
                {active && !collapsed && (
                  <span className="ms-auto h-1.5 w-1.5 rounded-full bg-ink-inverse/60" />
                )}
              </button>
            );
          })}
        </nav>

        {/* Project Selector */}
        <div className={clsx("border-t border-border px-3 py-4", collapsed && "hidden")}>
          <p className="text-body-sm font-semibold uppercase tracking-wider text-ink-tertiary mb-2">
            {t("shell.activeProject")}
          </p>
          <select
            disabled={projectsLoading || projects.length === 0}
            className="w-full rounded-lg border border-border bg-surface-secondary px-3 py-2 text-body-md text-ink outline-none focus:border-border-focus transition-colors duration-fast disabled:opacity-50"
            value={selectedProjectId ?? ""}
            onChange={(e) => setSelectedProjectId(e.target.value || null)}
          >
            {projects.length === 0 && <option value="">{t("shell.noProject")}</option>}
            {projects.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </div>

        {/* User footer */}
        <div className={clsx(
          "border-t border-border px-3 py-3 flex items-center",
          collapsed ? "justify-center" : "gap-3",
        )}>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <p className="text-body-sm font-semibold text-ink truncate">{user.full_name ?? user.email}</p>
              {user.is_superuser && (
                <span className="inline-block mt-0.5 rounded-full bg-brand/10 px-2 py-0.5 text-body-sm font-semibold text-brand">
                  {t("role.manager")}
                </span>
              )}
            </div>
          )}
          <button
            type="button"
            title={t("nav.logout")}
            onClick={() => void logout()}
            className="h-8 w-8 shrink-0 flex items-center justify-center rounded-lg text-ink-tertiary transition-colors duration-fast hover:bg-danger-subtle hover:text-danger"
          >
            <IconLogout className="h-4 w-4" />
          </button>
        </div>
      </aside>

      {/* ═══ MAIN CONTENT AREA ═══ */}
      <div
        className="flex min-h-screen flex-col transition-all duration-slow ease-apple"
        style={{ marginInlineStart: `${sidebarW}px` }}
      >
        {/* ── Top Header ── */}
        <header className="sticky top-0 z-sticky flex h-16 shrink-0 items-center gap-4 border-b border-border bg-surface/80 backdrop-blur-md px-4 lg:px-6">
          {/* Mobile menu */}
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            className="lg:hidden h-9 w-9 flex items-center justify-center rounded-lg text-ink-secondary hover:bg-surface-tertiary transition-colors duration-fast"
          >
            <IconMenu className="h-5 w-5" />
          </button>

          {/* Page title */}
          <h1 className="text-heading-sm text-ink">
            {navItems.find((n) => n.key === page)?.label ?? ""}
          </h1>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Language Toggle */}
          <LanguageToggle />

          {/* User badge */}
          <div className="hidden sm:flex items-center gap-2 rounded-full border border-border bg-surface-secondary px-3 py-1.5">
            <div className="h-6 w-6 rounded-full bg-brand flex items-center justify-center">
              <span className="text-body-sm font-bold text-ink-inverse">
                {(user.full_name ?? user.email).charAt(0).toUpperCase()}
              </span>
            </div>
            <span className="text-body-sm font-medium text-ink max-w-[120px] truncate">
              {user.full_name ?? user.email}
            </span>
          </div>
        </header>

        {/* ── Panel Content ── */}
        <main className="flex-1 px-4 py-6 lg:px-6 lg:py-8">
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
