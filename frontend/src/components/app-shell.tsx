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

  const sidebarW = collapsed ? 96 : 284; // Adjusted for the 24px lateral margin geometry

  return (
    <div className="min-h-screen bg-surface-secondary" dir={direction}>

      {/* ── Mobile overlay ── */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-overlay animate-fade-in bg-ink/40 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* ═══ SIDEBAR — FLOATING GLASS ISLAND ═══ */}
      <aside
        className={clsx(
          "fixed top-6 bottom-6 start-6 z-modal flex flex-col h-[calc(100vh-48px)]",
          "rounded-[3rem] border border-white/15 shadow-[0_20px_50px_rgba(0,0,0,0.5),0_10px_10px_rgba(0,0,0,0.2),inset_0_1px_1px_rgba(255,255,255,0.1)]",
          "transition-all duration-700 overflow-hidden",
          collapsed ? "w-[72px]" : "w-[260px]",
          "lg:translate-x-0 bg-gradient-to-b from-[#0d3328]/95 via-[#051c15]/98 to-[#020d0a] backdrop-blur-2xl",
        )}
        style={{
          transform: typeof window !== "undefined" && window.innerWidth < 1024 ? drawerTransform : undefined,
          transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)"
        }}
      >
        {/* Subtle radial glass convex reflection layer */}
        <div className="absolute inset-0 z-[-1] rounded-[3rem] bg-[radial-gradient(ellipse_at_top_start,rgba(255,255,255,0.05),transparent_50%)] pointer-events-none" />

        {/* Sidebar Header — brand + collapse toggle */}
        <div className={clsx(
          "flex shrink-0 items-center px-8 py-6 relative",
          collapsed ? "justify-center px-0" : "justify-between",
        )}>
          {!collapsed && (
            <div className="flex items-center gap-3">
              {/* Primary Brand Logo - Transparent */}
              <img src="/logo.png" alt="Smarlux" className="h-[34px] w-auto shrink-0 object-contain invert text-emerald-50 drop-shadow-md" />
              <span className="text-white font-black tracking-tighter text-xl pt-0.5">Smarlux</span>
            </div>
          )}
          {collapsed && (
            <img src="/logo.png" alt="Smarlux" className="h-8 w-8 shrink-0 object-contain invert drop-shadow-md" />
          )}

          {/* Toggle Button Clean Integration */}
          {!collapsed && (
            <button
              type="button"
              onClick={() => setCollapsed(true)}
              className="absolute inset-inline-end-6 hidden lg:flex h-8 w-8 items-center justify-center rounded-xl text-emerald-100/50 hover:bg-white/10 hover:text-emerald-50 transition-all duration-500"
              aria-label="Collapse sidebar"
            >
              <IconMenu className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Nav — Strict Axis alignment with Premium Spring interactions */}
        <nav className="flex-1 overflow-y-auto pt-2 space-y-1">
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
                  "relative flex items-center gap-4 px-5 py-3 mx-4 rounded-2xl transition-all duration-700 text-start w-[calc(100%-32px)] group overflow-hidden",
                  active
                    ? "bg-white/10 text-white font-medium shadow-sm"
                    : "text-emerald-100/60 hover:bg-white/5 hover:text-emerald-50",
                  collapsed && "justify-center px-0 mx-4 w-auto"
                )}
                style={{ transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)" }}
              >
                {/* Active Indicator Pill (Left edge) */}
                {active && !collapsed && (
                  <span className="absolute inset-inline-start-0 top-1/2 -translate-y-1/2 h-5 w-1 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
                )}

                {/* Active Glass blur layer */}
                {active && <div className="absolute inset-0 backdrop-blur-md z-[-1]" />}

                <div className="w-6 h-6 flex items-center justify-center shrink-0">
                  <Icon className={clsx("h-5 w-5 transition-transform duration-700 group-hover:scale-110", active && "drop-shadow-md")} />
                </div>
                {!collapsed && (
                  <span className="text-[14px] truncate translate-y-[0.5px]">{item.label}</span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Dynamic Footer Section (Card-in-Card style) */}
        <div className="mt-auto px-0 pb-6 flex flex-col gap-2">

          {/* Project Selector — Recessed Card within a Card */}
          <div className={clsx("w-auto bg-black/30 rounded-2xl p-4 mx-4 mb-2 border border-white/5", collapsed && "hidden")}>
            <select
              disabled={projectsLoading || projects.length === 0}
              title={t("shell.activeProject")}
              className="w-full rounded-xl bg-transparent text-[13px] font-medium text-emerald-50 outline-none focus:ring-2 focus:ring-emerald-500/50 transition-all duration-500 disabled:opacity-50 appearance-none cursor-pointer RTL-caret"
              value={selectedProjectId ?? ""}
              onChange={(e) => setSelectedProjectId(e.target.value || null)}
            >
              {projects.length === 0 && <option value="" className="bg-slate-900">{t("shell.noProject")}</option>}
              {projects.map((p) => (
                <option key={p.id} value={p.id} className="bg-slate-900 font-sans">{p.name}</option>
              ))}
            </select>
          </div>

          <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent my-1" />

          {/* User Profile & Logout - Clean Row */}
          <div className={clsx("flex items-center justify-between px-6", collapsed ? "flex-col gap-4 px-0" : "gap-3")}>
            {!collapsed && (
              <div className="flex items-center gap-3 min-w-0 flex-1">
                <div className="h-9 w-9 shrink-0 rounded-full bg-white/10 backdrop-blur-md flex items-center justify-center border border-white/10 text-white text-[13px] font-bold uppercase shadow-sm">
                  {user.email.substring(0, 2)}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-[14px] font-semibold text-white truncate drop-shadow-sm">{user.full_name ?? "Manager"}</p>
                  <p className="text-xs font-semibold text-emerald-100/60 truncate">{user.email}</p>
                </div>
              </div>
            )}

            <button
              type="button"
              title={t("nav.logout")}
              onClick={() => void logout()}
              className={clsx(
                "flex items-center justify-center rounded-full transition-all duration-500",
                "text-emerald-100/40 hover:text-red-400 hover:bg-white/5",
                collapsed ? "h-10 w-10 mx-auto" : "h-9 w-9 shrink-0"
              )}
            >
              <IconLogout className="h-4 w-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* Responsive toggle overlay logic handled natively */}

      {/* ═══ MAIN CONTENT AREA (Offset by Sidebar width + Margin Geometry) ═══ */}
      <div
        className="flex min-h-screen flex-col transition-all overflow-hidden"
        style={{
          marginInlineStart: `${sidebarW}px`,
          transitionDuration: "700ms",
          transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)"
        }}
      >
        {/* ── Top Header — no duplicate page title, just language + collapse ── */}
        <header className="sticky top-0 z-sticky flex h-14 shrink-0 items-center gap-4 border-b border-gray-100 bg-white/80 backdrop-blur-md px-4 lg:px-6">
          {/* Mobile menu */}
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            className="lg:hidden h-9 w-9 flex items-center justify-center rounded-lg text-gray-400 hover:bg-gray-100 transition-colors"
          >
            <IconMenu className="h-5 w-5" />
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
