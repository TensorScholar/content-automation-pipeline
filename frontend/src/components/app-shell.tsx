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

  const sidebarW = collapsed ? 104 : 312; // 280px width + 32px lateral margin (m-8)

  return (
    <div className="min-h-screen bg-[#FBFBFD]" dir={direction}>

      {/* ── Mobile overlay ── */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-overlay animate-fade-in bg-ink/40 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* ═══ SIDEBAR — 9D LIQUID EMERALD GLASS ═══ */}
      <aside
        className={clsx(
          "fixed start-6 top-1/2 -translate-y-1/2 z-modal flex flex-col h-[calc(100vh-48px)]",
          "rounded-[3.5rem] border-t border-l border-white/20 border-b border-r border-black/40",
          "shadow-[0_40px_100px_-20px_rgba(0,0,0,0.8),0_20px_40px_-10px_rgba(0,0,0,0.5),inset_0_1px_1px_rgba(255,255,255,0.2)]",
          "transition-all duration-700 overflow-hidden",
          collapsed ? "w-[72px]" : "w-[280px]",
          "lg:translate-x-0 bg-gradient-to-b from-[#0a2920]/95 via-[#041611]/98 to-[#010806] backdrop-blur-[50px]",
        )}
        style={{
          transform: typeof window !== "undefined" && window.innerWidth < 1024 ? drawerTransform : undefined,
          transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)"
        }}
      >
        {/* ── Removed the underlying radial bloop causing text-overlap shadow ── */}

        {/* Sidebar Header — brand + collapse toggle */}
        <div className={clsx(
          "flex shrink-0 items-center px-8 py-8 relative",
          "justify-center flex-col", // Always center the contents
        )}>
          {!collapsed && (
            <div className="flex items-center gap-3">
              {/* Refined Typographic Logo — Centered */}
              <span className="text-white text-3xl font-black tracking-tighter truncate">Smarlux</span>
            </div>
          )}
          {collapsed && (
            <span className="text-white text-3xl font-black tracking-tighter pt-0.5">S</span>
          )}

          {/* Toggle Button Clean Integration (Absolutely Positioned when open, centered below when closed) */}
          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            className={clsx(
              "hidden lg:flex h-8 w-8 items-center justify-center rounded-xl text-white/50 hover:bg-white/10 hover:text-white transition-all duration-500",
              collapsed ? "absolute inset-x-0 mx-auto mt-20" : "absolute right-4 top-1/2 -translate-y-1/2"
            )}
            aria-label="Toggle sidebar"
          >
            <IconMenu className={clsx("h-4 w-4 transition-transform duration-500", collapsed && "rotate-180")} />
          </button>
        </div>

        {/* Nav — Mathematical Axis alignment with VisionOS Spring interactions */}
        <nav className="flex-1 overflow-y-auto pt-2 space-y-0">
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
                  "relative flex items-center gap-4 px-6 py-4 mx-4 mb-2 rounded-[24px] transition-all duration-700 text-start w-[calc(100%-32px)] group",
                  active
                    ? "bg-white/10 text-white font-semibold shadow-[0_4px_12px_rgba(0,0,0,0.1)] backdrop-blur-md"
                    : "text-emerald-50/80 hover:bg-white/5 hover:text-white/95", // Brighter base text
                  collapsed && "justify-center px-0 mx-4 w-auto"
                )}
                style={{ transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)" }}
              >
                <div className="w-6 h-6 flex items-center justify-center shrink-0">
                  <Icon className={clsx("h-6 w-6 transition-transform duration-700 group-hover:scale-110", active ? "text-emerald-300 drop-shadow-[0_0_12px_rgba(52,211,153,0.8)]" : "text-emerald-300/90")} />
                </div>
                {!collapsed && (
                  <span className="text-[15px] translate-y-[0.5px] font-medium">{item.label}</span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Dynamic Footer Section (Card-in-Card style) */}
        <div className="mt-auto px-0 pb-6 flex flex-col gap-2">

          {/* Project Selector — High Contrast Active Project Control */}
          <div className={clsx("w-auto bg-white/10 rounded-2xl p-3 mx-5 mb-5 border border-white/20 shadow-[0_4px_12px_rgba(0,0,0,0.3)] relative backdrop-blur-md hover:bg-white/15 transition-colors cursor-pointer", collapsed && "hidden")}>
            <div className="text-[10px] uppercase tracking-widest text-[#5EEAD4] font-bold mb-1 px-1 pointer-events-none drop-shadow-sm">
              {t("shell.activeProject")}
            </div>
            <div className="relative flex justify-between items-center w-full px-1">
              <select
                disabled={projectsLoading || projects.length === 0}
                className="w-full absolute inset-0 opacity-0 cursor-pointer z-10"
                value={selectedProjectId ?? ""}
                onChange={(e) => setSelectedProjectId(e.target.value || null)}
              >
                {projects.length === 0 && <option value="" className="bg-slate-900">{t("shell.noProject")}</option>}
                {projects.map((p) => (
                  <option key={p.id} value={p.id} className="bg-slate-900 font-sans">{p.name}</option>
                ))}
              </select>
              <span className="text-[14px] font-medium text-white/90 truncate drop-shadow-sm pr-6">
                {projects.find(p => p.id === selectedProjectId)?.name || t("shell.noProject")}
              </span>
              <svg className="w-4 h-4 text-emerald-400/80 shrink-0 opacity-80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6" /></svg>
            </div>
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
        className="flex min-h-screen flex-col transition-all overflow-hidden relative"
        style={{
          marginInlineStart: `${sidebarW}px`,
          transitionDuration: "700ms",
          transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)"
        }}
      >
        {/* ── Floating Utilities ─ */}
        <div className="fixed top-8 end-8 z-50 flex items-center gap-2 bg-white/40 backdrop-blur-md rounded-2xl px-3 py-1.5 shadow-sm border border-white/50">
          <IconGlobe className="h-4 w-4 text-ink-tertiary" />
          <LanguageToggle />
        </div>

        {/* ── Panel Content ── */}
        <main className="flex-1 px-4 py-8 lg:px-12 lg:py-8 pt-10">
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
