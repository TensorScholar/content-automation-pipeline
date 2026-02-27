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

  const sidebarW = collapsed ? 72 : 240; // Reduced width (w-60)

  return (
    <div className="min-h-screen bg-surface-secondary" dir={direction}>

      {/* ── Mobile overlay ── */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-overlay animate-fade-in bg-ink/40 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* ═══ SIDEBAR — Matte Emerald Premium ═══ */}
      <aside
        className={clsx(
          "fixed inset-y-0 start-0 z-modal flex flex-col border-inline-end border-white/5",
          "transition-all duration-slow ease-apple overflow-hidden",
          collapsed ? "w-[72px]" : "w-[240px]",
          "lg:translate-x-0 bg-gradient-to-b from-[#0d3328] via-[#051c15] to-[#020d0a]",
        )}
        style={{
          transform: typeof window !== "undefined" && window.innerWidth < 1024 ? drawerTransform : undefined,
        }}
      >
        {/* Sidebar Header — brand + collapse toggle */}
        <div className={clsx(
          "flex shrink-0 items-center px-4 relative",
          collapsed ? "h-14 justify-center" : "h-16 justify-between",
        )}>
          {!collapsed && (
            <div className="flex items-center gap-3">
              {/* Primary Brand Logo - Transparent */}
              <img src="/logo.png" alt="Smarlux" className="h-8 w-auto shrink-0 object-contain invert" />
              <span className="text-emerald-50 font-bold tracking-tight text-lg">Smarlux</span>
            </div>
          )}
          {collapsed && (
            <img src="/logo.png" alt="Smarlux" className="h-8 w-8 shrink-0 object-contain invert" />
          )}

          {/* Toggle Button Clean Integration */}
          {!collapsed && (
            <button
              type="button"
              onClick={() => setCollapsed(true)}
              className="absolute inset-inline-end-4 hidden lg:flex h-8 w-8 items-center justify-center rounded-md text-emerald-100/50 hover:bg-white/10 hover:text-emerald-50 transition-colors"
              aria-label="Collapse sidebar"
            >
              <IconMenu className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Nav — Condensed & High-End */}
        <nav className="flex-1 overflow-y-auto pt-4 space-y-1">
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
                  "flex items-center gap-3 px-4 py-2 mx-2 rounded-lg transition-all text-start w-[calc(100%-16px)]",
                  active
                    ? "bg-emerald-800/40 text-white shadow-[inset_0_1px_0_0_rgba(255,255,255,0.1)] font-medium"
                    : "text-emerald-100/60 hover:bg-white/5 hover:text-emerald-50",
                  collapsed && "justify-center px-0 mx-2 w-auto"
                )}
              >
                <Icon className="h-5 w-5 shrink-0" />
                {!collapsed && (
                  <span className="text-[14px] truncate">{item.label}</span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Dynamic Footer Section */}
        <div className="mt-auto border-t border-white/10 pt-4 mb-4 mx-4 flex flex-col gap-3">

          {/* User Profile */}
          {!collapsed && (
            <div className="flex items-center gap-3 w-full animate-fade-in px-2">
              <div className="h-8 w-8 shrink-0 rounded-full bg-emerald-800/50 flex items-center justify-center border border-white/10 text-emerald-100 text-xs font-bold uppercase">
                {user.email.substring(0, 2)}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-[13px] font-semibold text-emerald-50 truncate">{user.full_name ?? "Manager"}</p>
                <p className="text-[11px] text-emerald-100/70 truncate">{user.email}</p>
              </div>
            </div>
          )}
          {/* Action Row: Logout */}
          <button
            type="button"
            title={t("nav.logout")}
            onClick={() => void logout()}
            className={clsx(
              "flex items-center gap-2 rounded-lg transition-colors duration-fast",
              "text-emerald-100/60 hover:text-red-400 hover:bg-red-400/10",
              collapsed ? "h-9 w-9 justify-center mx-auto" : "w-full px-2 py-1.5 text-[13px]"
            )}
          >
            <IconLogout className="h-4 w-4 shrink-0" />
            {!collapsed && <span className="font-medium">{t("nav.logout")}</span>}
          </button>
        </div>
      </aside>

      {/* Responsive toggle overlay logic handled natively */}

      {/* ═══ MAIN CONTENT AREA ═══ */}
      <div
        className="flex min-h-screen flex-col transition-all duration-slow ease-apple"
        style={{ marginInlineStart: `${sidebarW}px` }}
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
