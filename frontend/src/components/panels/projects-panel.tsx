"use client";

import { FormEvent, useEffect, useMemo, useState, useRef } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import {
  PerformanceImportResponse,
  PerformanceOpportunity,
  PerformanceSnapshot,
  Project,
  ProjectPerformanceFeedback,
  ProjectReadiness,
  SearchConsoleStatus,
  SeoIntelligenceResponse,
} from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { useToast } from "@/components/ui/toast";
import { InputField } from "@/components/ui/input-field";
import { SelectDropdown } from "@/components/ui/select-dropdown";
import type { SelectOption } from "@/components/ui/select-dropdown";
import { ReadinessTab, PerformanceTab, GeneralTab, WordPressTab, RulebookTab } from "./projects/project-tabs";

/* Projects workspace: master list + contextual configuration and operational state. */

interface ProjectsPanelProps {
  token: string;
  projects: Project[];
  selectedProjectId: string | null;
  canManageProjects: boolean;
  onSelectProject: (projectId: string | null) => void;
  onProjectsRefresh: () => Promise<void>;
}

interface RulebookResponse { content?: string; }

type ProjectTab = "general" | "wordpress" | "rules" | "readiness" | "performance";

import { VERTICAL_OPTIONS, READINESS_COPY, READINESS_ITEM_COPY, PERFORMANCE_COPY, SEARCH_CONSOLE_COPY, SEO_INTELLIGENCE_COPY, SEO_NEXT_ACTION_COPY, SEO_WARNING_COPY, PROJECT_ERROR_COPY, readinessItemKind, localizeReadinessLabel, localizeReadinessText, formatReadinessDate, extractError, localizeProjectError } from "./projects/project-constants";

function useClickOutside(ref: React.RefObject<HTMLElement | null>, handler: () => void) {
  useEffect(() => {
    const listener = (e: MouseEvent | TouchEvent) => {
      if (!ref.current || ref.current.contains(e.target as Node)) return;
      handler();
    };
    document.addEventListener("mousedown", listener);
    document.addEventListener("touchstart", listener);
    return () => {
      document.removeEventListener("mousedown", listener);
      document.removeEventListener("touchstart", listener);
    };
  }, [ref, handler]);
}

/* ── Workspace illustration for empty state ── */
function FolderIllustration() {
  return (
    <div className="mx-auto mb-5 flex h-12 w-12 items-center justify-center text-ink-tertiary">
      <svg viewBox="0 0 48 48" fill="none" className="h-10 w-10" aria-hidden>
        <path d="M5 13a4 4 0 0 1 4-4h10l5 5h15a4 4 0 0 1 4 4v17a4 4 0 0 1-4 4H9a4 4 0 0 1-4-4V13Z" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function DomainHelperText({
  accessibleLabel,
  withoutLabel,
}: {
  accessibleLabel: string;
  withoutLabel: string;
}) {
  return (
    <span dir="ltr" aria-label={accessibleLabel}>
      <bdi dir="ltr">example.com</bdi>
      {" ("}
      <bdi dir="auto">{withoutLabel}</bdi>
      {" "}
      <bdi dir="ltr">https://</bdi>
      {")"}
    </span>
  );
}

export function ProjectsPanel({
  token, projects, selectedProjectId, canManageProjects, onSelectProject, onProjectsRefresh,
}: ProjectsPanelProps) {
  const { t, locale } = useI18n();
  const { showToast } = useToast();

  // Creation state
  const [creating, setCreating] = useState(false);
  const [newProject, setNewProject] = useState({
    name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "",
  });

  // Editor states
  const [activeTab, setActiveTab] = useState<ProjectTab>("general");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [deletingProjectId, setDeletingProjectId] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<ProjectReadiness | null>(null);
  const [readinessLoading, setReadinessLoading] = useState(false);
  const [readinessError, setReadinessError] = useState<string | null>(null);
  const [performance, setPerformance] = useState<ProjectPerformanceFeedback | null>(null);
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const [performanceError, setPerformanceError] = useState<string | null>(null);
  const [seoIntelligence, setSeoIntelligence] = useState<SeoIntelligenceResponse | null>(null);
  const [seoIntelligenceLoading, setSeoIntelligenceLoading] = useState(false);
  const [seoIntelligenceError, setSeoIntelligenceError] = useState<string | null>(null);
  const [performanceImportOpen, setPerformanceImportOpen] = useState(false);
  const [performanceCsv, setPerformanceCsv] = useState("");
  const [performanceImporting, setPerformanceImporting] = useState(false);
  const [dismissingOpportunityId, setDismissingOpportunityId] = useState<string | null>(null);
  const [searchConsole, setSearchConsole] = useState<SearchConsoleStatus | null>(null);
  const [searchConsoleLoading, setSearchConsoleLoading] = useState(false);
  const [searchConsoleAction, setSearchConsoleAction] = useState<string | null>(null);
  const [searchConsoleError, setSearchConsoleError] = useState<string | null>(null);

  // Kebab Menu State
  const [kebabOpen, setKebabOpen] = useState(false);
  const kebabRef = useRef<HTMLDivElement>(null);
  const searchConsoleRefreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useClickOutside(kebabRef, () => setKebabOpen(false));

  useEffect(() => {
    return () => {
      if (searchConsoleRefreshTimerRef.current !== null) {
        globalThis.clearTimeout(searchConsoleRefreshTimerRef.current);
        searchConsoleRefreshTimerRef.current = null;
      }
    };
  }, [selectedProjectId]);

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId]
  );
  const readinessCopy = READINESS_COPY[locale];
  const performanceCopy = PERFORMANCE_COPY[locale];
  const searchConsoleCopy = SEARCH_CONSOLE_COPY[locale];
  const seoIntelligenceCopy = SEO_INTELLIGENCE_COPY[locale];

  // If selected project gets deleted, reset selection
  useEffect(() => {
    if (selectedProjectId === "__new__") return;

    if (selectedProjectId && projects.length > 0 && !projects.find(p => p.id === selectedProjectId)) {
      onSelectProject(projects[0].id);
    } else if (!selectedProjectId && projects.length > 0) {
      onSelectProject(projects[0].id);
    }
  }, [projects, selectedProjectId, onSelectProject]);

  useEffect(() => {
    if (!selectedProject || selectedProjectId === "__new__") {
      setReadiness(null);
      setReadinessError(null);
      setReadinessLoading(false);
      return;
    }

    const controller = new AbortController();
    setReadinessLoading(true);
    setReadinessError(null);

    apiRequest<ProjectReadiness>(`/projects/${selectedProject.id}/readiness`, {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => {
        if (!controller.signal.aborted) setReadiness(payload);
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setReadiness(null);
          setReadinessError(extractError(error));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setReadinessLoading(false);
      });

    return () => controller.abort();
  }, [selectedProject, selectedProjectId, token]);

  useEffect(() => {
    if (!selectedProject || selectedProjectId === "__new__") {
      setPerformance(null);
      setPerformanceError(null);
      setPerformanceLoading(false);
      return;
    }
    if (activeTab !== "performance") return;

    const controller = new AbortController();
    setPerformanceLoading(true);
    setPerformanceError(null);

    apiRequest<ProjectPerformanceFeedback>(`/projects/${selectedProject.id}/performance`, {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => {
        if (!controller.signal.aborted) setPerformance(payload);
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setPerformance(null);
          setPerformanceError(extractError(error));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setPerformanceLoading(false);
      });

    return () => controller.abort();
  }, [activeTab, selectedProject, selectedProjectId, token]);

  useEffect(() => {
    if (!selectedProject || selectedProjectId === "__new__") {
      setSeoIntelligence(null);
      setSeoIntelligenceError(null);
      setSeoIntelligenceLoading(false);
      return;
    }
    if (activeTab !== "performance") return;

    const controller = new AbortController();
    setSeoIntelligenceLoading(true);
    setSeoIntelligenceError(null);
    apiRequest<SeoIntelligenceResponse>(`/projects/${selectedProject.id}/seo-intelligence`, {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => {
        if (!controller.signal.aborted) setSeoIntelligence(payload);
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setSeoIntelligence(null);
          setSeoIntelligenceError(extractError(error));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setSeoIntelligenceLoading(false);
      });
    return () => controller.abort();
  }, [activeTab, selectedProject, selectedProjectId, token]);

  useEffect(() => {
    if (!selectedProject || selectedProjectId === "__new__" || activeTab !== "performance") return;
    const controller = new AbortController();
    setSearchConsoleLoading(true);
    setSearchConsoleError(null);
    apiRequest<SearchConsoleStatus>(`/projects/${selectedProject.id}/search-console/status`, {
      token,
      signal: controller.signal,
      timeoutMs: 10000,
    })
      .then((payload) => { if (!controller.signal.aborted) setSearchConsole(payload); })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setSearchConsole(null);
          setSearchConsoleError(extractError(error));
        }
      })
      .finally(() => { if (!controller.signal.aborted) setSearchConsoleLoading(false); });
    return () => controller.abort();
  }, [activeTab, selectedProject, selectedProjectId, token]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const callbackState = params.get("search_console");
    const callbackProject = params.get("project_id");
    if (!callbackState) return;

    if (callbackProject && callbackProject !== selectedProject?.id) {
      if (projects.some((project) => project.id === callbackProject)) {
        onSelectProject(callbackProject);
        return;
      }
      showToast("error", searchConsoleCopy.failed);
    } else if (!selectedProject) {
      return;
    } else {
      setActiveTab("performance");
      if (callbackState === "connected") showToast("success", searchConsoleCopy.connected);
      if (callbackState === "error") showToast("error", params.get("message") || searchConsoleCopy.failed);
    }

    ["search_console", "project_id", "category", "message"].forEach((key) => params.delete(key));
    const next = `${window.location.pathname}${params.toString() ? `?${params.toString()}` : ""}${window.location.hash}`;
    window.history.replaceState({}, "", next);
  }, [onSelectProject, projects, searchConsoleCopy, selectedProject, showToast]);

  const refreshReadiness = async () => {
    if (!selectedProject) return;
    setReadinessLoading(true);
    setReadinessError(null);
    try {
      const payload = await apiRequest<ProjectReadiness>(`/projects/${selectedProject.id}/readiness`, {
        token,
        timeoutMs: 10000,
      });
      setReadiness(payload);
    } catch (error) {
      setReadiness(null);
      setReadinessError(extractError(error));
    } finally {
      setReadinessLoading(false);
    }
  };

  const refreshPerformance = async () => {
    if (!selectedProject) return;
    setPerformanceLoading(true);
    setPerformanceError(null);
    try {
      const payload = await apiRequest<ProjectPerformanceFeedback>(`/projects/${selectedProject.id}/performance`, {
        token,
        timeoutMs: 10000,
      });
      setPerformance(payload);
    } catch (error) {
      setPerformance(null);
      setPerformanceError(extractError(error));
    } finally {
      setPerformanceLoading(false);
    }
  };

  const refreshSeoIntelligence = async () => {
    if (!selectedProject) return;
    setSeoIntelligenceLoading(true);
    setSeoIntelligenceError(null);
    try {
      const payload = await apiRequest<SeoIntelligenceResponse>(
        `/projects/${selectedProject.id}/seo-intelligence`,
        { token, timeoutMs: 10000 }
      );
      setSeoIntelligence(payload);
    } catch (error) {
      setSeoIntelligence(null);
      setSeoIntelligenceError(extractError(error));
    } finally {
      setSeoIntelligenceLoading(false);
    }
  };

  const refreshSearchConsole = async () => {
    if (!selectedProject) return;
    setSearchConsoleLoading(true);
    setSearchConsoleError(null);
    try {
      const payload = await apiRequest<SearchConsoleStatus>(`/projects/${selectedProject.id}/search-console/status`, { token, timeoutMs: 10000 });
      setSearchConsole(payload);
    } catch (error) {
      setSearchConsoleError(extractError(error));
    } finally {
      setSearchConsoleLoading(false);
    }
  };

  const connectSearchConsole = async () => {
    if (!selectedProject || searchConsoleAction) return;
    setSearchConsoleAction("connect");
    try {
      const payload = await apiRequest<{ authorization_url: string }>(`/projects/${selectedProject.id}/search-console/connect`, { method: "POST", token, timeoutMs: 10000 });
      window.location.assign(payload.authorization_url);
    } catch (error) {
      showToast("error", extractError(error));
      setSearchConsoleAction(null);
    }
  };

  const refreshSearchConsoleProperties = async () => {
    if (!selectedProject || searchConsoleAction) return;
    setSearchConsoleAction("properties");
    try {
      await apiRequest(`/projects/${selectedProject.id}/search-console/properties/refresh`, { method: "POST", token, timeoutMs: 30000 });
      await refreshSearchConsole();
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setSearchConsoleAction(null);
    }
  };

  const selectSearchConsoleProperty = async (siteUrl: string) => {
    if (!selectedProject || searchConsoleAction) return;
    setSearchConsoleAction("property");
    try {
      await apiRequest(`/projects/${selectedProject.id}/search-console/property`, {
        method: "PUT", token, body: { site_url: siteUrl }, timeoutMs: 15000,
      });
      await refreshSearchConsole();
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setSearchConsoleAction(null);
    }
  };

  const syncSearchConsole = async () => {
    if (!selectedProject || searchConsoleAction) return;
    setSearchConsoleAction("sync");
    try {
      await apiRequest(`/projects/${selectedProject.id}/search-console/sync`, { method: "POST", token, body: {}, timeoutMs: 15000 });
      showToast("success", searchConsoleCopy.syncing);
      await refreshSearchConsole();
      if (searchConsoleRefreshTimerRef.current !== null) {
        globalThis.clearTimeout(searchConsoleRefreshTimerRef.current);
      }
      searchConsoleRefreshTimerRef.current = globalThis.setTimeout(() => {
        searchConsoleRefreshTimerRef.current = null;
        void refreshSearchConsole();
        void refreshPerformance();
        void refreshSeoIntelligence();
      }, 2500);
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setSearchConsoleAction(null);
    }
  };

  const disconnectSearchConsole = async () => {
    if (!selectedProject || searchConsoleAction) return;
    setSearchConsoleAction("disconnect");
    try {
      await apiRequest(`/projects/${selectedProject.id}/search-console/disconnect`, { method: "POST", token, timeoutMs: 15000 });
      await refreshSearchConsole();
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setSearchConsoleAction(null);
    }
  };

  const importPerformanceCsv = async () => {
    if (!selectedProject || performanceImporting) return;
    if (!performanceCsv.trim()) {
      showToast("error", performanceCopy.importEmpty);
      return;
    }
    setPerformanceImporting(true);
    try {
      await apiRequest<PerformanceImportResponse, { csv_text: string; source: "manual_csv" }>(
        `/projects/${selectedProject.id}/performance/import-csv`,
        {
          method: "POST",
          token,
          body: { csv_text: performanceCsv, source: "manual_csv" },
          timeoutMs: 15000,
        }
      );
      showToast("success", performanceCopy.importSuccess);
      setPerformanceCsv("");
      setPerformanceImportOpen(false);
      await Promise.all([refreshPerformance(), refreshSeoIntelligence()]);
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setPerformanceImporting(false);
    }
  };

  const dismissOpportunity = async (opportunityId: string) => {
    if (!selectedProject || !canManageProjects || dismissingOpportunityId) return;
    setDismissingOpportunityId(opportunityId);
    try {
      await apiRequest(`/projects/${selectedProject.id}/performance/opportunities/${opportunityId}/dismiss`, {
        method: "POST",
        token,
        timeoutMs: 10000,
      });
      await Promise.all([refreshPerformance(), refreshSeoIntelligence()]);
    } catch (error) {
      showToast("error", extractError(error));
    } finally {
      setDismissingOpportunityId(null);
    }
  };

  const verticalOptions: SelectOption[] = useMemo(() => {
    const base = VERTICAL_OPTIONS.map((v) => ({
      value: v.value,
      label: locale === "fa" ? v.fa : locale === "ar" ? v.ar : v.en,
    }));
    base.push({ value: "__custom__", label: t("projects.customVertical") });
    return base;
  }, [locale, t]);

  const resolvedVertical = newProject.vertical === "__custom__"
    ? newProject.customVertical.trim()
    : VERTICAL_OPTIONS.find((v) => v.value === newProject.vertical)?.en ?? newProject.vertical;

  const onCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setCreating(true);
    try {
      const res = await apiRequest<Project, Record<string, string>>("/projects", {
        method: "POST", token,
        body: { name: newProject.name.trim(), domain: newProject.domain.trim(), vertical: resolvedVertical, description: newProject.description.trim() },
      });
      showToast("success", t("toast.projectCreated"));
      setNewProject({ name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "" });
      await onProjectsRefresh();
      onSelectProject(res.id); // Auto-select new project
    } catch (e) {
      showToast("error", extractError(e));
    } finally { setCreating(false); }
  };

  const onDelete = async (projectId: string) => {
    if (!canManageProjects || deletingProjectId) return;
    setDeletingProjectId(projectId);
    try {
      await apiRequest<void>(`/projects/${projectId}`, { method: "DELETE", token }, { cascade: true });
      showToast("success", t("toast.projectDeleted"));
      setDeleteConfirmId(null);
      if (selectedProjectId === projectId) onSelectProject(null);
      await onProjectsRefresh();
    } catch (e) {
      showToast("error", extractError(e));
    } finally {
      setDeletingProjectId(null);
    }
  };

  const domainPattern = /^[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$/;
  const domainValid = newProject.domain.length > 0 ? domainPattern.test(newProject.domain) : null;

  /* ═══════════════════════════════════════════════════════════════
     STATE A: EMPTY (0 Projects)
     ═══════════════════════════════════════════════════════════════ */
  if (projects.length === 0) {
    return (
      <section className="smx-page flex min-h-[calc(100dvh-110px)] items-center justify-center">
        <div className="w-full max-w-[520px] py-10">
          <div className="mb-8 text-center">
            <FolderIllustration />
            <h2 className="mb-2 text-xl font-semibold text-ink">{t("projects.emptyTitle")}</h2>
            <p className="text-base leading-[22px] text-ink-secondary">{t("projects.emptySubtitle")}</p>
          </div>

          {canManageProjects ? (
          <form className="space-y-6" onSubmit={onCreate}>
            <div className="space-y-4">
              <InputField
                label={t("projects.projectName")}
                required
                helperText={t("projects.projectNameHelper")}
                value={newProject.name}
                onChange={(e) => setNewProject((p) => ({ ...p, name: e.target.value }))}
              />
              <InputField
                label={t("projects.domain")}
                helperText={
                  <DomainHelperText
                    accessibleLabel={t("projects.domainHelper")}
                    withoutLabel={t("projects.domainWithoutProtocol")}
                  />
                }
                successText={domainValid === true ? t("projects.domainValid") : undefined}
                errorText={domainValid === false ? t("projects.domainInvalid") : undefined}
                value={newProject.domain}
                onChange={(e) => setNewProject((p) => ({ ...p, domain: e.target.value }))}
                dir="ltr"
              />
              <SelectDropdown
                label={t("projects.industry")}
                options={verticalOptions}
                value={newProject.vertical}
                onChange={(v) => setNewProject((p) => ({ ...p, vertical: v }))}
              />
              {newProject.vertical === "__custom__" && (
                <InputField
                  label={t("projects.customVertical")}
                  required
                  value={newProject.customVertical}
                  onChange={(e) => setNewProject((p) => ({ ...p, customVertical: e.target.value }))}
                />
              )}
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium text-ink-secondary">{t("projects.description")}</label>
                <textarea
                  aria-label={t("projects.description")}
                  placeholder={t("projects.descriptionPlaceholder")}
                  className="smx-input min-h-[100px] w-full resize-none"
                  value={newProject.description}
                  onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
                />
              </div>
            </div>

            <Button type="submit" variant="primary" loading={creating} fullWidth size="lg">
              {t("projects.createProject")}
            </Button>
          </form>
          ) : (
            <p className="border-s-2 border-warning bg-warning-subtle px-4 py-3 text-center text-sm text-warning">
              {t("toast.accessDenied")}
            </p>
          )}
        </div>
      </section>
    );
  }

  /* ═══════════════════════════════════════════════════════════════
     STATE B: MASTER-DETAIL (1+ Projects)
     ═══════════════════════════════════════════════════════════════ */
  return (
    <section className="smx-page !max-w-none !py-0 grid min-h-full min-w-0 items-start lg:grid-cols-[240px_minmax(0,1fr)]">

      {/* Delete confirmation modal */}
      <Modal
        open={Boolean(deleteConfirmId)}
        onClose={() => {
          if (!deletingProjectId) setDeleteConfirmId(null);
        }}
        title={t("projects.confirmDelete")}
        footer={
          <>
            <Button
              variant="outlined"
              disabled={Boolean(deletingProjectId)}
              onClick={() => setDeleteConfirmId(null)}
            >
              {t("common.cancel")}
            </Button>
            <Button
              variant="danger"
              loading={Boolean(deletingProjectId)}
              onClick={() => deleteConfirmId && void onDelete(deleteConfirmId)}
            >
              {t("common.delete")}
            </Button>
          </>
        }
      >
        <p className="text-base text-ink-secondary leading-relaxed">{t("projects.confirmDeleteMsg")}</p>
      </Modal>

      <Modal
        open={performanceImportOpen}
        onClose={() => {
          if (!performanceImporting) setPerformanceImportOpen(false);
        }}
        title={performanceCopy.importTitle}
        maxWidth="42rem"
        footer={
          <>
            <Button
              variant="outlined"
              disabled={performanceImporting}
              onClick={() => setPerformanceImportOpen(false)}
            >
              {t("common.cancel")}
            </Button>
            <Button
              variant="primary"
              loading={performanceImporting}
              disabled={!performanceCsv.trim()}
              onClick={() => void importPerformanceCsv()}
            >
              {performanceCopy.import}
            </Button>
          </>
        }
      >
        <div className="space-y-3">
          <p className="text-sm leading-5 text-ink-secondary">
            {performanceCopy.importSubtitle}
          </p>
          <code className="block overflow-x-auto rounded-md bg-ink/[0.04] px-3 py-2 text-xs text-ink-secondary" dir="ltr">
            {performanceCopy.importColumns}
          </code>
          <textarea
            aria-label={performanceCopy.importTitle}
            className="min-h-[220px] w-full resize-y rounded-xl border border-line bg-surface px-3 py-3 font-mono text-sm leading-5 text-ink outline-none transition-colors duration-150 placeholder:text-ink-muted focus:border-brand focus:ring-1 focus:ring-brand/20"
            placeholder={performanceCopy.importPlaceholder}
            value={performanceCsv}
            onChange={(event) => setPerformanceCsv(event.target.value)}
            dir="ltr"
            spellCheck={false}
          />
        </div>
      </Modal>

      {/* Project list */}
      <aside className="relative z-10 flex max-h-[280px] min-h-[220px] min-w-0 flex-col overflow-hidden border-e border-line bg-[rgb(var(--bg-secondary)/0.55)] lg:sticky lg:top-0 lg:max-h-[calc(100dvh-96px)]">
        <header className="flex h-[52px] shrink-0 items-center justify-between gap-3 border-b border-line px-4">
          <h2 className="text-base font-semibold text-ink">{t("projects.title")}</h2>
          <div className="flex items-center gap-1.5">
            <button type="button"
              onClick={() => void onProjectsRefresh()}
              className="smx-icon-button !h-8 !w-8"
              aria-label={t("common.refresh")}
              title={t("common.refresh")}
            >
              <svg className="w-[15px] h-[15px]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
            {canManageProjects && (
              <>
                <div className="w-[1px] h-4 bg-ink/[0.06] mx-0.5" />
                <button type="button"
                  onClick={() => {
                    setNewProject({ name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "" });
                    onSelectProject("__new__");
                  }}
                  className="flex h-8 w-8 items-center justify-center rounded-md bg-brand text-white transition-colors hover:bg-brand-hover"
                  aria-label={t("projects.createNew")}
                  title={t("projects.createNew")}
                >
                  <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                </button>
              </>
            )}
          </div>
        </header>

        {/* The Seamless List */}
        <div className="flex-1 overflow-y-auto py-2">
          {projects.map((project) => (
            <button type="button"
              key={project.id}
              onClick={() => onSelectProject(project.id)}
              className={clsx(
                "group relative w-full px-4 py-3 text-start transition-colors duration-fast",
                selectedProjectId === project.id
                  ? "bg-ink/[0.055]"
                  : "bg-transparent hover:bg-ink/[0.035]"
              )}
            >
              <div className="flex items-center justify-between gap-2 mb-0.5">
                <span className={clsx("truncate text-base", selectedProjectId === project.id ? "font-semibold text-ink" : "font-medium text-ink-secondary group-hover:text-ink")}>
                  {project.name}
                </span>
                {project.wordpress_url && (
                  <span className={clsx("shrink-0 text-xs font-medium", selectedProjectId === project.id ? "text-success" : "text-ink-tertiary")}>WP</span>
                )}
              </div>
              <span className={clsx("truncate block text-xs", selectedProjectId === project.id ? "text-ink-secondary" : "text-ink-tertiary")} dir="ltr">
                {project.domain || t("projects.noDomain")}
              </span>
            </button>
          ))}
        </div>
      </aside>

      {/* ── RIGHT COLUMN (DETAIL) ── */}
      <main className="min-w-0 overflow-hidden">

        {selectedProjectId === "__new__" ? (
          // Create Mode
          <div className="p-6 lg:p-8">
            <div className="max-w-xl">
              <h3 className="mb-6 text-xl font-semibold text-ink">{t("projects.createNew")}</h3>
              <form className="space-y-6" onSubmit={onCreate}>
                <div className="space-y-4">
                  <InputField
                    label={t("projects.projectName")}
                    required
                    helperText={t("projects.projectNameHelper")}
                    value={newProject.name}
                    onChange={(e) => setNewProject((p) => ({ ...p, name: e.target.value }))}
                  />
                  <InputField
                    label={t("projects.domain")}
                    helperText={
                  <DomainHelperText
                    accessibleLabel={t("projects.domainHelper")}
                    withoutLabel={t("projects.domainWithoutProtocol")}
                  />
                }
                    successText={domainValid === true ? t("projects.domainValid") : undefined}
                    errorText={domainValid === false ? t("projects.domainInvalid") : undefined}
                    value={newProject.domain}
                    onChange={(e) => setNewProject((p) => ({ ...p, domain: e.target.value }))}
                    dir="ltr"
                  />
                  <SelectDropdown
                    label={t("projects.industry")}
                    options={verticalOptions}
                    value={newProject.vertical}
                    onChange={(v) => setNewProject((p) => ({ ...p, vertical: v }))}
                  />
                  {newProject.vertical === "__custom__" && (
                    <InputField
                      label={t("projects.customVertical")}
                      required
                      value={newProject.customVertical}
                      onChange={(e) => setNewProject((p) => ({ ...p, customVertical: e.target.value }))}
                    />
                  )}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-sm font-medium text-ink-secondary">{t("projects.description")}</label>
                    <textarea
                      aria-label={t("projects.description")}
                      placeholder={t("projects.descriptionPlaceholder")}
                      className="smx-input min-h-[100px] w-full resize-none"
                      value={newProject.description}
                      onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
                    />
                  </div>
                </div>

                <div className="flex flex-row gap-2 border-block-start border-line pt-5">
                  <Button type="button" variant="outlined" onClick={() => onSelectProject(projects[0]?.id || null)} size="lg">
                    {t("common.cancel")}
                  </Button>
                  <Button type="submit" variant="primary" loading={creating} size="lg" className="min-w-[140px]">
                    {t("projects.createProject")}
                  </Button>
                </div>
              </form>
            </div>
          </div>
        ) : selectedProject ? (
          // View/Edit Mode
          <>
            <header className="flex shrink-0 flex-col border-block-end border-line">
              <div className="flex min-w-0 items-start justify-between gap-4 px-6 pb-4 pt-7 lg:px-8">
                <div className="min-w-0 flex-1">
                  <h2 className="mb-1.5 truncate text-xl font-semibold leading-6 text-ink">{selectedProject.name}</h2>
                  <p className="truncate text-sm text-ink-tertiary" dir="ltr">{selectedProject.domain || ""}</p>
                </div>

                {/* Project actions */}
                {canManageProjects && <div className="relative shrink-0" ref={kebabRef}>
                  <button type="button"
                    onClick={() => setKebabOpen(!kebabOpen)}
                    className={clsx(
                      "flex items-center justify-center w-8 h-8 rounded-md transition-all duration-200",
                      kebabOpen ? "bg-ink/[0.055] text-ink" : "text-ink-muted hover:bg-ink/[0.045] hover:text-ink"
                    )}
                    aria-label={t("common.moreOptions")}
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" /></svg>
                  </button>
                  {kebabOpen && (
                    <div className="absolute top-full inset-inline-end-0 z-50 mt-1 w-48 origin-top-right animate-fade-in rounded-xl border border-line bg-surface py-1">
                      <button type="button"
                        onClick={() => { setKebabOpen(false); setDeleteConfirmId(selectedProject.id); }}
                        className="w-full text-start px-4 py-2 text-sm font-medium text-danger hover:bg-danger-subtle flex items-center gap-2 transition-colors duration-fast"
                      >
                        <svg className="w-[14px] h-[14px]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                        {t("common.delete")}
                      </button>
                    </div>
                  )}
                </div>}
              </div>

              <div className="overflow-x-auto px-6 lg:px-8">
                <div className="flex min-w-max gap-5">
                  {[
                    { id: "readiness", label: readinessCopy.tab },
                    { id: "general", label: t("projects.tabGeneral") },
                    { id: "wordpress", label: t("projects.tabWordpress") },
                    { id: "performance", label: performanceCopy.tab },
                    { id: "rules", label: t("projects.tabRules") },
                  ].map((tab) => (
                    <button type="button"
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id as ProjectTab)}
                      className={clsx(
                        "min-h-10 border-b-2 px-0.5 py-2 text-center text-sm font-medium leading-5 transition-colors duration-fast",
                        activeTab === tab.id
                          ? "border-brand text-ink"
                          : "border-transparent text-ink-tertiary hover:text-ink"
                      )}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              </div>
            </header>

            <div className="relative min-w-0 p-6 lg:p-8">
              {activeTab === "readiness" && (
                <ReadinessTab
                  copy={readinessCopy}
                  locale={locale}
                  readiness={readiness}
                  loading={readinessLoading}
                  error={readinessError}
                  onRefresh={() => void refreshReadiness()}
                  onOpenRulebook={() => setActiveTab("rules")}
                  onOpenWordPress={() => setActiveTab("wordpress")}
                />
              )}
              {activeTab === "general" && (
                <GeneralTab
                  token={token}
                  project={selectedProject}
                  canManageProjects={canManageProjects}
                  verticalOptions={verticalOptions}
                  onProjectsRefresh={onProjectsRefresh}
                />
              )}
              {activeTab === "wordpress" && (
                <WordPressTab
                  token={token}
                  project={selectedProject}
                  canManageProjects={canManageProjects}
                  onProjectsRefresh={onProjectsRefresh}
                />
              )}
              {activeTab === "performance" && (
                <PerformanceTab
                  copy={performanceCopy}
                  locale={locale}
                  canManageProjects={canManageProjects}
                  feedback={performance}
                  loading={performanceLoading}
                  error={performanceError}
                  dismissingOpportunityId={dismissingOpportunityId}
                  onRefresh={() => {
                    void refreshPerformance();
                    void refreshSeoIntelligence();
                  }}
                  seoIntelligence={seoIntelligence}
                  seoIntelligenceCopy={seoIntelligenceCopy}
                  seoIntelligenceLoading={seoIntelligenceLoading}
                  seoIntelligenceError={seoIntelligenceError}
                  onOpenImport={() => setPerformanceImportOpen(true)}
                  onDismiss={(opportunityId) => void dismissOpportunity(opportunityId)}
                  searchConsole={searchConsole}
                  searchConsoleCopy={searchConsoleCopy}
                  searchConsoleLoading={searchConsoleLoading}
                  searchConsoleAction={searchConsoleAction}
                  searchConsoleError={searchConsoleError}
                  onConnectSearchConsole={() => void connectSearchConsole()}
                  onRefreshSearchConsole={() => void refreshSearchConsole()}
                  onRefreshSearchConsoleProperties={() => void refreshSearchConsoleProperties()}
                  onSelectSearchConsoleProperty={(siteUrl) => void selectSearchConsoleProperty(siteUrl)}
                  onSyncSearchConsole={() => void syncSearchConsole()}
                  onDisconnectSearchConsole={() => void disconnectSearchConsole()}
                />
              )}
              {activeTab === "rules" && (
                <RulebookTab
                  token={token}
                  project={selectedProject}
                  canManageProjects={canManageProjects}
                />
              )}
            </div>
          </>
        ) : null}
      </main>

    </section>
  );
}
