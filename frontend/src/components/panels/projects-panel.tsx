"use client";

import { FormEvent, useEffect, useMemo, useState, useRef } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import { Project } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { useToast } from "@/components/ui/toast";
import { InputField } from "@/components/ui/input-field";
import { SelectDropdown } from "@/components/ui/select-dropdown";
import type { SelectOption } from "@/components/ui/select-dropdown";

/* ═══════════════════════════════════════════════════════════════
   Spec: Screen 3 — Projects Page (Apple/Linear SaaS UI Tier)
   Architecture: 100vh Master-Detail split layout (NO SCROLL)
   - Left (30%): macOS-style Sidebar Project List
   - Right (70%): Tabbed Editor (Zero layout shift, Spatial Forms)
   ═══════════════════════════════════════════════════════════════ */

interface ProjectsPanelProps {
  token: string;
  projects: Project[];
  selectedProjectId: string | null;
  onSelectProject: (projectId: string | null) => void;
  onProjectsRefresh: () => Promise<void>;
}

interface RulebookResponse { content?: string; }

const VERTICAL_OPTIONS = [
  { value: "tech", fa: "فناوری و نرم‌افزار", ar: "التكنولوجيا والبرمجيات", en: "Technology and Software" },
  { value: "health", fa: "سلامت و پزشکی", ar: "الصحة والطب", en: "Health and Medical" },
  { value: "ecommerce", fa: "فروشگاه و تجارت", ar: "المتاجر والتجارة", en: "E-Commerce" },
  { value: "education", fa: "آموزش و یادگیری", ar: "التعليم والتعلم", en: "Education and Learning" },
  { value: "finance", fa: "مالی و اقتصادی", ar: "المالية والاقتصاد", en: "Finance and Economy" },
  { value: "marketing", fa: "بازاریابی دیجیتال", ar: "التسويق الرقمي", en: "Digital Marketing" },
];

function extractError(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  return "Unexpected error";
}

/* ── Hooks ── */
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
    <div className="mx-auto mb-6 flex h-24 w-24 items-center justify-center rounded-[2rem] bg-gradient-to-br from-teal-50 to-teal-100/60 shadow-sm border border-teal-100/50">
      <svg viewBox="0 0 48 48" fill="none" className="h-10 w-10 text-teal-600">
        <path d="M4 12C4 9.79086 5.79086 8 8 8H18.8284C19.8893 8 20.9067 8.42143 21.6569 9.17157L24 11.5147M24 11.5147L26.3431 13.8579C27.0933 14.608 28.1107 15.0294 29.1716 15.0294H40C42.2091 15.0294 44 16.8203 44 19.0294V36C44 38.2091 42.2091 40 40 40H8C5.79086 40 4 38.2091 4 36V12Z" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M14 26H28" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
        <path d="M14 32H22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
      </svg>
    </div>
  );
}

export function ProjectsPanel({
  token, projects, selectedProjectId, onSelectProject, onProjectsRefresh,
}: ProjectsPanelProps) {
  const { t, locale } = useI18n();
  const { showToast } = useToast();

  // Creation state
  const [creating, setCreating] = useState(false);
  const [newProject, setNewProject] = useState({
    name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "",
  });

  // Editor states
  const [activeTab, setActiveTab] = useState<"general" | "wordpress" | "rules">("general");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Kebab Menu State
  const [kebabOpen, setKebabOpen] = useState(false);
  const kebabRef = useRef<HTMLDivElement>(null);
  useClickOutside(kebabRef, () => setKebabOpen(false));

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId]
  );

  // If selected project gets deleted, reset selection
  useEffect(() => {
    if (selectedProjectId && projects.length > 0 && !projects.find(p => p.id === selectedProjectId)) {
      onSelectProject(projects[0].id);
    } else if (!selectedProjectId && projects.length > 0) {
      onSelectProject(projects[0].id);
    }
  }, [projects, selectedProjectId, onSelectProject]);

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
    setDeleteConfirmId(null);
    try {
      await apiRequest<void>(`/projects/${projectId}`, { method: "DELETE", token }, { cascade: true });
      showToast("success", t("toast.projectDeleted"));
      if (selectedProjectId === projectId) onSelectProject(null);
      await onProjectsRefresh();
    } catch (e) { showToast("error", extractError(e)); }
  };

  const domainPattern = /^[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$/;
  const domainValid = newProject.domain.length > 0 ? domainPattern.test(newProject.domain) : null;

  /* ═══════════════════════════════════════════════════════════════
     STATE A: EMPTY (0 Projects)
     ═══════════════════════════════════════════════════════════════ */
  if (projects.length === 0) {
    return (
      <section className="animate-fade-in flex min-h-[calc(100vh-80px)] items-center justify-center p-4">
        <div className="w-full max-w-lg rounded-3xl border border-slate-200/60 bg-white p-10 shadow-sm relative overflow-hidden">
          {/* Subtle gradient orb for Apple feel */}
          <div className="absolute top-0 inset-inline-start-1/2 -ml-32 w-64 h-64 bg-teal-500/10 rounded-full blur-3xl pointer-events-none -translate-y-1/2" />

          <div className="text-center mb-8 relative z-10">
            <FolderIllustration />
            <h2 className="text-[24px] font-bold text-slate-900 tracking-tight mb-2">{t("projects.emptyTitle")}</h2>
            <p className="text-[14px] text-slate-500">{t("projects.emptySubtitle")}</p>
          </div>

          <form className="space-y-6 relative z-10" onSubmit={onCreate}>
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
                  <span className="flex gap-1">
                    example.com ({t("common.without")} <span dir="ltr">https://</span>)
                  </span>
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
              <div className="flex flex-col gap-[6px]">
                <label className="text-[13px] font-semibold text-slate-700">{t("projects.description")}</label>
                <textarea
                  placeholder={t("projects.descriptionPlaceholder")}
                  className="w-full rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-[14px] text-slate-900 placeholder:text-slate-400 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 outline-none transition-all duration-200 resize-none min-h-[100px]"
                  value={newProject.description}
                  onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
                />
              </div>
            </div>

            <Button type="submit" variant="primary" loading={creating} fullWidth size="lg">
              {t("projects.createProject")}
            </Button>
          </form>
        </div>
      </section>
    );
  }

  /* ═══════════════════════════════════════════════════════════════
     STATE B: MASTER-DETAIL (1+ Projects)
     ═══════════════════════════════════════════════════════════════ */
  return (
    <section className="animate-fade-in flex h-[calc(100vh-80px)] overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">

      {/* Delete confirmation modal */}
      <Modal
        open={Boolean(deleteConfirmId)}
        onClose={() => setDeleteConfirmId(null)}
        title={t("projects.confirmDelete")}
        footer={
          <>
            <Button variant="outlined" onClick={() => setDeleteConfirmId(null)}>{t("common.cancel")}</Button>
            <Button variant="danger" onClick={() => deleteConfirmId && void onDelete(deleteConfirmId)}>{t("common.delete")}</Button>
          </>
        }
      >
        <p className="text-[14px] text-slate-600 leading-relaxed">{t("projects.confirmDeleteMsg")}</p>
      </Modal>

      {/* ── LEFT COLUMN (MASTER: macOS style sidebar list) ── */}
      <aside className="w-[30%] min-w-[300px] border-inline-end border-slate-200 flex flex-col bg-slate-50/50 relative z-10">
        <header className="flex h-16 shrink-0 items-center justify-between border-block-end border-slate-200 px-6 bg-slate-50/50 backdrop-blur-md">
          <h2 className="text-[16px] font-bold text-slate-900 tracking-tight">{t("projects.title")}</h2>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => void onProjectsRefresh()}
              className="h-8 w-8 flex items-center justify-center rounded-md text-slate-400 hover:bg-slate-200/60 hover:text-slate-700 transition-all duration-200"
              title={t("common.refresh")}
            >
              <svg className="w-[15px] h-[15px]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
            <div className="w-[1px] h-4 bg-slate-200 mx-0.5" />
            <button
              onClick={() => {
                setNewProject({ name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "" });
                onSelectProject("__new__");
              }}
              className="h-8 w-8 flex items-center justify-center rounded-md text-teal-600 bg-teal-500/10 hover:bg-teal-500/20 transition-all duration-200"
              title={t("projects.createNew")}
            >
              <svg className="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>
          </div>
        </header>

        {/* The Seamless List */}
        <div className="flex-1 overflow-y-auto py-3">
          {projects.map((project) => (
            <button
              key={project.id}
              onClick={() => onSelectProject(project.id)}
              className={clsx(
                "w-full text-start px-6 py-3 transition-all duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] relative group focus:outline-none",
                selectedProjectId === project.id
                  ? "bg-teal-50/50 shadow-[inset_3px_0_0_0_var(--tw-shadow-color)] shadow-teal-600"
                  : "bg-transparent hover:bg-slate-900/5 focus:bg-slate-900/5" // Native feel hover
              )}
            >
              <div className="flex items-center justify-between gap-2 mb-0.5">
                <span className={clsx("truncate text-[14px]", selectedProjectId === project.id ? "font-bold text-teal-900" : "font-medium text-slate-900 group-hover:text-black")}>
                  {project.name}
                </span>
                {project.wordpress_url && (
                  <span className={clsx("shrink-0 rounded-[4px] px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider", selectedProjectId === project.id ? "bg-teal-100 text-teal-700" : "bg-slate-100 text-slate-500")}>WP</span>
                )}
              </div>
              <span className={clsx("truncate block text-[12px]", selectedProjectId === project.id ? "text-teal-700/80 font-medium" : "text-slate-500")} dir="ltr">
                {project.domain || t("projects.noDomain")}
              </span>
            </button>
          ))}
        </div>
      </aside>

      {/* ── RIGHT COLUMN (DETAIL) ── */}
      <main className="flex-1 flex flex-col bg-white overflow-hidden min-w-0">

        {selectedProjectId === "__new__" ? (
          // Create Mode
          <div className="flex-1 overflow-y-auto p-8 md:p-12">
            <div className="max-w-xl">
              <h3 className="text-[20px] font-bold text-slate-900 tracking-tight mb-8">{t("projects.createNew")}</h3>
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
                      <span className="flex gap-1">
                        example.com ({t("common.without")} <span dir="ltr">https://</span>)
                      </span>
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
                  <div className="flex flex-col gap-[6px]">
                    <label className="text-[13px] font-semibold text-slate-700">{t("projects.description")}</label>
                    <textarea
                      placeholder={t("projects.descriptionPlaceholder")}
                      className="w-full rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-[14px] text-slate-900 placeholder:text-slate-400 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 outline-none transition-all duration-200 resize-none min-h-[100px]"
                      value={newProject.description}
                      onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
                    />
                  </div>
                </div>

                <div className="flex flex-row gap-3 pt-6 border-block-start border-slate-100">
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
            <header className="flex flex-col border-block-end border-slate-200 shrink-0">
              <div className="px-8 pt-8 pb-4 flex items-start justify-between">
                <div>
                  <h2 className="text-[24px] font-bold text-slate-900 tracking-tight leading-none mb-1.5">{selectedProject.name}</h2>
                  <p className="text-[13px] font-medium text-slate-500" dir="ltr">{selectedProject.domain || ""}</p>
                </div>

                {/* ── Polished Kebab Kenu ── */}
                <div className="relative" ref={kebabRef}>
                  <button
                    onClick={() => setKebabOpen(!kebabOpen)}
                    className={clsx(
                      "flex items-center justify-center w-8 h-8 rounded-md transition-all duration-200",
                      kebabOpen ? "bg-slate-100 text-slate-900" : "text-slate-400 hover:text-slate-700 hover:bg-slate-50"
                    )}
                    aria-label="More options"
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" /></svg>
                  </button>
                  {kebabOpen && (
                    <div className="absolute top-full inset-inline-end-0 mt-1 w-48 bg-white border border-slate-200 shadow-lg rounded-xl py-1 z-50 animate-fade-in origin-top-right">
                      <button
                        onClick={() => { setKebabOpen(false); setDeleteConfirmId(selectedProject.id); }}
                        className="w-full text-start px-4 py-2 text-[13px] font-medium text-red-600 hover:bg-red-50 flex items-center gap-2 transition-colors duration-fast"
                      >
                        <svg className="w-[14px] h-[14px]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                        {t("common.delete")} Project
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* TABS (Zero layout shift: inactive has transparent border) */}
              <div className="flex gap-8 px-8 overflow-x-auto no-scrollbar pt-2">
                {[
                  { id: "general", label: t("projects.tabGeneral") },
                  { id: "wordpress", label: t("projects.tabWordpress") },
                  { id: "rules", label: t("projects.tabRules") },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={clsx(
                      "pb-3 text-[13px] font-bold uppercase tracking-wider whitespace-nowrap transition-all duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] border-block-end-[2px]",
                      activeTab === tab.id
                        ? "border-teal-600 text-teal-600"
                        : "border-transparent text-slate-500 hover:text-slate-800"
                    )}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </header>

            <div className="flex-1 overflow-y-auto p-8 relative">
              {activeTab === "general" && (
                <GeneralTab token={token} project={selectedProject} onProjectsRefresh={onProjectsRefresh} />
              )}
              {activeTab === "wordpress" && (
                <WordPressTab token={token} project={selectedProject} onProjectsRefresh={onProjectsRefresh} />
              )}
              {activeTab === "rules" && (
                <RulebookTab token={token} project={selectedProject} />
              )}
            </div>
          </>
        ) : null}
      </main>

    </section>
  );
}

/* ═══════════════════════════════════════════════════════════════
   TAB COMPONENTS — Form Containment and Spatial Grouping
   ═══════════════════════════════════════════════════════════════ */

function GeneralTab({ token, project, onProjectsRefresh }: { token: string; project: Project; onProjectsRefresh: () => Promise<void> }) {
  const { t } = useI18n();
  const [draft, setDraft] = useState({ name: project.name, domain: project.domain ?? "", description: project.description ?? "" });
  const [saving, setSaving] = useState(false);
  const { showToast } = useToast();

  useEffect(() => {
    setDraft({ name: project.name, domain: project.domain ?? "", description: project.description ?? "" });
  }, [project]);

  const onSave = async () => {
    setSaving(true);
    try {
      await apiRequest(`/projects/${project.id}`, {
        method: "PUT", token,
        body: { name: draft.name.trim(), domain: draft.domain.trim(), description: draft.description.trim() },
      });
      showToast("success", t("common.success"));
      await onProjectsRefresh();
    } catch (e) { showToast("error", extractError(e)); }
    finally { setSaving(false); }
  };

  return (
    <div className="max-w-xl space-y-6 animate-fade-in">
      <div className="space-y-4">
        <InputField
          label={t("projects.projectName")}
          value={draft.name}
          onChange={(e) => setDraft((p) => ({ ...p, name: e.target.value }))}
        />
        <InputField
          label={t("projects.domain")}
          value={draft.domain}
          onChange={(e) => setDraft((p) => ({ ...p, domain: e.target.value }))}
          dir="ltr"
        />
        <div className="flex flex-col gap-[6px]">
          <label className="text-[13px] font-semibold text-slate-700">{t("projects.description")}</label>
          <textarea
            className="w-full rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-[14px] text-slate-900 outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition-all duration-200 resize-none min-h-[120px]"
            value={draft.description}
            onChange={(e) => setDraft((p) => ({ ...p, description: e.target.value }))}
          />
        </div>
      </div>

      <div className="flex justify-end pt-2">
        <Button variant="primary" loading={saving} onClick={() => void onSave()} className="min-w-[120px]">{t("common.save")}</Button>
      </div>
    </div>
  );
}

function WordPressTab({ token, project, onProjectsRefresh }: { token: string; project: Project; onProjectsRefresh: () => Promise<void> }) {
  const { t } = useI18n();
  const { showToast } = useToast();
  const [wpUrl, setWpUrl] = useState("");
  const [wpUsername, setWpUsername] = useState("");
  const [wpPassword, setWpPassword] = useState("");
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setWpUrl(project.wordpress_url ?? "");
    setWpUsername(project.wordpress_username ?? "");
    setWpPassword("");
  }, [project]);

  const save = async () => {
    setSaving(true);
    try {
      const payload: Record<string, string> = {
        wordpress_url: wpUrl.trim(), wordpress_username: wpUsername.trim(),
      };
      if (wpPassword.trim()) payload.wordpress_app_password = wpPassword.trim();
      await apiRequest(`/projects/${project.id}`, { method: "PUT", token, body: payload });
      showToast("success", t("toast.wpSaved"));
      await onProjectsRefresh();
    } catch (e) { showToast("error", extractError(e)); }
    finally { setSaving(false); }
  };

  const testConnection = async () => {
    setTesting(true);
    try {
      const payload = await apiRequest<{ connected?: boolean; actionable_message?: string }>(
        `/projects/${project.id}/wordpress/test-connection`, { method: "POST", token }
      );
      if (payload.connected) showToast("success", t("toast.wpTestSuccess"));
      else showToast("error", payload.actionable_message ?? t("toast.wpTestFailed"));
    } catch (e) { showToast("error", extractError(e)); }
    finally { setTesting(false); }
  };

  return (
    <div className="max-w-2xl animate-fade-in">
      <div className="mb-6">
        <h3 className="text-[15px] font-bold text-slate-900 mb-1">WordPress Integration</h3>
        <p className="text-[13px] font-medium text-slate-500 leading-relaxed">{t("projects.wpSubtitle")}</p>
      </div>

      {/* Form Spatial Containment */}
      <div className="bg-white border border-slate-200 shadow-sm rounded-2xl p-6 md:p-8 space-y-6">
        <InputField
          label={t("projects.wpUrl")}
          helperText={t("projects.wpUrlHelper")}
          value={wpUrl}
          onChange={(e) => setWpUrl(e.target.value)}
          dir="ltr"
        />
        <div className="grid md:grid-cols-2 gap-6">
          <InputField
            label={t("projects.wpUsername")}
            value={wpUsername}
            onChange={(e) => setWpUsername(e.target.value)}
            dir="ltr"
          />
          <InputField
            label={t("projects.wpPassword")}
            type="password"
            helperText={t("projects.wpPasswordTooltip")}
            value={wpPassword}
            onChange={(e) => setWpPassword(e.target.value)}
            dir="ltr"
          />
        </div>

        <div className="flex justify-end gap-3 pt-6 border-block-start border-slate-100">
          <Button variant="outlined" loading={testing} onClick={() => void testConnection()}>{t("projects.wpTestConnection")}</Button>
          <Button variant="primary" loading={saving} onClick={() => void save()} className="min-w-[120px]">{t("projects.wpSave")}</Button>
        </div>
      </div>
    </div>
  );
}

function RulebookTab({ token, project }: { token: string; project: Project }) {
  const { t } = useI18n();
  const { showToast } = useToast();
  const [rulebook, setRulebook] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    apiRequest<RulebookResponse>(`/projects/${project.id}/rulebook`, { token })
      .then(res => { if (mounted) setRulebook(res.content ?? ""); })
      .catch(() => { if (mounted) setRulebook(""); })
      .finally(() => { if (mounted) setLoading(false); });
    return () => { mounted = false; };
  }, [project.id, token]);

  const save = async () => {
    setSaving(true);
    try {
      await apiRequest(`/projects/${project.id}/rulebook`, { method: "POST", token, body: { content: rulebook } });
      showToast("success", t("toast.rulebookSaved"));
    } catch (e) { showToast("error", extractError(e)); }
    finally { setSaving(false); }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl animate-fade-in relative pb-16">

      <div className="mb-4 flex items-center justify-between">
        <p className="text-[13px] font-medium text-slate-500">{t("projects.rulebookEmpty")}</p>
      </div>

      {/* AI-Native Smart Container */}
      <div className="relative flex-1 group rounded-2xl border border-slate-200 bg-slate-50/50 transition-all duration-300 focus-within:ring-2 focus-within:ring-teal-500/20 focus-within:border-teal-500 focus-within:bg-white shadow-sm overflow-hidden flex flex-col min-h-[400px]">

        <textarea
          disabled={loading}
          className="w-full h-full flex-1 bg-transparent p-6 text-[14px] text-slate-900 leading-relaxed outline-none border-none resize-y disabled:opacity-50"
          value={rulebook}
          onChange={(e) => setRulebook(e.target.value)}
          placeholder="- Use formal tone&#10;- Avoid competitor names..."
        />

        {/* AI Sparkle Action Button strictly inside the container */}
        <button
          className="absolute bottom-4 inset-inline-end-4 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/60 backdrop-blur-md border border-slate-200/50 shadow-sm hover:shadow-md hover:bg-white transition-all duration-200 group/ai disabled:opacity-0"
          title="AI Sparkle (Coming soon)"
          disabled={loading}
        >
          <svg className="w-4 h-4 text-teal-500 group-hover/ai:text-teal-600 transition-colors" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" /></svg>
          <span className="text-[11px] font-bold bg-gradient-to-r from-teal-600 to-emerald-500 bg-clip-text text-transparent">AI Assist</span>
        </button>
      </div>

      {/* Primary Action Button (Spatially separated from the textarea) */}
      <div className="absolute bottom-0 inset-inline-end-0 flex justify-end">
        <Button variant="primary" loading={saving || loading} disabled={loading} onClick={() => void save()} className="min-w-[140px] shadow-sm">
          {t("common.save")} K-Rules
        </Button>
      </div>

    </div>
  );
}
