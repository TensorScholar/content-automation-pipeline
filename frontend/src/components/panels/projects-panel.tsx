"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import { Project } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { useToast } from "@/components/ui/toast";
import { InputField } from "@/components/ui/input-field";
import { SelectDropdown } from "@/components/ui/select-dropdown";
import { EmptyState, EmptyIllustration } from "@/components/ui/empty-state";
import { StatusBadge } from "@/components/ui/status-badge";
import type { SelectOption } from "@/components/ui/select-dropdown";

/* ═══════════════════════════════════════════════════════════════
   Spec: Screen 3 — Projects Page
   Architecture: 100vh Master-Detail split layout (NO SCROLL)
   - Left (30%): Project List
   - Right (70%): Tabbed Editor (General, WP, Rulebook)
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

/* ── Workspace illustration for empty state ── */
function FolderIllustration() {
  return (
    <div className="mx-auto mb-6 flex h-24 w-24 items-center justify-center rounded-3xl bg-gradient-to-br from-teal-50 to-teal-100/60 shadow-sm">
      <svg viewBox="0 0 48 48" fill="none" className="h-12 w-12 text-teal-600">
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
     Hide sidebar & tabs. Show centered creation card.
     ═══════════════════════════════════════════════════════════════ */
  if (projects.length === 0) {
    return (
      <section className="animate-fade-in flex min-h-[calc(100vh-80px)] items-center justify-center p-4">
        <div className="w-full max-w-lg rounded-2xl border border-gray-100 bg-white p-8 shadow-sm">
          <div className="text-center mb-8">
            <FolderIllustration />
            <h2 className="text-[24px] font-bold text-gray-900 mb-2">{t("projects.emptyTitle")}</h2>
            <p className="text-[14px] text-gray-500">{t("projects.emptySubtitle")}</p>
          </div>

          <form className="space-y-4" onSubmit={onCreate}>
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
                // Bidi fix: Keep domain LTR so punctuation doesn't jump
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
              <label className="text-[13px] font-medium text-gray-700">{t("projects.description")}</label>
              <textarea
                placeholder={t("projects.descriptionPlaceholder")}
                className="w-full rounded-xl border border-gray-200 bg-white px-4 py-2.5 text-[14px] text-gray-900 placeholder:text-gray-400 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 outline-none transition-all duration-200 resize-none h-20"
                value={newProject.description}
                onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
              />
            </div>
            <Button type="submit" variant="primary" loading={creating} fullWidth size="lg" className="mt-6">
              {t("projects.createProject")}
            </Button>
          </form>
        </div>
      </section>
    );
  }

  /* ═══════════════════════════════════════════════════════════════
     STATE B: MASTER-DETAIL (1+ Projects)
     100vh layout, split screen, tabbed navigation.
     ═══════════════════════════════════════════════════════════════ */
  return (
    <section className="animate-fade-in flex h-[calc(100vh-80px)] overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm">

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
        <p className="text-[14px] text-gray-600">{t("projects.confirmDeleteMsg")}</p>
      </Modal>

      {/* ── LEFT COLUMN (MASTER): PROJECTS LIST ── */}
      <aside className="w-[30%] min-w-[300px] border-e border-gray-200 flex flex-col bg-gray-50/50">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-gray-200 px-4 bg-white">
          <h2 className="text-[16px] font-bold text-gray-900">{t("projects.title")}</h2>
          <div className="flex items-center gap-1">
            <button
              onClick={() => {
                setNewProject({ name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "" });
                onSelectProject("__new__");
              }}
              className="h-8 w-8 flex items-center justify-center rounded-lg text-teal-600 hover:bg-teal-50 transition-colors"
              title={t("projects.createNew")}
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>
            <button
              onClick={() => void onProjectsRefresh()}
              className="h-8 w-8 flex items-center justify-center rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-700 transition-colors"
              title={t("common.refresh")}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {projects.map((project) => (
            <button
              key={project.id}
              onClick={() => onSelectProject(project.id)}
              className={clsx(
                "w-full text-start p-3 rounded-xl transition-all duration-200 border",
                selectedProjectId === project.id
                  ? "bg-white border-teal-500 shadow-sm ring-1 ring-teal-500"
                  : "bg-white border-gray-100 hover:border-gray-300 hover:shadow-sm"
              )}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="truncate text-[14px] font-bold text-gray-900">{project.name}</span>
                {project.wordpress_url && (
                  <span className="shrink-0 rounded-md bg-emerald-50 px-1.5 py-0.5 text-[10px] font-bold text-emerald-600 uppercase">WP</span>
                )}
              </div>
              <span className="truncate block text-[12px] text-gray-500" dir="ltr">
                {project.domain || t("projects.noDomain")}
              </span>
            </button>
          ))}
        </div>
      </aside>

      {/* ── RIGHT COLUMN (DETAIL): ACTIVE PROJECT TABBED VIEW ── */}
      <main className="flex-1 flex flex-col bg-white overflow-hidden min-w-0">

        {selectedProjectId === "__new__" ? (
          // Create Mode
          <div className="flex-1 overflow-y-auto p-8">
            <div className="max-w-xl mx-auto">
              <h3 className="text-[20px] font-bold text-gray-900 mb-6">{t("projects.createNew")}</h3>
              <form className="space-y-4" onSubmit={onCreate}>
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
                <div className="flex flex-col gap-[6px]">
                  <label className="text-[13px] font-medium text-gray-700">{t("projects.description")}</label>
                  <textarea
                    placeholder={t("projects.descriptionPlaceholder")}
                    className="w-full rounded-xl border border-gray-200 bg-white px-4 py-2.5 text-[14px] text-gray-900 placeholder:text-gray-400 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 outline-none transition-all duration-200 resize-none h-20"
                    value={newProject.description}
                    onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
                  />
                </div>
                <div className="flex gap-3 pt-4">
                  <Button type="submit" variant="primary" loading={creating} size="lg">
                    {t("projects.createProject")}
                  </Button>
                  <Button type="button" variant="outlined" onClick={() => onSelectProject(projects[0]?.id || null)} size="lg">
                    {t("common.cancel")}
                  </Button>
                </div>
              </form>
            </div>
          </div>
        ) : selectedProject ? (
          // View/Edit Mode
          <>
            <header className="flex flex-col border-b border-gray-200">
              <div className="px-6 py-4 flex items-center justify-between">
                <div>
                  <h2 className="text-[20px] font-bold text-gray-900">{selectedProject.name}</h2>
                  <p className="text-[13px] text-gray-500" dir="ltr">{selectedProject.domain || ""}</p>
                </div>
                <Button variant="danger" onClick={() => setDeleteConfirmId(selectedProject.id)} size="sm">
                  {t("common.delete")}
                </Button>
              </div>

              {/* TABS (Logical properties for padding/margins) */}
              <div className="flex gap-6 px-6 overflow-x-auto no-scrollbar">
                {[
                  { id: "general", label: t("projects.tabGeneral") },
                  { id: "wordpress", label: t("projects.tabWordpress") },
                  { id: "rules", label: t("projects.tabRules") },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={clsx(
                      "pb-3 text-[14px] font-semibold whitespace-nowrap transition-colors border-b-2",
                      activeTab === tab.id
                        ? "border-teal-600 text-teal-600"
                        : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                    )}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </header>

            <div className="flex-1 overflow-y-auto p-6">
              <div className="max-w-2xl">
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
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400">
            {t("projects.noProjects")}
          </div>
        )}
      </main>

    </section>
  );
}

/* ═══════════════════════════════════════════════════════════════
   TAB COMPONENTS
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
        <label className="text-[13px] font-medium text-gray-700">{t("projects.description")}</label>
        <textarea
          className="w-full rounded-xl border border-gray-200 bg-white px-4 py-2.5 text-[14px] text-gray-900 outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 resize-none h-24"
          value={draft.description}
          onChange={(e) => setDraft((p) => ({ ...p, description: e.target.value }))}
        />
      </div>
      <Button variant="primary" loading={saving} onClick={() => void onSave()}>{t("common.save")}</Button>
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
    <div className="space-y-4">
      <div className="mb-6 rounded-xl bg-blue-50/50 p-4 border border-blue-100/50">
        <p className="text-[13px] text-blue-800 leading-relaxed">{t("projects.wpSubtitle")}</p>
      </div>
      <InputField
        label={t("projects.wpUrl")}
        helperText={t("projects.wpUrlHelper")}
        value={wpUrl}
        onChange={(e) => setWpUrl(e.target.value)}
        dir="ltr"
      />
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
      <div className="flex gap-3 pt-4">
        <Button variant="primary" loading={saving} onClick={() => void save()}>{t("projects.wpSave")}</Button>
        <Button variant="outlined" loading={testing} onClick={() => void testConnection()}>{t("projects.wpTestConnection")}</Button>
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
    <div className="space-y-4 flex flex-col h-full">
      <p className="text-[13px] text-gray-500 mb-2">{t("projects.rulebookEmpty")}</p>
      <textarea
        disabled={loading}
        className="w-full flex-1 min-h-[300px] rounded-xl border border-gray-200 bg-white px-4 py-3 text-[14px] text-gray-900 outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 resize-none disabled:opacity-50"
        value={rulebook}
        onChange={(e) => setRulebook(e.target.value)}
        placeholder="- Use formal tone&#10;- Avoid competitor names..."
      />
      <div>
        <Button variant="primary" loading={saving || loading} disabled={loading} onClick={() => void save()}>
          {t("common.save")}
        </Button>
      </div>
    </div>
  );
}
