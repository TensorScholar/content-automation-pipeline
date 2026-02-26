"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
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
   Two-column split: create form + project list
   Bottom row: rulebook editor + WordPress config
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

export function ProjectsPanel({
  token, projects, selectedProjectId, onSelectProject, onProjectsRefresh,
}: ProjectsPanelProps) {
  const { t, locale } = useI18n();
  const { showToast } = useToast();
  const [creating, setCreating] = useState(false);
  const [editingProjectId, setEditingProjectId] = useState<string | null>(null);
  const [rulebook, setRulebook] = useState("");
  const [rulebookLoading, setRulebookLoading] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId]
  );

  const [newProject, setNewProject] = useState({
    name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "",
  });

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

  // Load rulebook for selected project
  useEffect(() => {
    const loadRulebook = async () => {
      if (!selectedProject) { setRulebook(""); return; }
      setRulebookLoading(true);
      try {
        const payload = await apiRequest<RulebookResponse>(`/projects/${selectedProject.id}/rulebook`, { token });
        setRulebook(payload.content ?? "");
      } catch { setRulebook(""); }
      finally { setRulebookLoading(false); }
    };
    void loadRulebook();
  }, [selectedProject, token]);

  const onCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setCreating(true);
    try {
      await apiRequest<Project, Record<string, string>>("/projects", {
        method: "POST", token,
        body: { name: newProject.name.trim(), domain: newProject.domain.trim(), vertical: resolvedVertical, description: newProject.description.trim() },
      });
      showToast("success", t("toast.projectCreated"));
      setNewProject({ name: "", domain: "", vertical: VERTICAL_OPTIONS[0].value, customVertical: "", description: "" });
      await onProjectsRefresh();
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

  const onSaveRulebook = async () => {
    if (!selectedProject) return;
    try {
      await apiRequest(`/projects/${selectedProject.id}/rulebook`, { method: "POST", token, body: { content: rulebook } });
      showToast("success", t("toast.rulebookSaved"));
    } catch (e) { showToast("error", extractError(e)); }
  };

  // Domain validation
  const domainPattern = /^[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$/;
  const domainValid = newProject.domain.length > 0 ? domainPattern.test(newProject.domain) : null;

  return (
    <section className="animate-fade-in space-y-6">
      <header className="flex items-center justify-between">
        <h2 className="text-display-lg text-ink">{t("projects.title")}</h2>
        <Button variant="outlined" onClick={() => void onProjectsRefresh()}>
          {t("common.refresh")}
        </Button>
      </header>

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
        <p className="text-body-md text-ink-secondary">{t("projects.confirmDeleteMsg")}</p>
      </Modal>

      <div className="grid gap-5 xl:grid-cols-2">
        {/* ── Create Project Form ── */}
        <article className="elevated-card p-5">
          <h3 className="mb-4 text-heading-sm text-ink">{t("projects.createNew")}</h3>
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
              helperText={t("projects.domainHelper")}
              successText={domainValid === true ? t("projects.domainValid") : undefined}
              errorText={domainValid === false ? t("projects.domainInvalid") : undefined}
              value={newProject.domain}
              onChange={(e) => setNewProject((p) => ({ ...p, domain: e.target.value }))}
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
              <label className="text-body-sm font-medium text-ink">{t("projects.description")}</label>
              <textarea
                placeholder={t("projects.descriptionPlaceholder")}
                className="smx-input h-20 resize-none rounded-sm"
                value={newProject.description}
                onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
              />
            </div>
            <Button type="submit" loading={creating} fullWidth size="lg">
              {t("projects.createProject")}
            </Button>
          </form>
        </article>

        {/* ── Project List ── */}
        <article className="elevated-card overflow-hidden">
          <div className="border-b border-border px-5 py-4">
            <h3 className="text-heading-sm text-ink">{t("projects.list")}</h3>
          </div>
          <div className="space-y-2 p-3">
            {projects.length === 0 ? (
              <EmptyState
                illustration={<EmptyIllustration className="h-20 w-20" />}
                title={t("projects.emptyTitle")}
                subtitle={t("projects.emptySubtitle")}
              />
            ) : (
              projects.map((project) => (
                <ProjectRow
                  key={project.id}
                  token={token}
                  project={project}
                  selected={project.id === selectedProjectId}
                  editing={editingProjectId === project.id}
                  onSelected={() => onSelectProject(project.id)}
                  onEditStart={() => setEditingProjectId(project.id)}
                  onEditDone={async () => { setEditingProjectId(null); await onProjectsRefresh(); }}
                  onDeleteRequest={() => setDeleteConfirmId(project.id)}
                />
              ))
            )}
          </div>
        </article>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        {/* ── Rulebook Editor ── */}
        <article className="elevated-card p-5">
          <h3 className="text-heading-sm text-ink">{t("projects.rulebook")}</h3>
          <p className="mt-1 text-body-sm text-ink-secondary">
            {selectedProject ? selectedProject.name : t("shell.noProject")}
          </p>
          <textarea
            disabled={!selectedProject || rulebookLoading}
            className="smx-input mt-4 h-40 resize-none rounded-sm disabled:opacity-50"
            value={rulebook}
            onChange={(e) => setRulebook(e.target.value)}
          />
          <Button className="mt-3" disabled={!selectedProject} onClick={() => void onSaveRulebook()}>
            {t("common.save")}
          </Button>
        </article>

        {/* ── WordPress Config ── */}
        <WordPressCard token={token} selectedProject={selectedProject} onProjectsRefresh={onProjectsRefresh} />
      </div>
    </section>
  );
}

/* ── Project Row ── */

function ProjectRow({
  token, project, selected, editing, onSelected, onEditStart, onEditDone, onDeleteRequest,
}: {
  token: string; project: Project; selected: boolean; editing: boolean;
  onSelected: () => void; onEditStart: () => void; onEditDone: () => Promise<void>; onDeleteRequest: () => void;
}) {
  const { t } = useI18n();
  const [draft, setDraft] = useState({ name: project.name, domain: project.domain ?? "", description: project.description ?? "" });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setDraft({ name: project.name, domain: project.domain ?? "", description: project.description ?? "" });
  }, [project.name, project.domain, project.description]);

  const onSave = async () => {
    setSaving(true);
    try {
      await apiRequest(`/projects/${project.id}`, {
        method: "PUT", token,
        body: { name: draft.name.trim(), domain: draft.domain.trim(), description: draft.description.trim() },
      });
      await onEditDone();
    } catch { /* handled at panel level */ }
    finally { setSaving(false); }
  };

  return (
    <div className={`rounded-sm border p-3 transition-all duration-base ${selected ? "border-brand/30 bg-brand/5" : "border-border bg-surface"
      }`}>
      {editing ? (
        <div className="space-y-2">
          <input className="smx-input rounded-sm" value={draft.name} onChange={(e) => setDraft((p) => ({ ...p, name: e.target.value }))} />
          <input className="smx-input rounded-sm" value={draft.domain} onChange={(e) => setDraft((p) => ({ ...p, domain: e.target.value }))} />
          <textarea className="smx-input h-16 resize-none rounded-sm" value={draft.description} onChange={(e) => setDraft((p) => ({ ...p, description: e.target.value }))} />
          <div className="flex gap-2">
            <Button size="sm" loading={saving} onClick={() => void onSave()}>{t("common.save")}</Button>
            <Button size="sm" variant="outlined" onClick={() => void onEditDone()}>{t("common.cancel")}</Button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <button type="button" onClick={onSelected} className="min-w-0 text-start">
            <div className="flex items-center gap-2">
              <p className="truncate text-body-md font-semibold text-ink">{project.name}</p>
              {project.wordpress_url && <StatusBadge variant="success" dot={false}>WP</StatusBadge>}
            </div>
            <p className="truncate text-body-sm text-ink-secondary">{project.domain || t("projects.noDomain")}</p>
          </button>
          <div className="flex shrink-0 gap-2">
            <Button size="sm" variant="outlined" onClick={onEditStart}>{t("common.edit")}</Button>
            <Button size="sm" variant="danger" onClick={onDeleteRequest}>{t("common.delete")}</Button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── WordPress Config Card ── */

function WordPressCard({
  token, selectedProject, onProjectsRefresh,
}: {
  token: string; selectedProject: Project | null; onProjectsRefresh: () => Promise<void>;
}) {
  const { t } = useI18n();
  const { showToast } = useToast();
  const [wpUrl, setWpUrl] = useState("");
  const [wpUsername, setWpUsername] = useState("");
  const [wpPassword, setWpPassword] = useState("");
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setWpUrl(selectedProject?.wordpress_url ?? "");
    setWpUsername(selectedProject?.wordpress_username ?? "");
    setWpPassword("");
  }, [selectedProject]);

  const save = async () => {
    if (!selectedProject) return;
    setSaving(true);
    try {
      const payload: Record<string, string> = {
        wordpress_url: wpUrl.trim(), wordpress_username: wpUsername.trim(),
      };
      if (wpPassword.trim()) payload.wordpress_app_password = wpPassword.trim();
      await apiRequest(`/projects/${selectedProject.id}`, { method: "PUT", token, body: payload });
      showToast("success", t("toast.wpSaved"));
      await onProjectsRefresh();
    } catch (e) { showToast("error", extractError(e)); }
    finally { setSaving(false); }
  };

  const testConnection = async () => {
    if (!selectedProject) return;
    setTesting(true);
    try {
      const payload = await apiRequest<{ connected?: boolean; actionable_message?: string }>(
        `/projects/${selectedProject.id}/wordpress/test-connection`, { method: "POST", token }
      );
      if (payload.connected) showToast("success", t("toast.wpTestSuccess"));
      else showToast("error", payload.actionable_message ?? t("toast.wpTestFailed"));
    } catch (e) { showToast("error", extractError(e)); }
    finally { setTesting(false); }
  };

  const connected = Boolean(selectedProject?.wordpress_url);

  return (
    <article className="elevated-card p-5">
      <div className="mb-1 flex items-center gap-2">
        <h3 className="text-heading-sm text-ink">{t("projects.wordpress")}</h3>
        {connected && <StatusBadge variant="success">{t("projects.wpConnected")}</StatusBadge>}
      </div>
      <p className="mb-4 text-body-sm text-ink-secondary">
        {selectedProject ? selectedProject.name : t("shell.noProject")}
      </p>
      <div className="space-y-3">
        <InputField label={t("projects.wpUrl")} helperText={t("projects.wpUrlHelper")} disabled={!selectedProject} value={wpUrl} onChange={(e) => setWpUrl(e.target.value)} />
        <InputField label={t("projects.wpUsername")} disabled={!selectedProject} value={wpUsername} onChange={(e) => setWpUsername(e.target.value)} />
        <InputField label={t("projects.wpPassword")} type="password" disabled={!selectedProject} value={wpPassword} onChange={(e) => setWpPassword(e.target.value)} />
      </div>
      <div className="mt-4 flex gap-2">
        <Button loading={saving} disabled={!selectedProject} onClick={() => void save()}>{t("projects.wpSave")}</Button>
        <Button variant="outlined" loading={testing} disabled={!selectedProject} onClick={() => void testConnection()}>{t("projects.wpTestConnection")}</Button>
      </div>
    </article>
  );
}
