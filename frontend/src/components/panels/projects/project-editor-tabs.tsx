"use client";

import { useEffect, useState } from "react";
import { ApiError, apiRequest } from "@/lib/api";
import { Project } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { useToast } from "@/components/ui/toast";
import { Button } from "@/components/ui/button";
import { InputField } from "@/components/ui/input-field";
import { SelectDropdown } from "@/components/ui/select-dropdown";
import type { SelectOption } from "@/components/ui/select-dropdown";
import { VERTICAL_OPTIONS, extractError } from "./project-constants";

interface RulebookResponse { content?: string; }

function projectDraft(project: Project) {
  const preset = VERTICAL_OPTIONS.find(
    (option) => option.value === project.vertical || option.en === project.vertical
  );
  return {
    name: project.name,
    domain: project.domain ?? "",
    description: project.description ?? "",
    vertical: preset?.value ?? (project.vertical ? "__custom__" : VERTICAL_OPTIONS[0].value),
    customVertical: preset ? "" : project.vertical ?? "",
  };
}

export function GeneralTab({
  token,
  project,
  canManageProjects,
  verticalOptions,
  onProjectsRefresh,
}: {
  token: string;
  project: Project;
  canManageProjects: boolean;
  verticalOptions: SelectOption[];
  onProjectsRefresh: () => Promise<void>;
}) {
  const { t } = useI18n();
  const [draft, setDraft] = useState(() => projectDraft(project));
  const [saving, setSaving] = useState(false);
  const { showToast } = useToast();

  useEffect(() => {
    setDraft(projectDraft(project));
  }, [project]);

  const initialDraft = projectDraft(project);
  const isDirty = JSON.stringify(draft) !== JSON.stringify(initialDraft);
  const normalizedName = draft.name.trim();
  const normalizedDomain = draft.domain.trim();
  const domainPattern = /^[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$/;
  const domainValid = normalizedDomain.length === 0 || domainPattern.test(
    normalizedDomain.replace(/^https?:\/\//, "").replace(/\/+$/, "")
  );
  const resolvedVertical = draft.vertical === "__custom__"
    ? draft.customVertical.trim()
    : VERTICAL_OPTIONS.find((option) => option.value === draft.vertical)?.en ?? draft.vertical;
  const canSave = canManageProjects
    && isDirty
    && normalizedName.length > 0
    && domainValid
    && resolvedVertical.length > 0;

  const onSave = async () => {
    if (!canSave || saving) return;
    setSaving(true);
    try {
      await apiRequest(`/projects/${project.id}`, {
        method: "PUT", token,
        body: {
          name: normalizedName,
          domain: normalizedDomain,
          description: draft.description.trim(),
          vertical: resolvedVertical,
        },
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
          required
          value={draft.name}
          disabled={!canManageProjects}
          onChange={(e) => setDraft((p) => ({ ...p, name: e.target.value }))}
        />
        <InputField
          label={t("projects.domain")}
          value={draft.domain}
          disabled={!canManageProjects}
          errorText={!domainValid ? t("projects.domainInvalid") : undefined}
          successText={domainValid && normalizedDomain ? t("projects.domainValid") : undefined}
          onChange={(e) => setDraft((p) => ({ ...p, domain: e.target.value }))}
          dir="ltr"
        />
        <SelectDropdown
          label={t("projects.industry")}
          options={verticalOptions}
          value={draft.vertical}
          disabled={!canManageProjects}
          onChange={(vertical) => setDraft((current) => ({ ...current, vertical }))}
        />
        {draft.vertical === "__custom__" && (
          <InputField
            label={t("projects.customVertical")}
            required
            disabled={!canManageProjects}
            value={draft.customVertical}
            onChange={(event) => setDraft((current) => ({
              ...current,
              customVertical: event.target.value,
            }))}
          />
        )}
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-ink-secondary">{t("projects.description")}</label>
          <textarea
            aria-label={t("projects.description")}
            disabled={!canManageProjects}
            className="min-h-[120px] w-full resize-none rounded-xl border border-line bg-surface px-3 py-2 text-base text-ink outline-none transition-colors duration-150 focus:border-brand focus:ring-1 focus:ring-brand/20 disabled:cursor-not-allowed disabled:opacity-60"
            value={draft.description}
            onChange={(e) => setDraft((p) => ({ ...p, description: e.target.value }))}
          />
        </div>
      </div>

      {canManageProjects && (
        <div className="flex justify-end pt-2">
          <Button
            variant="primary"
            loading={saving}
            disabled={!canSave}
            onClick={() => void onSave()}
            className="min-w-[120px]"
          >
            {t("common.save")}
          </Button>
        </div>
      )}
    </div>
  );
}

export function WordPressTab({
  token,
  project,
  canManageProjects,
  onProjectsRefresh,
}: {
  token: string;
  project: Project;
  canManageProjects: boolean;
  onProjectsRefresh: () => Promise<void>;
}) {
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
    if (!canManageProjects || saving) return;
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
        <h3 className="mb-1 text-body-lg font-bold text-ink">{t("projects.tabWordpress")}</h3>
        <p className="text-sm text-ink-tertiary leading-relaxed">{t("projects.wpSubtitle")}</p>
      </div>

      {/* WordPress settings */}
      <div className="space-y-6 smx-panel-subtle p-5 md:p-6">
        <InputField
          label={t("projects.wpUrl")}
          helperText={t("projects.wpUrlHelper")}
          value={wpUrl}
          disabled={!canManageProjects}
          onChange={(e) => setWpUrl(e.target.value)}
          dir="ltr"
        />
        <div className="grid md:grid-cols-2 gap-6">
          <InputField
            label={t("projects.wpUsername")}
            value={wpUsername}
            disabled={!canManageProjects}
            onChange={(e) => setWpUsername(e.target.value)}
            dir="ltr"
          />
          <InputField
            label={t("projects.wpPassword")}
            type="password"
            helperText={t("projects.wpPasswordTooltip")}
            value={wpPassword}
            disabled={!canManageProjects}
            onChange={(e) => setWpPassword(e.target.value)}
            dir="ltr"
          />
        </div>

        <div className="flex justify-end gap-3 pt-6 border-block-start border-line">
          <Button variant="outlined" loading={testing} onClick={() => void testConnection()}>{t("projects.wpTestConnection")}</Button>
          {canManageProjects && (
            <Button variant="primary" loading={saving} onClick={() => void save()} className="min-w-[120px]">
              {t("projects.wpSave")}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

export function RulebookTab({
  token,
  project,
  canManageProjects,
}: {
  token: string;
  project: Project;
  canManageProjects: boolean;
}) {
  const { t, locale } = useI18n();
  const { showToast } = useToast();
  const [rulebook, setRulebook] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const rulebookPlaceholder = locale === "fa"
    ? "- از لحن رسمی استفاده شود\n- نام رقیب ذکر نشود..."
    : locale === "ar"
      ? "- استخدم نبرة رسمية\n- تجنب ذكر أسماء المنافسين..."
      : "- Use formal tone\n- Avoid competitor names...";

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    apiRequest<RulebookResponse>(`/projects/${project.id}/rulebook`, { token, signal: controller.signal })
      .then(res => {
        if (!controller.signal.aborted) setRulebook(res.content ?? "");
      })
      .catch(() => {
        if (!controller.signal.aborted) setRulebook("");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [project.id, token]);

  const save = async () => {
    if (!canManageProjects || saving || loading || !rulebook.trim()) return;
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
        <p className="text-sm text-ink-tertiary">{t("projects.rulebookEmpty")}</p>
      </div>

      {/* Rulebook editor */}
      <div className="group relative flex min-h-[400px] flex-1 flex-col overflow-hidden rounded-xl border border-line bg-surface transition-colors duration-150 focus-within:border-brand focus-within:ring-2 focus-within:ring-brand/20">

          <textarea
            aria-label={t("projects.rulebook")}
            disabled={loading || !canManageProjects}
            className="w-full h-full flex-1 bg-transparent p-6 text-base text-ink leading-relaxed outline-none border-none resize-y disabled:opacity-50"
            value={rulebook}
            onChange={(e) => setRulebook(e.target.value)}
            placeholder={rulebookPlaceholder}
          />
      </div>

      {/* Save action */}
      {canManageProjects && (
        <div className="absolute bottom-0 inset-inline-end-0 flex justify-end">
          <Button
            variant="primary"
            loading={saving || loading}
            disabled={loading || !rulebook.trim()}
            onClick={() => void save()}
            className="min-w-[140px] shadow-sm"
          >
            {t("common.save")}
          </Button>
        </div>
      )}

    </div>
  );
}
