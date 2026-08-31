"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import { User } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { InputField } from "@/components/ui/input-field";
import { useToast } from "@/components/ui/toast";
import { ToggleSwitch } from "@/components/ui/toggle-switch";
import { EmptyState } from "@/components/ui/empty-state";
import { Modal } from "@/components/ui/modal";

/* ═══════════════════════════════════════════════════════════════
   Spec: Screen 7 — Users Management (Manager-Only)
   Create user form + users table with activate/deactivate
   RoleGuard enforced at app-shell level (isAdmin prop)
   ═══════════════════════════════════════════════════════════════ */

interface UsersPanelProps {
  token: string;
  isAdmin: boolean;
  currentUserId: string;
}

export function UsersPanel({ token, isAdmin, currentUserId }: UsersPanelProps) {
  const { t, locale } = useI18n();
  const { showToast } = useToast();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newEmail, setNewEmail] = useState("");
  const [newFullName, setNewFullName] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newIsAdmin, setNewIsAdmin] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);

  const loadUsers = useCallback(async (signal?: AbortSignal) => {
    if (!isAdmin) {
      setUsers([]);
      setLoading(false);
      return;
    }
    try {
      const list = await apiRequest<User[]>("/auth/users", { token, signal });
      if (signal?.aborted) return;
      setUsers(Array.isArray(list) ? list : []);
    } catch {
      if (signal?.aborted) return;
      setUsers([]);
    } finally {
      if (signal?.aborted) return;
      setLoading(false);
    }
  }, [isAdmin, token]);

  useEffect(() => {
    if (!isAdmin) {
      setUsers([]);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    void loadUsers(controller.signal);
    return () => controller.abort();
  }, [isAdmin, loadUsers]);

  const counts = useMemo(() => {
    const total = users.length;
    const active = users.filter((u) => u.is_active !== false).length;
    return { total, active, inactive: total - active };
  }, [users]);

  const onCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!isAdmin) return;
    setCreating(true);
    try {
      await apiRequest("/auth/users", {
        method: "POST", token,
        body: { email: newEmail.trim(), full_name: newFullName.trim(), password: newPassword, role: newIsAdmin ? "admin" : "user" },
      });
      showToast("success", t("users.userCreated"));
      setNewEmail(""); setNewFullName(""); setNewPassword(""); setNewIsAdmin(false);
      setCreateOpen(false);
      await loadUsers();
    } catch (e) {
      showToast("error", e instanceof ApiError ? e.detail : t("common.unexpectedError"));
    } finally { setCreating(false); }
  };

  const toggleActive = async (user: User) => {
    if (!isAdmin || user.id === currentUserId) return;
    const action = user.is_active === false ? "activate" : "deactivate";
    try {
      await apiRequest(`/auth/users/${user.id}/${action}`, { method: "POST", token });
      showToast("success", action === "activate" ? t("users.activated") : t("users.deactivated"));
      await loadUsers();
    } catch (e) {
      showToast("error", e instanceof ApiError ? e.detail : t("common.unexpectedError"));
    }
  };

  if (!isAdmin) {
    return (
      <EmptyState
        illustration={(
          <span className="grid h-10 w-10 place-items-center rounded-md border border-line bg-surface text-ink-secondary" aria-hidden>
            <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <rect x="5.5" y="10" width="13" height="9" rx="2" />
              <path d="M8.5 10V7.5a3.5 3.5 0 0 1 7 0V10" />
            </svg>
          </span>
        )}
        title={t("users.adminRequired")}
        subtitle={t("users.adminRequiredMsg")}
      />
    );
  }

  return (
    <section className="smx-page !max-w-none relative flex min-h-0 flex-col gap-4">
      <div className="smx-page-header">
        <div className="min-w-0 flex-1">
          <h2 className="smx-page-title">{t("users.title")}</h2>
          <p className="mt-1 text-sm text-ink-muted">
            {counts.total} {t("users.total")} · {counts.active} {t("users.active")}
            {counts.inactive > 0 ? ` · ${counts.inactive} ${t("users.inactive")}` : ""}
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>{t("users.addUser")}</Button>
      </div>

      <Modal
        open={createOpen}
        onClose={() => !creating && setCreateOpen(false)}
        title={t("users.addUser")}
        footer={null}
      >
        <form className="space-y-4" onSubmit={onCreate}>
          <InputField label={t("common.email")} type="email" required value={newEmail} onChange={(e) => setNewEmail(e.target.value)} />
          <InputField label={t("users.fullName")} required value={newFullName} onChange={(e) => setNewFullName(e.target.value)} />
          <InputField label={t("users.password")} type="password" required value={newPassword} onChange={(e) => setNewPassword(e.target.value)} />
          <div className="border-t border-line pt-4">
            <ToggleSwitch checked={newIsAdmin} onChange={setNewIsAdmin} label={t("users.grantAdmin")} />
          </div>
          <div className="flex justify-end gap-2 border-t border-line pt-4">
            <Button type="button" variant="ghost" disabled={creating} onClick={() => setCreateOpen(false)}>{t("common.cancel")}</Button>
            <Button type="submit" loading={creating}>{t("users.createUser")}</Button>
          </div>
        </form>
      </Modal>

      <div className="min-h-0 flex-1 overflow-auto border-t border-line">
        <table className="w-full border-collapse text-start">
          <thead className="sticky top-0 z-10 border-b border-line bg-[rgb(var(--bg-secondary))]">
            <tr className="text-xs font-medium text-ink-tertiary">
              <th className="px-4 py-3 text-start font-medium">{t("common.email")}</th>
              <th className="px-4 py-3 text-start font-medium">{t("users.fullName")}</th>
              <th className="px-4 py-3 text-start font-medium">{t("common.role")}</th>
              <th className="px-4 py-3 text-start font-medium">{t("users.statusLabel")}</th>
              <th className="px-4 py-3 text-start font-medium">{t("users.createdAt")}</th>
              <th className="sr-only w-14 px-4 py-3 text-end">{t("users.action")}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {loading ? (
              [1, 2, 3, 4, 5].map((i) => (
                <tr key={i} className="animate-pulse">
                  <td className="px-4 py-4"><div className="h-4 w-3/4 rounded bg-ink/[0.055]" /></td>
                  <td className="px-4 py-4"><div className="h-4 w-1/2 rounded bg-ink/[0.055]" /></td>
                  <td className="px-4 py-4"><div className="h-4 w-16 rounded bg-ink/[0.055]" /></td>
                  <td className="px-4 py-4"><div className="h-4 w-20 rounded bg-ink/[0.055]" /></td>
                  <td className="px-4 py-4"><div className="h-4 w-24 rounded bg-ink/[0.055]" /></td>
                  <td className="px-4 py-4" />
                </tr>
              ))
            ) : users.length === 0 ? (
              <tr><td colSpan={6} className="px-6 py-16 text-center text-base text-ink-muted">{t("common.noData")}</td></tr>
            ) : users.map((user) => {
              const isActive = user.is_active !== false;
              const isManagerRole = user.is_superuser || user.role === "admin" || user.role === "manager";
              const roleLabel = isManagerRole ? t("role.manager") : t("role.user");
              const isSelf = user.id === currentUserId;
              return (
                <tr key={user.id} className="transition-colors duration-fast hover:bg-ink/[0.03]">
                  <td className="px-4 py-3.5 text-sm font-medium text-ink">{user.email}</td>
                  <td className="px-4 py-3.5 text-sm text-ink-secondary">{user.full_name || "—"}</td>
                  <td className="px-4 py-3.5 text-sm text-ink-secondary">{roleLabel}</td>
                  <td className="px-4 py-3.5">
                    <span className={clsx("inline-flex items-center gap-2 text-sm font-medium", isActive ? "text-success" : "text-ink-muted")}>
                      <span className={clsx("h-1.5 w-1.5 rounded-full", isActive ? "bg-success" : "bg-ink-tertiary")} />
                      {isActive ? t("users.active") : t("users.inactive")}
                    </span>
                  </td>
                  <td className="px-4 py-3.5 text-xs tabular-nums text-ink-tertiary">{formatDate(user.created_at, locale)}</td>
                  <td className="px-4 py-3.5 text-end">
                    <button
                      type="button"
                      onClick={() => void toggleActive(user)}
                      disabled={isSelf}
                      aria-label={isSelf ? t("users.cannotRemoveSelf") : isActive ? t("users.deactivate") : t("users.activate")}
                      title={isSelf ? t("users.cannotRemoveSelf") : isActive ? t("users.deactivate") : t("users.activate")}
                      className={clsx(
                        "inline-flex h-8 w-8 items-center justify-center rounded-md transition-colors duration-fast",
                        isSelf ? "cursor-not-allowed text-ink-tertiary opacity-35" : isActive ? "text-ink-muted hover:bg-error/10 hover:text-error" : "text-success hover:bg-success/10",
                      )}
                    >
                      {isActive ? (
                        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" /></svg>
                      ) : (
                        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M5 13l4 4L19 7" /></svg>
                      )}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatDate(d: string | undefined, locale: string): string {
  if (!d) return "—";
  const localeName = locale === "fa" ? "fa-IR" : locale === "ar" ? "ar-SA" : "en-US";
  try { return new Intl.DateTimeFormat(localeName, { year: "numeric", month: "short", day: "numeric" }).format(new Date(d)); }
  catch { return d; }
}
