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
  const { t } = useI18n();
  const { showToast } = useToast();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newEmail, setNewEmail] = useState("");
  const [newFullName, setNewFullName] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newIsAdmin, setNewIsAdmin] = useState(false);

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
        illustration={<span className="text-[3rem]">🔒</span>}
        title={t("users.adminRequired")}
        subtitle={t("users.adminRequiredMsg")}
      />
    );
  }

  return (
    <section className="macos-content-scope animate-fade-in relative flex min-h-[calc(100vh-96px)] flex-col space-y-4 bg-transparent p-3 md:p-4">
      <div className="flex flex-col gap-4 pb-1 md:flex-row md:items-start md:justify-between">
        <div className="flex-1">
          <h2 className="text-[24px] font-semibold leading-tight tracking-normal text-ink">{t("users.title")}</h2>
        </div>
        <div className="smx-panel-subtle flex flex-wrap items-center gap-1.5 p-1.5">
          <div className="rounded-full bg-white/90 px-3 py-1.5 text-[11px] font-semibold text-ink-secondary dark:bg-white/10 dark:text-gray-200">{t("users.total")}: <span className="ms-1 font-bold tabular-nums text-ink">{counts.total}</span></div>
          <div className="rounded-full bg-emerald-500/10 px-3 py-1.5 text-[11px] font-semibold text-emerald-700 dark:text-emerald-300">{t("users.active")}: <span className="ms-1 font-bold tabular-nums">{counts.active}</span></div>
          {counts.inactive > 0 && (
            <div className="rounded-full bg-amber-500/10 px-3 py-1.5 text-[11px] font-semibold text-amber-700 dark:text-amber-300">{t("users.inactive")}: <span className="ms-1 font-bold tabular-nums">{counts.inactive}</span></div>
          )}
        </div>
      </div>

      <div className="grid min-h-0 flex-1 items-start gap-4 xl:grid-cols-[1fr_minmax(300px,360px)]">
        {/* ── Add User Form ── */}
        <article className="smx-panel order-2 shrink-0 p-5 xl:order-2">
          <div className="mb-5">
            <h3 className="text-[16px] font-semibold text-ink">{t("users.addUser")}</h3>
          </div>
          <form className="space-y-4" onSubmit={onCreate}>
            <InputField label={t("common.email")} type="email" required value={newEmail} onChange={(e) => setNewEmail(e.target.value)} />
            <InputField label={t("users.fullName")} required value={newFullName} onChange={(e) => setNewFullName(e.target.value)} />
            <InputField label={t("users.password")} type="password" required value={newPassword} onChange={(e) => setNewPassword(e.target.value)} />

            <div className="border-b border-black/5 pb-4 pt-2 dark:border-white/10">
              <ToggleSwitch checked={newIsAdmin} onChange={setNewIsAdmin} label={t("users.grantAdmin")} />
            </div>

            <Button type="submit" loading={creating} fullWidth className="h-10 rounded-[12px] text-sm font-semibold">
              {t("users.createUser")}
            </Button>
          </form>
        </article>

        {/* ── Users Table ── */}
        <div className="smx-panel order-1 flex min-h-0 flex-col overflow-hidden xl:order-1">
          <div className="flex-1 overflow-auto">
            <table className="w-full text-start border-collapse">
              <thead className="sticky top-0 z-10 border-b border-black/5 bg-black/[0.02] dark:border-white/10 dark:bg-white/[0.03]">
                <tr className="text-[11px] font-semibold uppercase tracking-[0.02em] text-ink-tertiary">
                  <th className="px-5 py-3.5 text-start font-semibold">{t("common.email")}</th>
                  <th className="px-5 py-3.5 text-start font-semibold">{t("users.fullName")}</th>
                  <th className="px-5 py-3.5 text-start font-semibold">{t("common.role")}</th>
                  <th className="px-5 py-3.5 text-start font-semibold">{t("users.statusLabel")}</th>
                  <th className="px-5 py-3.5 text-start font-semibold">{t("users.createdAt")}</th>
                  <th className="px-6 py-3.5 text-end font-semibold sr-only w-16">{t("users.action")}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-black/5 dark:divide-white/10">
                {loading ? (
                  [1, 2, 3, 4, 5].map((i) => (
                    <tr key={i} className="animate-pulse">
                      <td className="px-5 py-4"><div className="h-4 w-3/4 rounded-md bg-slate-100 dark:bg-white/10"></div></td>
                      <td className="px-5 py-4"><div className="h-4 w-1/2 rounded-md bg-slate-100 dark:bg-white/10"></div></td>
                      <td className="px-5 py-4"><div className="h-6 w-16 rounded-full bg-slate-100 dark:bg-white/10"></div></td>
                      <td className="px-5 py-4"><div className="h-4 w-20 rounded-md bg-slate-100 dark:bg-white/10"></div></td>
                      <td className="px-5 py-4"><div className="h-4 w-24 rounded-md bg-slate-100 dark:bg-white/10"></div></td>
                      <td className="px-5 py-4 text-end"><div className="ms-auto h-9 w-9 rounded-full bg-slate-100 dark:bg-white/10"></div></td>
                    </tr>
                  ))
                ) : users.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-14 text-center">
                      <div className="flex flex-col items-center justify-center opacity-80">
                        <svg className="mb-4 h-16 w-16 text-slate-300 dark:text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                        </svg>
                        <span className="text-[14px] font-medium text-ink-tertiary">{t("common.noData") || "No users found"}</span>
                      </div>
                    </td>
                  </tr>
                ) : (
                  users.map((user) => {
                    const isActive = user.is_active !== false;
                    const isManagerRole = user.is_superuser || user.role === "admin" || user.role === "manager";
                    const roleLabel = isManagerRole ? t("role.manager") : t("role.user");
                    const isSelf = user.id === currentUserId;
                    return (
                      <tr key={user.id} className="transition-colors duration-200 hover:bg-black/[0.025] dark:hover:bg-white/[0.04]">
                        <td className="px-5 py-4 text-start text-[13px] font-semibold text-ink">{user.email}</td>
                        <td className="px-5 py-4 text-[13px] font-medium text-ink-secondary">{user.full_name || "—"}</td>
                        <td className="px-5 py-4">
                          <span className={clsx("inline-flex items-center rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.03em]", isManagerRole ? "bg-indigo-500/10 text-indigo-700 dark:text-indigo-200" : "bg-black/[0.04] text-ink-secondary dark:bg-white/[0.07] dark:text-gray-300")}>
                            {roleLabel}
                          </span>
                        </td>
                        <td className="px-5 py-4">
                          <span className={clsx("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold", isActive ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300" : "bg-amber-500/10 text-amber-700 dark:text-amber-300")}>
                            <span className={clsx("w-2 h-2 rounded-full", isActive ? "bg-emerald-500" : "bg-slate-300 dark:bg-white/35")} />
                            {isActive ? t("users.active") : t("users.inactive")}
                          </span>
                        </td>
                        <td className="px-5 py-4 text-[12px] font-medium tabular-nums text-ink-tertiary">{formatDate(user.created_at)}</td>
                        <td className="px-5 py-4 text-end">
                          <button
                            onClick={() => void toggleActive(user)}
                            disabled={isSelf}
                            title={isSelf ? t("users.cannotRemoveSelf") : isActive ? t("users.deactivate") : t("users.activate")}
                            className={clsx(
                              "rounded-full border p-2 transition-all duration-200",
                              isSelf
                                ? "cursor-not-allowed border-black/5 bg-black/[0.02] text-ink-tertiary opacity-40 dark:border-white/10 dark:bg-white/[0.05]"
                                : isActive ? "cursor-pointer border-black/5 bg-black/[0.02] text-rose-500 hover:bg-rose-500/10 dark:border-white/10 dark:bg-white/[0.05] dark:hover:bg-rose-500/15" : "cursor-pointer border-black/5 bg-black/[0.02] text-emerald-500 hover:bg-emerald-500/10 dark:border-white/10 dark:bg-white/[0.05] dark:hover:bg-emerald-500/15"
                            )}
                          >
                            {isActive ? (
                              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" /></svg>
                            ) : (
                              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                            )}
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}

function formatDate(d?: string): string {
  if (!d) return "—";
  try { return new Date(d).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }); }
  catch { return d; }
}
