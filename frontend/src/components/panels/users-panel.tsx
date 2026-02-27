"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { ApiError, apiRequest } from "@/lib/api";
import { User } from "@/types/models";
import { useI18n } from "@/i18n/provider";
import { Button } from "@/components/ui/button";
import { InputField } from "@/components/ui/input-field";
import { useToast } from "@/components/ui/toast";
import { StatusBadge } from "@/components/ui/status-badge";
import { ToggleSwitch } from "@/components/ui/toggle-switch";
import { SkeletonRows } from "@/components/ui/skeleton-loader";
import { EmptyState } from "@/components/ui/empty-state";

/* ═══════════════════════════════════════════════════════════════
   Spec: Screen 7 — Users Management (Manager-Only)
   Create user form + users table with activate/deactivate
   RoleGuard enforced at app-shell level (isAdmin prop)
   ═══════════════════════════════════════════════════════════════ */

interface UsersPanelProps {
  token: string;
  isAdmin: boolean;
}

export function UsersPanel({ token, isAdmin }: UsersPanelProps) {
  const { t } = useI18n();
  const { showToast } = useToast();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newEmail, setNewEmail] = useState("");
  const [newFullName, setNewFullName] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newIsAdmin, setNewIsAdmin] = useState(false);

  useEffect(() => {
    void loadUsers();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const loadUsers = async () => {
    try {
      const list = await apiRequest<User[]>("/auth/users", { token });
      setUsers(Array.isArray(list) ? list : []);
    } catch { setUsers([]); }
    finally { setLoading(false); }
  };

  const counts = useMemo(() => {
    const total = users.length;
    const active = users.filter((u) => u.is_active !== false).length;
    return { total, active, inactive: total - active };
  }, [users]);

  const onCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
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
    <section className="animate-fade-in relative flex flex-col space-y-6 bg-[#F5F5F7] min-h-[calc(100vh-80px)] p-4 md:p-8">
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-6 pb-2">
        <div className="flex-1">
          <h2 className="text-[28px] font-bold text-slate-900 tracking-tight">{t("users.title")}</h2>
        </div>
        <div className="flex flex-wrap items-center gap-2 mt-2 md:mt-0 bg-white/60 backdrop-blur-md border border-gray-200/60 p-2 rounded-2xl shadow-sm">
          <div className="px-3 py-1.5 rounded-xl bg-white border border-slate-200 text-slate-600 text-[13px] font-medium shadow-sm">{t("users.total")}: <span className="font-bold text-slate-900 ms-1">{counts.total}</span></div>
          <div className="px-3 py-1.5 rounded-xl bg-emerald-50 border border-emerald-100 text-emerald-700 text-[13px] font-medium shadow-sm">{t("users.active")}: <span className="font-bold text-emerald-900 ms-1">{counts.active}</span></div>
          {counts.inactive > 0 && (
            <div className="px-3 py-1.5 rounded-xl bg-red-50 border border-red-100 text-red-700 text-[13px] font-medium shadow-sm">{t("users.inactive")}: <span className="font-bold text-red-900 ms-1">{counts.inactive}</span></div>
          )}
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[340px_1fr] flex-1 min-h-0 items-start">
        {/* ── Add User Form ── */}
        <article className="bg-white rounded-3xl border border-slate-200/60 shadow-[0_4px_24px_rgba(0,0,0,0.02)] p-6 shrink-0">
          <h3 className="mb-6 text-[18px] font-bold text-slate-900">{t("users.addUser")}</h3>
          <form className="space-y-4" onSubmit={onCreate}>
            <InputField label={t("common.email")} type="email" required value={newEmail} onChange={(e) => setNewEmail(e.target.value)} />
            <InputField label={t("users.fullName")} required value={newFullName} onChange={(e) => setNewFullName(e.target.value)} />
            <InputField label={t("users.password")} type="password" required value={newPassword} onChange={(e) => setNewPassword(e.target.value)} />

            <div className="pt-2 pb-4 border-b border-slate-100">
              <ToggleSwitch checked={newIsAdmin} onChange={setNewIsAdmin} label={t("users.grantAdmin")} />
            </div>

            <Button type="submit" loading={creating} fullWidth className="rounded-2xl h-11 shadow-sm bg-teal-600 hover:bg-teal-700 text-[14px] font-semibold">
              {t("users.createUser")}
            </Button>
          </form>
        </article>

        {/* ── Users Table ── */}
        <div className="bg-white rounded-3xl border border-slate-200/60 shadow-[0_4px_24px_rgba(0,0,0,0.02)] overflow-hidden flex flex-col min-h-0">
          <div className="flex-1 overflow-auto rounded-3xl">
            <table className="w-full text-start border-collapse">
              <thead className="bg-[#f8fafc] sticky top-0 z-10 border-b border-slate-200/80 backdrop-blur-xl">
                <tr className="text-[12px] font-bold text-slate-500 uppercase tracking-wider">
                  <th className="px-6 py-5 text-start font-bold">{t("common.email")}</th>
                  <th className="px-6 py-5 text-start font-bold">{t("users.fullName")}</th>
                  <th className="px-6 py-5 text-start font-bold">{t("common.role")}</th>
                  <th className="px-6 py-5 text-start font-bold">{t("users.statusLabel")}</th>
                  <th className="px-6 py-5 text-start font-bold">{t("users.createdAt")}</th>
                  <th className="px-6 py-5 text-end font-bold sr-only w-16">{t("users.action")}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {loading ? (
                  [1, 2, 3, 4, 5].map((i) => (
                    <tr key={i} className="animate-pulse">
                      <td className="px-6 py-4"><div className="h-4 bg-slate-100 rounded-md w-3/4"></div></td>
                      <td className="px-6 py-4"><div className="h-4 bg-slate-50 rounded-md w-1/2"></div></td>
                      <td className="px-6 py-4"><div className="h-6 bg-slate-100 rounded-md w-16"></div></td>
                      <td className="px-6 py-4"><div className="h-4 bg-slate-50 rounded-md w-20"></div></td>
                      <td className="px-6 py-4"><div className="h-4 bg-slate-100 rounded-md w-24"></div></td>
                      <td className="px-6 py-4 text-end"><div className="h-9 w-9 bg-slate-100 rounded-xl ms-auto"></div></td>
                    </tr>
                  ))
                ) : users.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-16 text-center">
                      <div className="flex flex-col items-center justify-center opacity-80">
                        <svg className="w-16 h-16 text-slate-300 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                        </svg>
                        <span className="text-[15px] font-medium text-slate-500">{t("common.noData") || "No users found"}</span>
                      </div>
                    </td>
                  </tr>
                ) : (
                  users.map((user) => {
                    const isActive = user.is_active !== false;
                    const roleLabel = user.is_superuser ? t("role.manager") : (user.role ?? "user");
                    return (
                      <tr key={user.id} className="transition-colors duration-200 hover:bg-slate-50/50">
                        {/* Remove dir="ltr" to fix visual bug. Put email text strictly at start without html dir override */}
                        <td className="px-6 py-4 text-[14px] font-medium text-slate-900 text-start">{user.email}</td>
                        <td className="px-6 py-4 text-[14px] text-slate-600">{user.full_name || "—"}</td>
                        <td className="px-6 py-4">
                          <span className={clsx("inline-flex items-center px-2 py-1 rounded-md text-[11px] font-bold uppercase tracking-wider", roleLabel === t("role.manager") ? "bg-indigo-50 text-indigo-700" : "bg-slate-100 text-slate-600")}>
                            {roleLabel}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <span className={clsx("inline-flex items-center gap-1.5 text-[13px] font-medium", isActive ? "text-emerald-600" : "text-slate-400")}>
                            <span className={clsx("w-2 h-2 rounded-full", isActive ? "bg-emerald-500" : "bg-slate-300")} />
                            {isActive ? t("users.active") : t("users.inactive")}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-[13px] text-slate-500">{formatDate(user.created_at)}</td>
                        <td className="px-6 py-4 text-end">
                          <button
                            onClick={() => void toggleActive(user)}
                            title={isActive ? t("users.deactivate") : t("users.activate")}
                            className={clsx(
                              "p-2 rounded-xl transition-colors shadow-sm border",
                              isActive ? "bg-white text-red-500 border-red-100 hover:bg-red-50 hover:border-red-200 cursor-pointer" : "bg-white text-emerald-500 border-emerald-100 hover:bg-emerald-50 hover:border-emerald-200 cursor-pointer"
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
