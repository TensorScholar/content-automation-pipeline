"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
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
    <section className="animate-fade-in space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-display-lg text-ink">{t("users.title")}</h2>
        <div className="flex items-center gap-2">
          <StatusBadge variant="neutral">{t("users.total")}: {counts.total}</StatusBadge>
          <StatusBadge variant="success">{t("users.active")}: {counts.active}</StatusBadge>
          {counts.inactive > 0 && (
            <StatusBadge variant="error">{t("users.inactive")}: {counts.inactive}</StatusBadge>
          )}
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[380px_1fr]">
        {/* ── Add User Form ── */}
        <article className="elevated-card p-5">
          <h3 className="mb-4 text-heading-sm text-ink">{t("users.addUser")}</h3>
          <form className="space-y-3" onSubmit={onCreate}>
            <InputField label={t("common.email")} type="email" required value={newEmail} onChange={(e) => setNewEmail(e.target.value)} />
            <InputField label={t("users.fullName")} required value={newFullName} onChange={(e) => setNewFullName(e.target.value)} />
            <InputField label={t("users.password")} type="password" required value={newPassword} onChange={(e) => setNewPassword(e.target.value)} />
            <ToggleSwitch checked={newIsAdmin} onChange={setNewIsAdmin} label={t("users.grantAdmin")} />
            <Button type="submit" loading={creating} fullWidth size="lg">
              {t("users.createUser")}
            </Button>
          </form>
        </article>

        {/* ── Users Table ── */}
        <div className="elevated-card overflow-hidden">
          <div className="max-h-[60vh] overflow-auto">
            <table className="w-full text-start">
              <thead className="sticky top-0 z-10 border-b border-border bg-surface-alt/95 backdrop-blur-sm">
                <tr className="text-body-sm font-semibold uppercase tracking-wider text-ink-secondary">
                  <th className="px-4 py-3 text-start">{t("common.email")}</th>
                  <th className="px-4 py-3 text-start">{t("users.fullName")}</th>
                  <th className="px-4 py-3 text-start">{t("common.role")}</th>
                  <th className="px-4 py-3 text-start">{t("users.statusLabel")}</th>
                  <th className="px-4 py-3 text-start">{t("users.createdAt")}</th>
                  <th className="px-4 py-3 text-end">{t("users.action")}</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <SkeletonRows rows={4} cols={6} />
                ) : users.length === 0 ? (
                  <tr><td colSpan={6} className="px-4 py-8 text-center text-body-md text-ink-secondary">{t("common.noData")}</td></tr>
                ) : (
                  users.map((user) => {
                    const isActive = user.is_active !== false;
                    const roleLabel = user.is_superuser ? t("role.manager") : (user.role ?? "user");
                    return (
                      <tr key={user.id} className="border-b border-border transition-colors duration-fast hover:bg-surface-alt/50">
                        <td dir="ltr" className="px-4 py-3 text-body-md text-ink text-start">{user.email}</td>
                        <td className="px-4 py-3 text-body-md text-ink-secondary">{user.full_name || "—"}</td>
                        <td className="px-4 py-3">
                          <StatusBadge
                            variant={roleLabel === t("role.manager") ? "info" : "neutral"}
                            dot={false}
                          >
                            {roleLabel}
                          </StatusBadge>
                        </td>
                        <td className="px-4 py-3">
                          <StatusBadge variant={isActive ? "success" : "error"}>
                            {isActive ? t("users.active") : t("users.inactive")}
                          </StatusBadge>
                        </td>
                        <td className="px-4 py-3 text-body-sm text-ink-secondary">{formatDate(user.created_at)}</td>
                        <td className="px-4 py-3 text-end">
                          <Button
                            size="sm"
                            variant={isActive ? "danger" : "primary"}
                            onClick={() => void toggleActive(user)}
                          >
                            {isActive ? t("users.deactivate") : t("users.activate")}
                          </Button>
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
