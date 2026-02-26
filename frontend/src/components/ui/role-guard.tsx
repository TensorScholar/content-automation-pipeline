"use client";
import { useAuth } from "@/providers/auth-provider";

export interface RoleGuardProps {
    allowedRoles: string[];
    children: React.ReactNode;
    fallback?: React.ReactNode;
}

/**
 * RoleGuard — renders children only if the current user's role is in allowedRoles.
 *
 * Role resolution:
 * - is_superuser = true → role is 'manager'
 * - Otherwise uses user.role (lowercased)
 *
 * Never throws — renders fallback or null if access denied.
 */
export function RoleGuard({ allowedRoles, children, fallback = null }: RoleGuardProps) {
    const { user } = useAuth();

    if (!user) return <>{fallback}</>;

    const userRole = user.is_superuser ? "manager" : (user.role ?? "user").toLowerCase();

    if (!allowedRoles.includes(userRole)) {
        return <>{fallback}</>;
    }

    return <>{children}</>;
}

/** Hook version for conditional logic outside JSX */
export function useUserRole(): string {
    const { user } = useAuth();
    if (!user) return "guest";
    return user.is_superuser ? "manager" : (user.role ?? "user").toLowerCase();
}

export function useHasRole(allowedRoles: string[]): boolean {
    const role = useUserRole();
    return allowedRoles.includes(role);
}
