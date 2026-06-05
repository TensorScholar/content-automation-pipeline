"use client";

import { AppShell } from "./app-shell";
import { LoginCard } from "./login-card";
import { useAuth } from "@/providers/auth-provider";
import { Skeleton } from "./ui/skeleton";

export function RootApp() {
  const { token, user, loading } = useAuth();

  if (loading) {
    return (
      <main className="mx-auto flex min-h-screen max-w-7xl items-center px-6">
        <div className="grid w-full gap-4">
          <Skeleton className="h-10 w-56" />
          <Skeleton className="h-56 w-full" />
          <Skeleton className="h-56 w-full" />
        </div>
      </main>
    );
  }

  if (!token || !user) {
    return <LoginCard />;
  }

  return <AppShell token={token} user={user} />;
}
