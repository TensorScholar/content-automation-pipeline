"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import { apiRequest, ApiError } from "@/lib/api";
import { User } from "@/types/models";

interface LoginResponse {
  access_token: string;
  token_type: string;
}

interface AuthContextValue {
  token: string | null;
  user: User | null;
  loading: boolean;
  isAdmin: boolean;
  login: (email: string, password: string, remember?: boolean) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const PERSISTENT_TOKEN_STORAGE_KEY = "cap.access_token";
const SESSION_TOKEN_STORAGE_KEY = "cap.session_access_token";
const MANAGER_ROLE_ALIASES = new Set(["admin", "manager", "superuser", "owner"]);

function delay(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve) => {
    if (signal?.aborted) {
      resolve();
      return;
    }
    const onAbort = () => {
      window.clearTimeout(timeout);
      resolve();
    };
    const timeout = window.setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const mountedRef = useRef(true);
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const loadUser = useCallback(async (authToken: string, signal?: AbortSignal) => {
    const me = await apiRequest<User>("/auth/users/me", { token: authToken, signal });
    if (mountedRef.current && !signal?.aborted) {
      setUser(me);
    }
    return me;
  }, []);

  const loadUserWithRetry = useCallback(async (authToken: string, attempts = 3, signal?: AbortSignal): Promise<User | undefined> => {
    let lastError: unknown = null;
    for (let index = 0; index < attempts; index += 1) {
      if (signal?.aborted) return undefined;
      try {
        const userData = await loadUser(authToken, signal);
        return userData;
      } catch (error) {
        if (signal?.aborted) return undefined;
        lastError = error;
        const shouldRetry =
          index < attempts - 1 &&
          (!(error instanceof ApiError) || error.status >= 500 || error.status === 401 || error.status === 408);
        if (!shouldRetry) {
          throw error;
        }
        await delay(350 * (index + 1), signal);
      }
    }
    throw lastError;
  }, [loadUser]);

  const refreshUser = useCallback(async () => {
    if (!token) {
      return;
    }
    await loadUser(token);
  }, [loadUser, token]);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const bootstrap = async () => {
      const storedToken =
        window.localStorage.getItem(PERSISTENT_TOKEN_STORAGE_KEY) ??
        window.sessionStorage.getItem(SESSION_TOKEN_STORAGE_KEY);
      if (!storedToken) {
        if (active && mountedRef.current) {
          setLoading(false);
        }
        return;
      }
      try {
        if (active && mountedRef.current) {
          setToken(storedToken);
        }
        await loadUserWithRetry(storedToken, 3, controller.signal);
      } catch {
        window.localStorage.removeItem(PERSISTENT_TOKEN_STORAGE_KEY);
        window.sessionStorage.removeItem(SESSION_TOKEN_STORAGE_KEY);
        if (active && mountedRef.current) {
          setToken(null);
          setUser(null);
        }
      } finally {
        if (active && mountedRef.current) {
          setLoading(false);
        }
      }
    };
    void bootstrap();
    return () => {
      active = false;
      controller.abort();
    };
  }, [loadUserWithRetry]);

  const login = useCallback(async (email: string, password: string, remember = true) => {
    const normalizedEmail = email.trim();
    const formData = new URLSearchParams();
    formData.set("username", normalizedEmail);
    formData.set("password", password);

    const tokenResponse = await apiRequest<LoginResponse>("/auth/token", {
      method: "POST",
      formData,
      headers: { "X-Login-Identifier": normalizedEmail.toLowerCase() }
    });
    const nextToken = tokenResponse.access_token;

    try {
      // CRITICAL FIX: Load user data BEFORE setting token state
      // This prevents race conditions where useEffect hooks fire with token=truthy but user=null
      const userData = await loadUserWithRetry(nextToken);

      // Verify user data was actually loaded
      if (!userData) {
        throw new Error("Failed to load user data");
      }

      // Atomically update both user and token state to prevent intermediate states
      if (mountedRef.current) {
        setUser(userData);
        setToken(nextToken);
      }

      // Persist to storage after state update
      if (remember) {
        window.localStorage.setItem(PERSISTENT_TOKEN_STORAGE_KEY, nextToken);
        window.sessionStorage.removeItem(SESSION_TOKEN_STORAGE_KEY);
      } else {
        window.sessionStorage.setItem(SESSION_TOKEN_STORAGE_KEY, nextToken);
        window.localStorage.removeItem(PERSISTENT_TOKEN_STORAGE_KEY);
      }
    } catch (error) {
      // Clean up on failure
      window.localStorage.removeItem(PERSISTENT_TOKEN_STORAGE_KEY);
      window.sessionStorage.removeItem(SESSION_TOKEN_STORAGE_KEY);
      if (mountedRef.current) {
        setToken(null);
        setUser(null);
      }
      throw error;
    }
  }, [loadUserWithRetry]);

  const logout = useCallback(async () => {
    try {
      if (token) {
        await apiRequest("/auth/logout", { method: "POST", token });
      }
    } catch {
      // Intentionally ignored: local session still gets cleared.
    } finally {
      window.localStorage.removeItem(PERSISTENT_TOKEN_STORAGE_KEY);
      window.sessionStorage.removeItem(SESSION_TOKEN_STORAGE_KEY);
      if (mountedRef.current) {
        setToken(null);
        setUser(null);
      }
    }
  }, [token]);

  const value = useMemo<AuthContextValue>(() => {
    const role = (user?.role ?? "").toLowerCase();
    const isAdmin = Boolean(user?.is_superuser) || MANAGER_ROLE_ALIASES.has(role);
    return {
      token,
      user,
      loading,
      isAdmin,
      login,
      logout,
      refreshUser
    };
  }, [loading, login, logout, refreshUser, token, user]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return ctx;
}
