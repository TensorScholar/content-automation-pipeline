export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ?? "http://127.0.0.1:8000";
const API_PROXY_BASE_URL = "/api";

export class ApiError extends Error {
  status: number;
  detail: string;
  retryAfter: number | null;

  constructor(status: number, detail: string, retryAfter: number | null = null) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.retryAfter = retryAfter;
  }
}

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

interface ApiRequestOptions<TBody> {
  method?: HttpMethod;
  token?: string | null;
  body?: TBody;
  formData?: URLSearchParams;
  timeoutMs?: number;
  signal?: AbortSignal;
}

function toQueryString(query?: Record<string, string | number | boolean | undefined>) {
  if (!query) {
    return "";
  }
  const params = new URLSearchParams();
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined) {
      params.set(key, String(value));
    }
  });
  const encoded = params.toString();
  return encoded ? `?${encoded}` : "";
}

function parseDetail(payload: unknown, status: number): string {
  if (typeof payload === "string" && payload.length > 0) {
    return payload;
  }
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as Record<string, unknown>).detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      return detail
        .map((entry) => (typeof entry === "string" ? entry : JSON.stringify(entry)))
        .join(" | ");
    }
    if (detail !== undefined && detail !== null) {
      return String(detail);
    }
  }
  return `HTTP ${status}`;
}

function isLikelyNetworkOrCorsError(error: unknown): boolean {
  return error instanceof TypeError;
}

function canUseNextDevProxy(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  const { hostname, port } = window.location;
  return (hostname === "localhost" || hostname === "127.0.0.1") && port === "3001";
}

function backendUnavailableError(baseUrl: string): ApiError {
  return new ApiError(0, `Smarlux backend is unavailable at ${baseUrl}. Confirm the API service is running and reachable.`);
}

export async function apiRequest<TResponse, TBody = Record<string, unknown>>(
  path: string,
  options?: ApiRequestOptions<TBody>,
  query?: Record<string, string | number | boolean | undefined>
): Promise<TResponse> {
  const {
    method = "GET",
    token = null,
    body,
    formData,
    timeoutMs = 30000,
    signal
  } = options ?? {};

  const requestController = new AbortController();
  let timedOut = false;
  const abortRequest = () => {
    if (!requestController.signal.aborted) {
      requestController.abort();
    }
  };

  if (signal?.aborted) {
    abortRequest();
  } else {
    signal?.addEventListener("abort", abortRequest, { once: true });
  }

  const timeout = window.setTimeout(() => {
    timedOut = true;
    abortRequest();
  }, timeoutMs);
  const composedSignal = requestController.signal;

  const headers = new Headers();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const requestInit: RequestInit = {
    method,
    headers,
    signal: composedSignal
  };

  if (formData) {
    headers.set("Content-Type", "application/x-www-form-urlencoded");
    requestInit.body = formData.toString();
  } else if (body !== undefined) {
    headers.set("Content-Type", "application/json");
    requestInit.body = JSON.stringify(body);
  }

  const fetchJson = async (baseUrl: string): Promise<TResponse> => {
    const requestUrl = `${baseUrl}${path}${toQueryString(query)}`;
    let response: Response;
    try {
      response = await fetch(requestUrl, {
        ...requestInit,
        mode: "cors"
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        if (signal?.aborted && !timedOut) {
          throw error;
        }
        throw new ApiError(408, "Request timeout");
      }
      throw error;
    }
    const contentType = response.headers.get("Content-Type") ?? "";

    let payload: unknown = null;
    if (contentType.includes("application/json")) {
      payload = await response.json();
    } else {
      const text = await response.text();
      payload = text.length > 0 ? text : null;
    }

    if (!response.ok) {
      let retryAfter: number | null = null;
      const retryHeader = response.headers.get("Retry-After");
      if (retryHeader) retryAfter = parseInt(retryHeader, 10) || null;
      if (!retryAfter && typeof payload === "object" && payload !== null && "retry_after" in payload) {
        retryAfter = Number((payload as Record<string, unknown>).retry_after) || null;
      }
      throw new ApiError(response.status, parseDetail(payload, response.status), retryAfter);
    }

    return payload as TResponse;
  };

  try {
    try {
      return await fetchJson(API_BASE_URL);
    } catch (error) {
      const shouldTryProxy =
        API_BASE_URL !== API_PROXY_BASE_URL &&
        canUseNextDevProxy() &&
        isLikelyNetworkOrCorsError(error);
      if (!shouldTryProxy) {
        if (isLikelyNetworkOrCorsError(error)) {
          throw backendUnavailableError(API_BASE_URL);
        }
        throw error;
      }
      try {
        return await fetchJson(API_PROXY_BASE_URL);
      } catch (proxyError) {
        if (isLikelyNetworkOrCorsError(proxyError)) {
          throw backendUnavailableError(API_BASE_URL);
        }
        throw proxyError;
      }
    }
  } finally {
    window.clearTimeout(timeout);
    signal?.removeEventListener("abort", abortRequest);
  }
}
