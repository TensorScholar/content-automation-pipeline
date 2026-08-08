interface PendingRequest {
  promise: Promise<unknown>;
  timestamp: number;
}

const pendingRequests = new Map<string, PendingRequest>();
const REQUEST_DEDUP_TTL = 2_000;
const DEFAULT_GET_RETRIES = 2;
const RETRYABLE_STATUS_CODES = new Set([408, 425, 429, 500, 502, 503, 504]);
let requestCounter = 0;

function stableSerialize(value: unknown): string {
  if (value === undefined) return "";
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableSerialize).join(",")}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${stableSerialize(record[key])}`)
    .join(",")}}`;
}

function fingerprint(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function generateRequestKey(
  path: string,
  options?: ApiRequestOptions<unknown>,
  query?: Record<string, string | number | boolean | undefined>
): string {
  const method = options?.method ?? "GET";
  const identity = fingerprint(options?.token ?? "anonymous");
  const body = stableSerialize(options?.body);
  const queryString = stableSerialize(query);
  const headers = stableSerialize(options?.headers);
  return `${method}:${identity}:${path}:${queryString}:${headers}:${body}`;
}

function cleanupStaleRequests() {
  const now = Date.now();
  for (const [key, request] of pendingRequests.entries()) {
    if (now - request.timestamp > REQUEST_DEDUP_TTL) {
      pendingRequests.delete(key);
    }
  }
}

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ?? "http://127.0.0.1:8000";
const API_PROXY_BASE_URL = "/api";

export class ApiError extends Error {
  status: number;
  detail: string;
  retryAfter: number | null;
  requestId: string | null;

  constructor(
    status: number,
    detail: string,
    retryAfter: number | null = null,
    requestId: string | null = null
  ) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.retryAfter = retryAfter;
    this.requestId = requestId;
  }
}

function createRequestId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  requestCounter += 1;
  return `smarlux-${Date.now().toString(36)}-${requestCounter.toString(36)}`;
}

function createAbortError(): Error {
  const error = new Error("Aborted");
  error.name = "AbortError";
  return error;
}

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

interface ApiRequestOptions<TBody> {
  method?: HttpMethod;
  token?: string | null;
  body?: TBody;
  formData?: URLSearchParams;
  headers?: Record<string, string>;
  timeoutMs?: number;
  maxRetries?: number;
  signal?: AbortSignal;
}

function toQueryString(query?: Record<string, string | number | boolean | undefined>) {
  if (!query) return "";
  const params = new URLSearchParams();
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined) params.set(key, String(value));
  });
  const encoded = params.toString();
  return encoded ? `?${encoded}` : "";
}

function parseDetail(payload: unknown, status: number): string {
  if (typeof payload === "string" && payload.length > 0) return payload;
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as Record<string, unknown>).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((entry) => (typeof entry === "string" ? entry : JSON.stringify(entry)))
        .join(" | ");
    }
    if (detail !== undefined && detail !== null) return String(detail);
  }
  return `HTTP ${status}`;
}

function parseRetryAfter(response: Response, payload: unknown): number | null {
  const retryHeader = response.headers.get("Retry-After");
  if (retryHeader) {
    const seconds = Number(retryHeader);
    if (Number.isFinite(seconds) && seconds >= 0) return Math.min(seconds, 30);
    const retryDate = Date.parse(retryHeader);
    if (Number.isFinite(retryDate)) {
      return Math.min(30, Math.max(0, Math.ceil((retryDate - Date.now()) / 1_000)));
    }
  }
  if (typeof payload === "object" && payload !== null && "retry_after" in payload) {
    const seconds = Number((payload as Record<string, unknown>).retry_after);
    if (Number.isFinite(seconds) && seconds >= 0) return Math.min(seconds, 30);
  }
  return null;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function isLikelyNetworkOrCorsError(error: unknown): boolean {
  return error instanceof TypeError;
}

function canUseNextDevProxy(): boolean {
  if (typeof window === "undefined") return false;
  const { hostname, port } = window.location;
  return (hostname === "localhost" || hostname === "127.0.0.1") && port === "3001";
}

function backendUnavailableError(baseUrl: string, requestId: string): ApiError {
  return new ApiError(
    0,
    `Smarlux backend is unavailable at ${baseUrl}. Confirm the API service is running and reachable.`,
    null,
    requestId
  );
}

function shouldRetry(error: unknown): boolean {
  if (isLikelyNetworkOrCorsError(error)) return true;
  return error instanceof ApiError && RETRYABLE_STATUS_CODES.has(error.status);
}

function retryDelayMs(error: unknown, attempt: number): number {
  if (error instanceof ApiError && error.retryAfter !== null) {
    return error.retryAfter * 1_000;
  }
  const exponential = Math.min(2_000, 250 * 2 ** attempt);
  const deterministicJitter = (requestCounter * 37 + attempt * 53) % 120;
  return exponential + deterministicJitter;
}

async function sleep(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) throw createAbortError();
  await new Promise<void>((resolve, reject) => {
    const timer = globalThis.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      globalThis.clearTimeout(timer);
      reject(createAbortError());
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
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
    headers: extraHeaders,
    timeoutMs = 30_000,
    maxRetries = method === "GET" ? DEFAULT_GET_RETRIES : 0,
    signal,
  } = options ?? {};
  const shouldDeduplicate = method === "GET" && signal === undefined;
  const requestKey = generateRequestKey(
    path,
    options as ApiRequestOptions<unknown> | undefined,
    query
  );

  if (shouldDeduplicate) {
    const pending = pendingRequests.get(requestKey);
    if (pending && Date.now() - pending.timestamp < REQUEST_DEDUP_TTL) {
      return pending.promise as Promise<TResponse>;
    }
    requestCounter += 1;
    if (requestCounter % 50 === 0) cleanupStaleRequests();
  }

  const requestController = new AbortController();
  let timedOut = false;
  const abortRequest = () => {
    if (!requestController.signal.aborted) requestController.abort();
  };

  if (signal?.aborted) abortRequest();
  else signal?.addEventListener("abort", abortRequest, { once: true });

  const timeout = globalThis.setTimeout(() => {
    timedOut = true;
    abortRequest();
  }, Math.max(1_000, timeoutMs));

  const cleanupRequestLifecycle = () => {
    globalThis.clearTimeout(timeout);
    signal?.removeEventListener("abort", abortRequest);
  };
  const composedSignal = requestController.signal;

  const headers = new Headers(extraHeaders);
  const clientRequestId = headers.get("X-Request-ID") ?? createRequestId();
  headers.set("X-Request-ID", clientRequestId);
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const requestInit: RequestInit = { method, headers, signal: composedSignal };
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
      response = await fetch(requestUrl, { ...requestInit, mode: "cors" });
    } catch (error) {
      if (isAbortError(error)) {
        if (signal?.aborted && !timedOut) throw error;
        throw new ApiError(408, "Request timeout", null, clientRequestId);
      }
      throw error;
    }

    let payload: unknown = null;
    const contentType = response.headers.get("Content-Type") ?? "";
    const responseRequestId =
      response.headers.get("X-Request-ID") ??
      response.headers.get("X-Correlation-ID") ??
      clientRequestId;
    const hasResponseBody = response.status !== 204 && response.status !== 205;
    if (hasResponseBody) {
      const text = await response.text();
      if (text.length > 0 && contentType.includes("application/json")) {
        try {
          payload = JSON.parse(text);
        } catch {
          if (response.ok) {
            throw new ApiError(502, "Invalid JSON response from the API", null, responseRequestId);
          }
          payload = text;
        }
      } else {
        payload = text.length > 0 ? text : null;
      }
    }

    if (!response.ok) {
      throw new ApiError(
        response.status,
        parseDetail(payload, response.status),
        parseRetryAfter(response, payload),
        responseRequestId
      );
    }
    return payload as TResponse;
  };

  const executeOnce = async (): Promise<TResponse> => {
    try {
      return await fetchJson(API_BASE_URL);
    } catch (error) {
      const tryProxy =
        API_BASE_URL !== API_PROXY_BASE_URL &&
        canUseNextDevProxy() &&
        isLikelyNetworkOrCorsError(error);
      if (!tryProxy) {
        if (isLikelyNetworkOrCorsError(error)) throw backendUnavailableError(API_BASE_URL, clientRequestId);
        throw error;
      }
      try {
        return await fetchJson(API_PROXY_BASE_URL);
      } catch (proxyError) {
        if (isLikelyNetworkOrCorsError(proxyError)) {
          throw backendUnavailableError(API_BASE_URL, clientRequestId);
        }
        throw proxyError;
      }
    }
  };

  const executeRequest = async (): Promise<TResponse> => {
    const retries = method === "GET" ? Math.max(0, Math.min(maxRetries, 3)) : 0;
    for (let attempt = 0; ; attempt += 1) {
      try {
        return await executeOnce();
      } catch (error) {
        if (composedSignal.aborted || attempt >= retries || !shouldRetry(error)) throw error;
        await sleep(retryDelayMs(error, attempt), composedSignal);
      }
    }
  };

  if (shouldDeduplicate) {
    const requestPromise = executeRequest().finally(() => {
      cleanupRequestLifecycle();
      pendingRequests.delete(requestKey);
    });
    pendingRequests.set(requestKey, { promise: requestPromise, timestamp: Date.now() });
    return requestPromise;
  }

  try {
    return await executeRequest();
  } finally {
    cleanupRequestLifecycle();
  }
}
