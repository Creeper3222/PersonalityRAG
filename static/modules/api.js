export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = Number(status || 0);
  }
}

function detailText(detail, fallback) {
  if (typeof detail === "string" && detail) return detail;
  if (detail && typeof detail === "object") {
    return detail.message || detail.detail || JSON.stringify(detail);
  }
  return fallback;
}

export async function responseError(response) {
  let detail = response.statusText;
  try {
    const payload = await response.json();
    detail = detailText(payload?.detail, detail);
  } catch {}
  return new ApiError(detail, response.status);
}

export function createApiClient({
  baseUrl = "/api/v1",
  onUnauthorized = () => {},
  unauthorizedMessage = () => "Unauthorized",
  onOperationalError = () => {},
} = {}) {
  const inFlightGets = new Map();
  const signalIds = new WeakMap();
  let nextSignalId = 1;

  function signalKey(signal) {
    if (!signal) return "none";
    if (!signalIds.has(signal)) signalIds.set(signal, nextSignalId++);
    return String(signalIds.get(signal));
  }

  function singleFlightKey(path, requestOptions, headers, errorPolicy) {
    const method = String(requestOptions.method || "GET").toUpperCase();
    if (method !== "GET" || requestOptions.body != null) return "";
    const headerKey = Object.entries(headers)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([name, value]) => `${name}:${value}`)
      .join("|");
    return `${path}|${headerKey}|${requestOptions.cache || ""}|${signalKey(requestOptions.signal)}|${errorPolicy}`;
  }

  async function execute(path, requestOptions, headers, suppressUnauthorizedHandler, suppressOperationalError) {
    let response;
    try {
      response = await fetch(baseUrl + path, {
        credentials: "same-origin",
        ...requestOptions,
        headers,
      });
    } catch (cause) {
      if (cause?.name === "AbortError") throw cause;
      const error = new ApiError(cause?.message || "Network request failed", 0);
      if (!suppressOperationalError) onOperationalError(error);
      throw error;
    }
    if (response.status === 401) {
      if (!suppressUnauthorizedHandler) {
        onUnauthorized();
        throw new ApiError(unauthorizedMessage(), 401);
      }
      throw await responseError(response);
    }
    if (!response.ok) {
      const error = await responseError(response);
      if (error.status >= 500 && !suppressOperationalError) {
        onOperationalError(error);
      }
      throw error;
    }
    return response.status === 204 ? null : response.json();
  }

  return async function api(path, options = {}) {
    const {
      suppressUnauthorizedHandler = false,
      suppressOperationalError = false,
      ...requestOptions
    } = options;
    const headers = { ...(requestOptions.headers || {}) };
    if (!(requestOptions.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const key = singleFlightKey(
      path,
      requestOptions,
      headers,
      `${suppressUnauthorizedHandler}:${suppressOperationalError}`,
    );
    if (key && inFlightGets.has(key)) return inFlightGets.get(key);
    const request = execute(
      path,
      requestOptions,
      headers,
      suppressUnauthorizedHandler,
      suppressOperationalError,
    );
    if (!key) return request;
    inFlightGets.set(key, request);
    try {
      return await request;
    } finally {
      if (inFlightGets.get(key) === request) inFlightGets.delete(key);
    }
  };
}

export function parseDownloadFilename(disposition, fallback) {
  const text = disposition || "";
  const encoded = text.match(/filename\*=UTF-8''([^;]+)/i);
  if (encoded) {
    try {
      return decodeURIComponent(encoded[1]);
    } catch {}
  }
  const quoted = text.match(/filename="?([^";]+)"?/i);
  return quoted?.[1] || fallback;
}
