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
  getUnauthorizedGeneration = () => 0,
} = {}) {
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
    const unauthorizedGeneration = getUnauthorizedGeneration();
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
      if (
        !suppressUnauthorizedHandler
        && unauthorizedGeneration === getUnauthorizedGeneration()
      ) {
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
