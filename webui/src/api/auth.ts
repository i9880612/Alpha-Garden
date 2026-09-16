const consolePaths = new Set(["/", "/runs", "/formulas", "/submissions", "/settings", "/seeds", "/optimization", "/archive", "/submitted", "/data", "/operators", "/analysis"]);

export function consoleReturnPath(value: string | null) {
  try {
    const url = new URL(value || "/", "http://console.local");
    return url.origin === "http://console.local" && consolePaths.has(url.pathname)
      ? `${url.pathname}${url.search}${url.hash}` : "/";
  } catch { return "/"; }
}

export const loginPath = (from: string) => `/login?from=${encodeURIComponent(consoleReturnPath(from))}`;

export function requireLogin() {
  if (window.location.pathname !== "/login")
    window.location.replace(loginPath(window.location.pathname + window.location.search + window.location.hash));
}

export async function getAuth(signal?: AbortSignal): Promise<{ authenticated: boolean; username: string | null }> {
  const timeout = AbortSignal.timeout(15000);
  const response = await fetch("/api/auth", { cache: "no-store", signal: signal ? AbortSignal.any([signal, timeout]) : timeout });
  if (!response.ok) throw new Error("console_data_unavailable");
  return response.json();
}

export async function login(username: string, password: string) {
  const response = await fetch("/api/login", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }), signal: AbortSignal.timeout(15000),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "console_data_unavailable");
}
