import assert from "node:assert/strict";
import { test } from "node:test";
import { consoleReturnPath, login, loginPath } from "../../src/api/auth.ts";

test("login returns only to known console routes, never an external URL", () => {
  for (const path of ["/data", "/operators", "/analysis", "/seeds", "/optimization", "/archive", "/submitted"])
    assert.equal(consoleReturnPath(`${path}?search=close`), `${path}?search=close`);
  assert.equal(consoleReturnPath("/runs?page=2"), "/runs?page=2");
  assert.equal(loginPath("/runs?page=2"), "/login?from=%2Fruns%3Fpage%3D2");
  for (const value of [null, "//evil.invalid/runs", "https://evil.invalid", "javascript:alert(1)", "http://[", "/login", "/missing"])
    assert.equal(consoleReturnPath(value), "/");
});

test("login success and invalid credentials follow the server response", async t => {
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  globalThis.fetch = async (path, options) => {
    assert.equal(path, "/api/login");
    assert.equal(options.method, "POST");
    assert.equal(options.headers["Content-Type"], "application/json");
    assert.deepEqual(JSON.parse(options.body), { username: "admin", password: "123123" });
    return new Response(JSON.stringify({ authenticated: true }));
  };
  await login("admin", "123123");
  globalThis.fetch = async () => new Response(JSON.stringify({ error: "console_credentials_invalid" }), { status: 401 });
  await assert.rejects(login("admin", "wrong"), /console_credentials_invalid/);
});
