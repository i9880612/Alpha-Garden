import assert from "node:assert/strict";
import { test } from "node:test";
import { ConsoleEventStream } from "../../src/api/event-stream.ts";

const tick = () => new Promise(resolve => setImmediate(resolve));
const response = data => new Response(JSON.stringify(data));
function transport(t, read = async () => response({ value: 1 })) {
  const previous = { fetch: globalThis.fetch, EventSource: globalThis.EventSource };
  const requests = [], signals = [];
  class Source {
    static CLOSED = 2;
    static instances = [];
    readyState = 0;
    listeners = new Map();
    constructor(url) { this.url = url; Source.instances.push(this); }
    addEventListener(name, listener) { this.listeners.set(name, listener); }
    close() { this.readyState = Source.CLOSED; }
    open() { this.readyState = 1; this.onopen?.(); }
    emit(versions = { database: 0, jobs: 0, session: 0 }) {
      this.listeners.get("versions")?.({ data: JSON.stringify(versions) });
    }
  }
  globalThis.EventSource = Source;
  globalThis.fetch = (path, options) => { requests.push(path); signals.push(options.signal); return read(path, options); };
  const stream = new ConsoleEventStream();
  stream.start();
  t.after(() => { stream.close(); Object.assign(globalThis, previous); });
  return { stream, requests, signals, Source };
}

test("initial reads run once and notifications refresh only changed topics", async t => {
  const { stream, requests, Source } = transport(t);
  const off = [stream.subscribe("/api/jobs", () => {}), stream.subscribe("/api/jobs", () => {}),
    stream.subscribe("/api/dashboard/activity", () => {})];
  await tick();
  assert.deepEqual(requests, []);
  assert.equal(Source.instances.length, 1);
  const source = Source.instances[0];
  assert.equal(source.url, "/api/events");
  source.open(); source.emit(); await tick();
  assert.deepEqual(requests, ["/api/jobs", "/api/dashboard/activity"]);
  source.emit(); await tick();
  assert.equal(requests.length, 2);
  source.emit({ database: 0, jobs: 1, session: 0 }); await tick();
  assert.deepEqual(requests, ["/api/jobs", "/api/dashboard/activity", "/api/jobs"]);
  off.forEach(unsubscribe => unsubscribe());
  source.emit({ database: 1, jobs: 2, session: 1 }); await tick();
  assert.equal(requests.length, 3);
  assert.equal(stream.getResource("/api/jobs").data, null);
});

test("development subscription replay sends one read per surviving resource", async t => {
  const { stream, requests, signals, Source } = transport(t);
  const paths = ["/api/session", "/api/jobs", "/api/dashboard/activity"];
  const first = paths.map(path => stream.subscribe(path, () => {}));
  first.forEach(off => off()); stream.close();
  const second = paths.map(path => stream.subscribe(path, () => {})); stream.start();
  await tick(); Source.instances[0].open(); Source.instances[0].emit(); await tick();
  assert.deepEqual(requests, paths);
  assert.ok(signals.every(signal => !signal.aborted));
  assert.equal(Source.instances.length, 1);
  second.forEach(off => off());
});

test("a resource removed before initialization sends no request", async t => {
  const { stream, requests, Source } = transport(t);
  stream.subscribe("/api/formulas?page=1", () => {})(); await tick();
  assert.deepEqual(requests, []);
  assert.equal(Source.instances.length, 0);
});

test("changing pages and filters keeps the connection and discards abandoned responses", async t => {
  const pending = [];
  const { stream, Source, requests, signals } = transport(t, () => new Promise(resolve => pending.push(resolve)));
  const firstPath = "/api/formulas?execution=finished&page=1";
  const nextPath = "/api/formulas?execution=planned&page=2&page_size=20";
  const off = stream.subscribe(firstPath, () => {});
  await tick(); const source = Source.instances[0]; source.open(); source.emit();
  off(); const offNext = stream.subscribe(nextPath, () => {}); await tick();
  assert.equal(Source.instances.length, 1);
  assert.equal(source.readyState, 1);
  assert.equal(signals[0].aborted, true);
  pending[1](response({ page: 2 })); pending[0](response({ page: 1 })); await tick();
  assert.equal(stream.getResource(nextPath).data.page, 2);
  assert.equal(stream.getResource(firstPath).data, null);
  assert.deepEqual(requests, [firstPath, nextPath]);
  offNext();
});

test("slow reads coalesce notifications while fast sections and errors stay independent", async t => {
  const pending = [];
  const { stream, Source, requests } = transport(t, path => path === "/api/jobs"
    ? Promise.resolve(response({ busy: false })) : new Promise(resolve => pending.push(resolve)));
  const off = [stream.subscribe("/api/jobs", () => {}), stream.subscribe("/api/dashboard/research", () => {})];
  await tick(); const source = Source.instances[0]; source.open(); source.emit(); await tick();
  for (let i = 1; i <= 10; i++) source.emit({ database: i, jobs: 1, session: 0 });
  await tick();
  assert.equal(pending.length, 1);
  assert.equal(stream.getResource("/api/jobs").data.busy, false);
  pending[0](response({ count: 1 })); await tick();
  assert.equal(pending.length, 2);
  pending[1](new Response(JSON.stringify({ error: "console_data_unavailable" }), { status: 503 })); await tick();
  assert.equal(stream.getResource("/api/dashboard/research").error, "console_data_unavailable");
  assert.equal(stream.getResource("/api/jobs").error, null);
  assert.equal(requests.length, 4);
  source.emit({ database: 11, jobs: 1, session: 0 });
  pending[2](response({ count: 3 })); await tick();
  assert.equal(stream.getResource("/api/dashboard/research").error, null);
  assert.equal(stream.getResource("/api/dashboard/research").data.count, 3);
  off.forEach(unsubscribe => unsubscribe());
});

test("reconnection refreshes current data even after server revisions reset", async t => {
  let busy = true;
  const { stream, Source, requests } = transport(t, async path => response(path === "/api/auth" ? { authenticated: true } : { busy }));
  const off = stream.subscribe("/api/jobs", () => {});
  await tick(); const source = Source.instances[0]; source.open(); source.emit(); await tick();
  assert.equal(stream.getResource("/api/jobs").data.busy, true);
  source.readyState = 0; source.onerror();
  assert.equal(stream.getConnection().connected, false);
  busy = false; source.open(); source.emit(); await tick();
  assert.equal(stream.getConnection().connected, true);
  assert.equal(stream.getResource("/api/jobs").data.busy, false);
  assert.deepEqual(requests, ["/api/jobs", "/api/auth", "/api/jobs"]);
  assert.equal(Source.instances.length, 1);
  source.close(); source.onerror();
  assert.equal(stream.getConnection().failed, true);
  off();
});

test("a revoked stream closes and returns to login without starting another stream", async t => {
  const previousWindow = globalThis.window;
  const redirects = [];
  globalThis.window = { location: { pathname: "/runs", search: "", hash: "", replace: path => redirects.push(path) } };
  t.after(() => { globalThis.window = previousWindow; });
  const { stream, Source } = transport(t);
  const off = stream.subscribe("/api/jobs", () => {});
  await tick(); const source = Source.instances[0]; source.open(); source.emit(); await tick();
  source.listeners.get("unauthorized")({ data: "{}" });
  assert.equal(source.readyState, Source.CLOSED);
  assert.deepEqual(redirects, ["/login?from=%2Fruns"]);
  assert.equal(Source.instances.length, 1);
  off();
});

test("a rejected reconnect checks authentication before returning to login", async t => {
  const previousWindow = globalThis.window;
  const redirects = [];
  globalThis.window = { location: { pathname: "/runs", search: "", hash: "", replace: path => redirects.push(path) } };
  t.after(() => { globalThis.window = previousWindow; });
  const { stream, Source } = transport(t, async () => response({ authenticated: false }));
  const off = stream.subscribe("/api/jobs", () => {});
  await tick(); const source = Source.instances[0]; source.onerror(); await tick();
  assert.equal(source.readyState, Source.CLOSED);
  assert.deepEqual(redirects, ["/login?from=%2Fruns"]);
  off();
});
