import { getJson } from "./console.ts";
import { getAuth, requireLogin } from "./auth.ts";

type Topic = "database" | "jobs" | "session";
type Versions = Record<Topic, number>;
export interface ResourceState { data: unknown; error: string | null; loading: boolean }
interface Resource {
  state: ResourceState; listeners: Set<() => void>; controller: AbortController;
  revision: number; reading: boolean;
}
const loading: ResourceState = { data: null, error: null, loading: true };
const inactive: ResourceState = { data: null, error: null, loading: false };
const topicFor = (path: string): Topic => {
  const endpoint = path.split("?")[0];
  return endpoint === "/api/jobs" ? "jobs"
    : ["/api/session", "/api/settings"].includes(endpoint) ? "session" : "database";
};

/** One stable notification connection per tab; HTTP owns resource reads. */
export class ConsoleEventStream {
  private resources = new Map<string, Resource>();
  private listeners = new Set<() => void>();
  private source: EventSource | null = null;
  private versions: Versions | null = null;
  private connection = { ready: false, connected: false, failed: false };
  private active = false;
  private scheduled = false;

  getConnection = () => this.connection;
  subscribeConnection = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };
  getResource = (path: string | null) => path ? this.resources.get(path)?.state ?? loading : inactive;

  start() { this.active = true; this.scheduleConnection(); }
  close() {
    this.active = false;
    this.source?.close();
    this.source = null;
    this.versions = null;
  }

  subscribe(path: string | null, listener: () => void) {
    if (!path) return () => {};
    let resource = this.resources.get(path);
    if (!resource) {
      resource = { state: loading, listeners: new Set(), controller: new AbortController(), revision: 0, reading: false };
      this.resources.set(path, resource);
      const subscription = resource;
      // Development replay and a page changed before mounting send no abandoned reads.
      queueMicrotask(() => {
        if (this.active && this.versions && this.resources.get(path) === subscription && subscription.listeners.size)
          void this.refresh(path, subscription);
      });
      this.scheduleConnection();
    }
    resource.listeners.add(listener);
    return () => {
      resource.listeners.delete(listener);
      if (!resource.listeners.size && this.resources.get(path) === resource) {
        resource.controller.abort();
        this.resources.delete(path);
      }
    };
  }

  private async refresh(path: string, resource: Resource) {
    if (resource.reading) return;
    resource.reading = true;
    const revision = resource.revision;
    try {
      const data = await getJson(path, resource.controller.signal);
      if (this.resources.get(path) === resource)
        this.update(resource, { data, error: null, loading: false });
    } catch (error) {
      if (this.resources.get(path) === resource)
        this.update(resource, { data: resource.state.data, error: error instanceof Error ? error.message : "console_data_unavailable", loading: false });
    } finally {
      resource.reading = false;
      // Commit notifications during a slow read become one follow-up, never parallel reads.
      if (this.active && this.resources.get(path) === resource && resource.revision !== revision)
        void this.refresh(path, resource);
    }
  }

  private update(resource: Resource, state: ResourceState) {
    resource.state = state;
    resource.listeners.forEach(listener => listener());
  }

  private setConnection(connected: boolean, failed = false) {
    this.connection = { ready: this.connection.ready || connected, connected, failed };
    this.listeners.forEach(listener => listener());
  }

  private scheduleConnection() {
    if (this.scheduled) return;
    this.scheduled = true;
    queueMicrotask(() => {
      this.scheduled = false;
      if (!this.active || this.source || !this.resources.size) return;
      this.setConnection(false);
      const source = new EventSource("/api/events");
      this.source = source;
      source.onopen = () => {
        if (this.source !== source) return;
        // Reconnection must refresh even when the backend restarted its revision counters.
        this.versions = null;
      };
      source.addEventListener("versions", event => {
        if (this.source !== source) return;
        const versions = JSON.parse((event as MessageEvent<string>).data) as Versions;
        const previous = this.versions;
        this.versions = versions;
        this.setConnection(true);
        for (const [path, resource] of this.resources) {
          const topic = topicFor(path);
          if (!previous || previous[topic] !== versions[topic]) {
            resource.revision += 1;
            void this.refresh(path, resource);
          }
        }
      });
      source.addEventListener("unauthorized", () => {
        if (this.source !== source) return;
        this.close();
        requireLogin();
      });
      source.onerror = () => {
        if (this.source !== source) return;
        this.setConnection(false, source.readyState === EventSource.CLOSED);
        void getAuth().then(auth => {
          if (this.source === source && !auth.authenticated) { this.close(); requireLogin(); }
        }).catch(() => { /* Network failure retains the existing reconnect behavior. */ });
      };
    });
  }
}
