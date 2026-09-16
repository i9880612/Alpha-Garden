import { createContext, useContext, useRef, useState } from "react";
import { postJson, type JobRequest, type Jobs, type Session } from "./console";
import { useApi } from "./use-api";
import { useConsoleEvents } from "./use-events";

export function useConsoleConnection() {
  const events = useConsoleEvents();
  const session = useApi<Session>("/api/session", events.stream);
  const jobs = useApi<Jobs>("/api/jobs", events.stream);
  const [pending, setPending] = useState(false);
  const [acceptedJob, setAcceptedJob] = useState<{ id: string; sessionToken: string } | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const pendingRequest = useRef<{ body: string; id: string } | null>(null);
  const requestInFlight = useRef(false);
  const awaitingJob = acceptedJob !== null && acceptedJob.sessionToken === session.data?.csrf_token
    && !jobs.data?.items.some(job => job.id === acceptedJob.id);
  const act = async (path: string, request: unknown) => {
    if (!session.data || requestInFlight.current || awaitingJob) return;
    if (path === "/api/jobs" && jobs.data?.busy) return;
    requestInFlight.current = true;
    setPending(true);
    setActionError(null);
    const body = JSON.stringify({ path, request });
    // A request observed in the job list has a known outcome; a later click is a new operation.
    const knownRequest = jobs.data?.items.some(job => job.request_id === pendingRequest.current?.id);
    if (pendingRequest.current?.body !== body || knownRequest) pendingRequest.current = { body, id: crypto.randomUUID() };
    try {
      const result = await postJson(path, request, session.data.csrf_token, pendingRequest.current.id);
      pendingRequest.current = null;
      if (path === "/api/jobs") {
        setAcceptedJob({ id: result.job_id, sessionToken: session.data.csrf_token });
        return result.job_id as string;
      }
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "console_operation_failed");
    } finally {
      requestInFlight.current = false;
      setPending(false);
    }
  };
  return { session, jobs, events, pending: pending || awaitingJob, actionError,
    canOperate: events.connected && !!session.data?.ready && !session.data.read_only && !jobs.error && !!jobs.data && !jobs.data.busy && !pending && !awaitingJob,
    start: (request: JobRequest) => act("/api/jobs", request), stop: (id: string) => act("/api/jobs/stop", { job_id: id }) };
}

export const Context = createContext<ReturnType<typeof useConsoleConnection> | null>(null);
export const useConsole = () => {
  const value = useContext(Context);
  if (!value) throw new Error("Console provider missing");
  return value;
};
