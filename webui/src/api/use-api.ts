import { useCallback, useContext, useSyncExternalStore } from "react";
import type { ConsoleEventStream } from "./event-stream";
import { StreamContext } from "./use-events";

export function useApi<T>(path: string | null, connection?: ConsoleEventStream) {
  const context = useContext(StreamContext);
  const stream = connection ?? context;
  if (!stream) throw new Error("Console stream provider missing");
  const subscribe = useCallback((listener: () => void) => stream.subscribe(path, listener), [stream, path]);
  const snapshot = useCallback(() => stream.getResource(path), [stream, path]);
  const state = useSyncExternalStore(subscribe, snapshot);
  return { ...state, data: state.data as T | null };
}
