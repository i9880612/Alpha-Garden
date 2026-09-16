import { createContext, useEffect, useState, useSyncExternalStore } from "react";
import { ConsoleEventStream } from "./event-stream";

export const StreamContext = createContext<ConsoleEventStream | null>(null);

export function useConsoleEvents() {
  const [stream] = useState(() => new ConsoleEventStream());
  const state = useSyncExternalStore(stream.subscribeConnection, stream.getConnection);
  useEffect(() => {
    stream.start();
    return () => stream.close();
  }, [stream]);
  return { stream, ...state };
}
