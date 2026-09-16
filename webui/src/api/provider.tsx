import type { ReactNode } from "react";
import { Context, useConsoleConnection } from "./use-console";
import { StreamContext } from "./use-events";
export function ConsoleProvider({ children }: { children: ReactNode }) {
  const value = useConsoleConnection();
  return <Context.Provider value={value}><StreamContext.Provider value={value.events.stream}>{children}</StreamContext.Provider></Context.Provider>;
}
