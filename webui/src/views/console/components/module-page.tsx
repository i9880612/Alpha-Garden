import type { ReactNode } from "react";
import { ConnectionNotice } from "./connection-notice";

interface ModulePageProps {
  children: ReactNode;
}
const ModulePage = ({ children }: ModulePageProps) => (
  <section className="ag-module-page">
    <ConnectionNotice />
    {children}
  </section>
);
export default ModulePage;
