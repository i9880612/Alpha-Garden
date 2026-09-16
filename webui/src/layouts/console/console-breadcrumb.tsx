import { Breadcrumb } from "antd";

import { createConsoleBreadcrumbItems } from "@/router/console-routes";

import type { ConsoleLocale } from "./locale";
import type { ConsoleTheme } from "./theme";

interface ConsoleBreadcrumbProps {
  theme: ConsoleTheme;
  locale: ConsoleLocale;
  pathname: string;
}

const ConsoleBreadcrumb = ({ theme, locale, pathname }: ConsoleBreadcrumbProps) => {
  const items = createConsoleBreadcrumbItems(locale, pathname).map((item, index, list) => ({
    ...item,
    title: <span className={index === list.length - 1 ? "ag-breadcrumb-current" : "ag-breadcrumb-link"}>{item.title}</span>,
  }));

  return <Breadcrumb className={`ag-console-breadcrumb ag-console-breadcrumb--${theme}`} items={items} />;
};

export default ConsoleBreadcrumb;
