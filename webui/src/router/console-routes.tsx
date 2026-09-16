import { DatabaseOutlined, HomeOutlined, PlayCircleOutlined, SendOutlined, SettingOutlined, AppstoreOutlined, FunctionOutlined, BarChartOutlined, BranchesOutlined, ExperimentOutlined, InboxOutlined, CheckCircleOutlined } from "@ant-design/icons";
import type { MenuProps } from "antd";
import type { RouteObject } from "react-router-dom";
import type { ConsoleLocale } from "@/layouts/console/locale";
import HomePage from "@/views/home";
import RunsPage from "@/views/console/runs";
import FormulasPage from "@/views/console/formulas";
import SubmissionPage from "@/views/console/submission-queue";
import SettingsPage from "@/views/console/settings";
import NotFoundPage from "@/views/console/not-found";

import CatalogPage from "@/views/console/catalog";
import AnalysisPage from "@/views/console/analysis";
import SeedsPage from "@/views/console/seeds";

const pages = [
  { path: "/", title: { "zh-CN": "工作台", "en-US": "Dashboard" }, icon: <HomeOutlined />, element: <HomePage /> },
  { path: "/runs", title: { "zh-CN": "运行中心", "en-US": "Runs" }, icon: <PlayCircleOutlined />, element: <RunsPage /> },
  { path: "/formulas", title: { "zh-CN": "公式库", "en-US": "Formulas" }, icon: <DatabaseOutlined />, element: <FormulasPage key="all" /> },
  { path: "/seeds", title: { "zh-CN": "种子池", "en-US": "Seed pool" }, icon: <BranchesOutlined />, element: <SeedsPage /> },
  { path: "/optimization", title: { "zh-CN": "专项优化公式", "en-US": "Optimization formulas" }, icon: <ExperimentOutlined />, element: <FormulasPage key="optimization" category="optimization" /> },
  { path: "/data", title: { "zh-CN": "数据目录", "en-US": "Data catalog" }, icon: <AppstoreOutlined />, element: <CatalogPage key="fields" kind="fields" /> },
  { path: "/operators", title: { "zh-CN": "算子库", "en-US": "Operators" }, icon: <FunctionOutlined />, element: <CatalogPage key="operators" kind="operators" /> },
  { path: "/analysis", title: { "zh-CN": "回测分析", "en-US": "Backtest analysis" }, icon: <BarChartOutlined />, element: <AnalysisPage /> },
  { path: "/archive", title: { "zh-CN": "合格归档", "en-US": "Qualified archive" }, icon: <InboxOutlined />, element: <FormulasPage key="archive" category="archive" /> },
  { path: "/submitted", title: { "zh-CN": "已提交公式", "en-US": "Submitted formulas" }, icon: <CheckCircleOutlined />, element: <FormulasPage key="submitted" category="submitted" /> },
  { path: "/submissions", title: { "zh-CN": "提交管理", "en-US": "Submissions" }, icon: <SendOutlined />, element: <SubmissionPage /> },
  { path: "/settings", title: { "zh-CN": "运行设置", "en-US": "Settings" }, icon: <SettingOutlined />, element: <SettingsPage /> },
];

export const consoleRouteObjects: RouteObject[] = [
  ...pages.map(({ path, element }): RouteObject => path === "/" ? { index: true, element } : { path: path.slice(1), element }),
  { path: "*", element: <NotFoundPage /> },
];

export const createConsoleMenuItems = (locale: ConsoleLocale): MenuProps["items"] =>
  [
    { key: "overview", label: locale === "zh-CN" ? "概览" : "Overview", type: "group", children: ["/"] },
    { key: "research", label: locale === "zh-CN" ? "研究" : "Research", type: "group", children: ["/runs", "/seeds", "/optimization", "/formulas", "/data", "/operators"] },
    { key: "results", label: locale === "zh-CN" ? "成果" : "Results", type: "group", children: ["/analysis", "/archive", "/submissions", "/submitted"] },
    { key: "system", label: locale === "zh-CN" ? "系统" : "System", type: "group", children: ["/settings"] },
  ].map(group => ({ ...group, type: "group" as const, children: group.children.map(key => {
    const page = pages.find(item => item.path === key)!;
    return { key, label: page.title[locale], icon: page.icon };
  }) }));

const findPage = (pathname: string) => pages.find(({ path }) => path === (pathname.replace(/\/$/, "") || "/"));
export const getSelectedConsoleRouteKey = (pathname: string) => findPage(pathname)?.path ?? "";
export const createConsoleBreadcrumbItems = (locale: ConsoleLocale, pathname: string) => [
  { title: findPage(pathname)?.title[locale] ?? (locale === "zh-CN" ? "页面不存在" : "Page not found") },
];
