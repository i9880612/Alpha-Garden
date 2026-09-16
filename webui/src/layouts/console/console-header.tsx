import { MenuFoldOutlined, MenuUnfoldOutlined } from "@ant-design/icons";
import { Layout } from "antd";
import type { CSSProperties, ComponentType, HTMLAttributes } from "react";

import ConsoleBreadcrumb from "./console-breadcrumb";
import ConsoleHeaderActions from "./console-header-actions";
import type { ConsoleLocale } from "./locale";
import type { ConsoleTheme } from "./theme";
import { consoleThemeTokens } from "./theme";

const { Header } = Layout;

interface ConsoleHeaderProps {
  collapsed: boolean;
  theme: ConsoleTheme;
  locale: ConsoleLocale;
  pathname: string;
  onToggle: () => void;
  onLocaleChange: (locale: ConsoleLocale) => void;
  onThemeChange: (theme: ConsoleTheme) => void;
}

const headerContentStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  height: 64,
};

const headerLeftStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  minWidth: 0,
};

const breadcrumbWrapStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  minWidth: 0,
};

const ConsoleHeader = ({ collapsed, theme, locale, pathname, onToggle, onLocaleChange, onThemeChange }: ConsoleHeaderProps) => {
  const TriggerIcon = (collapsed ? MenuUnfoldOutlined : MenuFoldOutlined) as unknown as ComponentType<HTMLAttributes<HTMLSpanElement>>;
  const tokens = consoleThemeTokens[theme];

  const triggerIconStyle: CSSProperties = {
    width: 64,
    height: 64,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 18,
    color: tokens.textColor,
    cursor: "pointer",
    transition: "background 0.2s, color 0.2s",
  };

  return (
    <Header
      className="site-layout-background ag-console-header"
      style={{
        height: 64,
        padding: 0,
        background: tokens.headerBg,
        borderBottom: `1px solid ${tokens.borderColor}`,
      }}
    >
      <div className="ag-console-header-inner" style={headerContentStyle}>
        <div className="ag-console-header-left" style={headerLeftStyle}>
          <button className="ag-icon-button ag-console-menu-trigger" onClick={onToggle} style={triggerIconStyle} aria-label={locale === "zh-CN" ? "切换导航" : "Toggle navigation"} aria-expanded={!collapsed} type="button"><TriggerIcon /></button>
          <div className="ag-console-breadcrumb-wrap" style={breadcrumbWrapStyle}>
            <ConsoleBreadcrumb theme={theme} locale={locale} pathname={pathname} />
          </div>
        </div>
        <ConsoleHeaderActions theme={theme} locale={locale} onLocaleChange={onLocaleChange} onThemeChange={onThemeChange} />
      </div>
    </Header>
  );
};

export default ConsoleHeader;
