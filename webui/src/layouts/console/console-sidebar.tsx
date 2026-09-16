import type { MenuProps } from "antd";
import { Layout, Menu } from "antd";
import { useMemo } from "react";

import alphaGardenLogoUrl from "@/assets/logo/logo-full.svg";
import alphaLogoUrl from "@/assets/logo/logo-symbol.svg";
import { createConsoleMenuItems } from "@/router/console-routes";

import type { ConsoleLocale } from "./locale";
import type { ConsoleTheme } from "./theme";
import { consoleThemeTokens } from "./theme";

const { Sider } = Layout;

interface ConsoleSidebarProps {
  theme: ConsoleTheme;
  collapsed: boolean;
  locale: ConsoleLocale;
  selectedKey: string;
  onMenuClick: MenuProps["onClick"];
}

const ConsoleSidebar = ({ theme, collapsed, locale, selectedKey, onMenuClick }: ConsoleSidebarProps) => {
  const menuItems = useMemo(() => createConsoleMenuItems(locale), [locale]);
  const tokens = consoleThemeTokens[theme];

  return (
    <Sider
      className={`ag-console-sider ${collapsed ? "is-collapsed" : "is-expanded"}`}
      trigger={null}
      collapsible
      collapsed={collapsed}
      collapsedWidth={80}
      theme={theme}
      width={256}
      style={{
        borderRight: `1px solid ${tokens.borderColor}`,
        background: tokens.siderBg,
      }}
    >
      <div
        className="ag-console-logo"
        style={{
          background: tokens.logoTileBg,
        }}
      >
        <img className="ag-console-logo-full" src={alphaGardenLogoUrl} alt="Alpha Garden" style={{ filter: tokens.logoFilter }} />
        <img className="ag-console-logo-symbol" src={alphaLogoUrl} alt="Alpha" style={{ filter: tokens.logoFilter }} />
      </div>

      <Menu
        style={{ width: "100%", background: "transparent", flex: 1, minHeight: 0, overflowY: "auto", overflowX: "hidden" }}
        selectedKeys={[selectedKey]}
        mode="inline"
        inlineCollapsed={collapsed}
        theme={theme}
        items={menuItems}
        onClick={onMenuClick}
      />
    </Sider>
  );
};

export default ConsoleSidebar;
