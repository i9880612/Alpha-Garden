import type { MenuProps } from "antd";
import { ConfigProvider, Layout, theme as antdTheme } from "antd";
import enUS from "antd/es/locale/en_US";
import zhCN from "antd/es/locale/zh_CN";
import { useEffect, useState } from "react";
import { flushSync } from "react-dom";
import { Outlet, useLocation, useNavigate } from "react-router-dom";

import { getSelectedConsoleRouteKey } from "@/router/console-routes";
import { ConsoleProvider } from "@/api/provider";
import AlphaLoading from "@/components/alpha-loading";

import ConsoleHeader from "./console/console-header";
import ConsoleSidebar from "./console/console-sidebar";
import type { ConsoleLocale } from "./console/locale";
import { defaultConsoleLocale, isConsoleLocale } from "./console/locale";
import type { ConsoleTheme } from "./console/theme";
import { consoleThemeTokens } from "./console/theme";

const { Content } = Layout;
const localeStorageKey = "alphagarden-locale";
const themeStorageKey = "alphagarden-console-theme";
const mobileMediaQuery = "(max-width: 768px)";

const isMobileViewport = () => typeof window !== "undefined" && window.matchMedia(mobileMediaQuery).matches;

const getInitialLocale = (): ConsoleLocale => {
  const storedLocale = localStorage.getItem(localeStorageKey);
  return isConsoleLocale(storedLocale) ? storedLocale : defaultConsoleLocale;
};

const getInitialTheme = (): ConsoleTheme => {
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
};

const ConsoleLayout = () => {
  const [isMobile, setIsMobile] = useState(isMobileViewport);
  const [collapsed, setCollapsed] = useState(() => isMobileViewport());
  const [locale, setLocale] = useState<ConsoleLocale>(getInitialLocale);
  const [consoleTheme, setConsoleTheme] = useState<ConsoleTheme>(getInitialTheme);
  const navigate = useNavigate();
  const location = useLocation();
  const themeTokens = consoleThemeTokens[consoleTheme];

  const selectedKey = getSelectedConsoleRouteKey(location.pathname);

  useEffect(() => {
    document.documentElement.lang = locale;
    document.documentElement.classList.toggle("dark", consoleTheme === "dark");
  }, [locale, consoleTheme]);

  useEffect(() => {
    const media = window.matchMedia(mobileMediaQuery);
    const updateViewport = () => {
      const nextIsMobile = media.matches;
      setIsMobile(nextIsMobile);
      if (nextIsMobile) {
        setCollapsed(true);
      }
    };

    updateViewport();
    media.addEventListener("change", updateViewport);
    return () => media.removeEventListener("change", updateViewport);
  }, []);

  const handleMenuClick: MenuProps["onClick"] = ({ key }) => {
    const nextPath = String(key);

    if (nextPath.startsWith("/")) {
      navigate(nextPath);
      if (isMobile) {
        setCollapsed(true);
      }
    }
  };

  const handleLocaleChange = (nextLocale: ConsoleLocale) => {
    setLocale(nextLocale);
    localStorage.setItem(localeStorageKey, nextLocale);
  };

  const handleThemeChange = (nextTheme: ConsoleTheme) => {
    if (nextTheme === consoleTheme) return;

    const applyThemeChange = () => {
      setConsoleTheme(nextTheme);
      localStorage.setItem(themeStorageKey, nextTheme);
    };

    if ("startViewTransition" in document) {
      document.documentElement.classList.add("ag-theme-revealing");
      const transition = document.startViewTransition(() => {
        flushSync(applyThemeChange);
      });

      transition.finished.finally(() => {
        document.documentElement.classList.remove("ag-theme-revealing");
      });
      return;
    }

    applyThemeChange();
  };

  return (
    <ConsoleProvider>
    <ConfigProvider
      locale={locale === "zh-CN" ? zhCN : enUS}
      spin={{ indicator: <AlphaLoading /> }}
      theme={{
        algorithm: consoleTheme === "dark" ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        token: {
          colorBgLayout: themeTokens.pageBg,
          colorBgContainer: themeTokens.surfaceBg,
          colorBorderSecondary: themeTokens.borderColor,
        },
        components: {
          Table: { cellPaddingInline: 16, cellPaddingInlineMD: 16, cellPaddingInlineSM: 16 },
          Pagination: { itemBg: "transparent", itemActiveBg: "transparent", itemLinkBg: "transparent", itemInputBg: "transparent" },
          Menu: consoleTheme === "dark" ? {
            darkItemBg: "rgb(10, 10, 10)",
            darkSubMenuItemBg: "rgb(10, 10, 10)",
          } : {},
        },
      }}
    >
      <Layout className={`ag-console-shell ag-console-shell--${consoleTheme} ${location.pathname === "/" ? "ag-console-shell--home" : ""} ${isMobile ? "ag-console-shell--mobile" : ""}`} style={{ minHeight: "100vh", background: themeTokens.pageBg }}>
        {isMobile && !collapsed ? <button className="ag-mobile-sider-mask is-open" aria-label={locale === "zh-CN" ? "关闭导航" : "Close navigation"} type="button" onClick={() => setCollapsed(true)} /> : null}
        <ConsoleSidebar theme={consoleTheme} collapsed={collapsed} locale={locale} selectedKey={selectedKey} onMenuClick={handleMenuClick} />

        <Layout className="site-layout" style={{ background: themeTokens.pageBg }}>
          <ConsoleHeader collapsed={collapsed} theme={consoleTheme} locale={locale} pathname={location.pathname} onToggle={() => setCollapsed((value) => !value)} onLocaleChange={handleLocaleChange} onThemeChange={handleThemeChange} />

          <Content
            className="site-layout-background ag-console-content"
            style={{
              margin: isMobile ? "16px 12px 24px" : "20px 20px 20px",
              padding: 0,
              minHeight: 280,
              background: themeTokens.pageBg,
            }}
          >
            <Outlet context={{ theme: consoleTheme, locale }} />
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
    </ConsoleProvider>
  );
};

export default ConsoleLayout;
