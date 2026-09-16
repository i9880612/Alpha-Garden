import { GlobalOutlined, LogoutOutlined, SettingOutlined } from "@ant-design/icons";
import { Avatar, Dropdown, Typography, message } from "antd";
import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import avatarBlackUrl from "@/assets/image/avatar_black.png";
import avatarWhiteUrl from "@/assets/image/avatar_white.png";
import { consoleMessages, type ConsoleLocale } from "./locale";
import { MoonIcon, SunIcon } from "./theme-icon";
import { consoleThemeTokens, type ConsoleTheme } from "./theme";
import { useConsole } from "@/api/use-console";
import { postJson } from "@/api/console";

interface ConsoleHeaderActionsProps {
  theme: ConsoleTheme;
  locale: ConsoleLocale;
  onLocaleChange: (locale: ConsoleLocale) => void;
  onThemeChange: (theme: ConsoleTheme) => void;
}

const ConsoleHeaderActions = ({ theme, locale, onLocaleChange, onThemeChange }: ConsoleHeaderActionsProps) => {
  const { header } = consoleMessages[locale];
  const tokens = consoleThemeTokens[theme];
  const { events, session } = useConsole();
  const [messages, messageContext] = message.useMessage();
  const [loggingOut, setLoggingOut] = useState(false);
  const logoutPending = useRef(false);
  const logout = async () => {
    if (!session.data || logoutPending.current) return;
    logoutPending.current = true; setLoggingOut(true);
    try {
      await postJson("/api/logout", {}, session.data.csrf_token, crypto.randomUUID());
      window.location.replace("/login");
    } catch { messages.error(locale === "zh-CN" ? "退出失败，请稍后重试。" : "Unable to sign out. Please retry."); }
    finally { logoutPending.current = false; setLoggingOut(false); }
  };
  const connectionLabel = events.connected ? (locale === "zh-CN" ? "实时连接正常" : "Live updates connected")
    : events.failed ? (locale === "zh-CN" ? "实时连接不可用，请检查服务后刷新页面" : "Live updates unavailable; check the service and reload")
    : events.ready ? (locale === "zh-CN" ? "连接中断，正在重连 · 数据可能滞后" : "Reconnecting · data may be stale")
    : (locale === "zh-CN" ? "正在连接实时更新" : "Connecting live updates");
  return (
    <div className="ag-console-header-actions" style={{ display: "flex", alignItems: "center", gap: 12, paddingRight: 24, color: tokens.textColor }}>
      {messageContext}
      <Dropdown menu={{ items: [
        { key: "zh-CN", label: "简体中文", onClick: () => onLocaleChange("zh-CN") },
        { key: "en-US", label: "English", onClick: () => onLocaleChange("en-US") },
      ], selectedKeys: [locale] }} placement="bottomRight" trigger={["click"]}>
        <button className="ag-icon-button ag-header-icon-action" type="button" aria-label={header.language}><GlobalOutlined /></button>
      </Dropdown>
      <button className="ag-icon-button ag-header-icon-action" type="button" aria-label={theme === "dark" ? header.lightTheme : header.darkTheme} onClick={() => onThemeChange(theme === "dark" ? "light" : "dark")}>
        {theme === "dark" ? <SunIcon /> : <MoonIcon />}
      </button>
      <Link className="ag-icon-button ag-header-settings-action" to="/settings" aria-label={header.systemSettings}><SettingOutlined /></Link>
      <Dropdown menu={{ items: [{ key: "logout", icon: <LogoutOutlined />, label: locale === "zh-CN" ? "退出登录" : "Sign out", disabled: loggingOut || !session.data, onClick: logout }] }} trigger={["click"]} placement="bottomRight">
      <button type="button" aria-label={locale === "zh-CN" ? "账号菜单" : "Account menu"} className="ag-header-account-action" style={{ display: "flex", alignItems: "center", gap: 8, background: "transparent", border: 0, padding: 0, cursor: "pointer", textAlign: "left" }}>
        <Avatar size={30} src={theme === "dark" ? avatarWhiteUrl : avatarBlackUrl} />
        <span className="ag-header-account-text" style={{ display: "flex", flexDirection: "column", lineHeight: 1.4 }}>
          <Typography.Text style={{ fontSize: 13 }}>admin</Typography.Text>
          <Typography.Text type="secondary" style={{ fontSize: 12 }} role="status" title={connectionLabel}>{events.connected ? header.role : events.failed ? (locale === "zh-CN" ? "连接失败，请刷新" : "Connection failed · Reload") : (locale === "zh-CN" ? "正在连接…" : "Connecting…")}</Typography.Text>
        </span>
      </button>
      </Dropdown>
    </div>
  );
};
export default ConsoleHeaderActions;
