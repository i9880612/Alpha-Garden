export type ConsoleTheme = "light" | "dark";

export const defaultConsoleTheme: ConsoleTheme = "dark";

export const isConsoleTheme = (value: string | null): value is ConsoleTheme => {
  return value === "light" || value === "dark";
};

export const consoleThemeTokens = {
  light: {
    pageBg: "#f5f5f5",
    surfaceBg: "#fff",
    elevatedBg: "#fff",
    siderBg: "#fff",
    headerBg: "#fff",
    contentBg: "#fff",
    borderColor: "#f0f0f0",
    textColor: "rgba(0, 0, 0, 0.88)",
    secondaryTextColor: "rgba(0, 0, 0, 0.45)",
    logoFilter: "none",
    logoTileBg: "transparent",
    triggerHoverBg: "rgba(0, 0, 0, 0.04)",
  },
  dark: {
    pageBg: "rgb(10, 10, 10)",
    surfaceBg: "rgb(10, 10, 10)",
    elevatedBg: "rgb(10, 10, 10)",
    siderBg: "rgb(10, 10, 10)",
    headerBg: "rgb(10, 10, 10)",
    contentBg: "rgb(10, 10, 10)",
    borderColor: "rgba(255, 255, 255, 0.08)",
    textColor: "rgba(255, 255, 255, 0.88)",
    secondaryTextColor: "rgba(255, 255, 255, 0.48)",
    logoFilter: "brightness(0) invert(1)",
    logoTileBg: "transparent",
    triggerHoverBg: "rgba(255, 255, 255, 0.08)",
  },
} as const;
