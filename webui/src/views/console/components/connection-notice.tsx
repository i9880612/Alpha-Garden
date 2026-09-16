import { Alert, Empty } from "antd";
import { useOutletContext } from "react-router-dom";
import { useConsole } from "@/api/use-console";
import { errorMessage } from "@/api/console";
import type { ConsoleLocale } from "@/layouts/console/locale";

export const ConnectionNotice = () => {
  const { locale } = useOutletContext<{ locale: ConsoleLocale }>();
  const { session, actionError } = useConsole();
  const zh = locale === "zh-CN";
  const error = actionError || session.error || session.data?.error;
  if (error) return <Alert type="error" showIcon message={errorMessage(error, zh)} />;
  if (session.data?.read_only)
    return (
      <Alert
        type="info"
        showIcon
        message={
          zh
            ? "当前服务为只读模式，可查看真实数据；运行和提交操作已禁用。"
            : "Read-only service: live local data is available; run and submission actions are disabled."
        }
      />
    );
  return null;
};
export const UnavailableData = () => {
  const { locale } = useOutletContext<{ locale: ConsoleLocale }>();
  return (
    <Empty
      image={Empty.PRESENTED_IMAGE_SIMPLE}
      description={locale === "zh-CN" ? "暂无记录" : "No records"}
    />
  );
};
