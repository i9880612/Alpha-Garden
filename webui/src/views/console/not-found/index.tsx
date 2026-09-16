import { Button, Result } from "antd";
import { Link, useOutletContext } from "react-router-dom";
import type { ConsoleLocale } from "@/layouts/console/locale";

export default function NotFoundPage() {
  const { locale } = useOutletContext<{ locale: ConsoleLocale }>();
  const zh = locale === "zh-CN";
  return <Result status="404" title="404" subTitle={zh ? "页面不存在" : "Page not found"} extra={<Link to="/"><Button type="primary">{zh ? "返回工作台" : "Back to dashboard"}</Button></Link>} />;
}
