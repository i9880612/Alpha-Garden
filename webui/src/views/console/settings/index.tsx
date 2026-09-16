import { useEffect, useRef, useState, type ReactNode } from "react";
import { Alert, Button, Collapse, Form, Input, InputNumber, Select, Space, Switch, Tabs, Tag, Typography, message } from "antd";
import { QuestionCircleOutlined, ReloadOutlined, SaveOutlined, SettingOutlined, SlidersOutlined } from "@ant-design/icons";
import { useOutletContext } from "react-router-dom";
import type { ConsoleLocale } from "@/layouts/console/locale";
import { errorMessage, postJson, type BacktestPolicy, type Jobs, type RunDefaults, type Session, type Settings } from "@/api/console";
import { useApi } from "@/api/use-api";
import ModulePage from "../components/module-page";
import AlphaLoading from "@/components/alpha-loading";

const neutralizations = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"];
const options = (values: string[]) => values.map(value => ({ value, label: value }));
const categories: Record<string, string> = {
  Analyst: "分析师", Earnings: "财报", Fundamental: "基本面", Model: "模型", News: "新闻",
  Option: "期权", "Price Volume": "量价", Sentiment: "情绪", "Social Media": "社交媒体",
};

function Section({ title, description, children }: { title: string; description?: string; children: ReactNode }) {
  return <section className="ag-settings-section">
    <div className="ag-settings-section-heading"><h3>{title}</h3>{description && <Typography.Text type="secondary">{description}</Typography.Text>}</div>
    <div className="ag-settings-fields">{children}</div>
  </section>;
}

type SettingsValue = BacktestPolicy | RunDefaults;
function SettingsEditor({ section, initial, revision, disabled, busy, token, zh }: {
  section: "backtest" | "run"; initial: SettingsValue; revision: string;
  disabled: boolean; busy: boolean; token?: string; zh: boolean;
}) {
  const [form] = Form.useForm();
  const [messages, messageContext] = message.useMessage();
  const [baseline, setBaseline] = useState({ value: initial, revision });
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const savingRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const latest = useRef({ value: initial, revision });
  const seenRevision = useRef(revision);
  useEffect(() => {
    if (seenRevision.current === revision) return;
    seenRevision.current = revision;
    latest.current = { value: initial, revision };
    // Live notifications must not overwrite a draft or an in-flight save.
    if (!dirty && !savingRef.current) {
      form.resetFields();
      form.setFieldsValue(initial);
      setBaseline(latest.current);
    }
  }, [revision, initial, dirty, form]);
  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const t = (cn: string, en: string) => zh ? cn : en;
  const required = [{ required: true, message: t("请填写此项", "This field is required") }];
  const integer = [{ required: true, type: "integer" as const, min: 1, max: Number.MAX_SAFE_INTEGER, message: t("请输入大于 0 的整数", "Enter a positive integer") }];
  const select = (name: string | string[], label: string, values: string[]) => <Form.Item name={name} label={label} rules={required}><Select options={options(values)} /></Form.Item>;
  const number = (name: string, label: string) => <Form.Item name={name} label={label} rules={integer}><InputNumber min={1} precision={0} /></Form.Item>;
  const reset = () => {
    form.resetFields();
    form.setFieldsValue(latest.current.value);
    setBaseline(latest.current);
    setDirty(false);
    setError(null);
  };
  const save = async (value: SettingsValue) => {
    if (!token || disabled || busy || savingRef.current) return;
    savingRef.current = true;
    setSaving(true);
    setError(null);
    try {
      const saved = await postJson("/api/settings", { section, revision: baseline.revision, value }, token, crypto.randomUUID()) as { value: SettingsValue; revision: string };
      form.setFieldsValue(saved.value);
      latest.current = saved;
      setBaseline(saved);
      setDirty(false);
      messages.success(t("设置已保存，将用于新运行", "Settings saved for new runs"));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "console_operation_failed");
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  };
  return <Form form={form} layout="vertical" initialValues={initial} requiredMark={false}
    disabled={disabled || saving} onFinish={save} onValuesChange={() => { setDirty(true); setError(null); }} scrollToFirstError>
    {messageContext}
    <div className="ag-settings-toolbar">
      <div className="ag-settings-note"><QuestionCircleOutlined /><span>{t("保存后用于新运行；恢复已有运行仍沿用原计划。轮数和研究模式在运行中心选择。", "Saved defaults apply to new runs. Resuming keeps the original plan. Choose cycles and research mode in Runs.")}</span></div>
      <Space className="ag-settings-actions" wrap>
        {dirty && <Tag color="gold">{t("未保存", "Unsaved")}</Tag>}
        <Button icon={<ReloadOutlined />} disabled={!dirty || saving} onClick={reset}>{t("撤销修改", "Discard changes")}</Button>
        <Button type="primary" htmlType="submit" icon={<SaveOutlined />} loading={saving} disabled={disabled || busy || !dirty}>{t("保存设置", "Save settings")}</Button>
      </Space>
    </div>
    {busy && <Alert showIcon type="info" message={t("当前有操作正在运行，可以编辑，暂停或完成后再保存。", "An operation is running. You can edit now and save after it pauses or finishes.")} />}
    {error && <Alert showIcon type="error" message={errorMessage(error, zh)} />}
    {section === "backtest" ? <>
      <Section title={t("数据范围", "Data scope")} description={t("更改范围后，开始研究前需同步对应的数据目录。", "Sync the matching data catalog before starting research with a different scope.")}>
        <Form.Item name={["catalogContext", "instrumentType"]} label={t("资产类型", "Instrument type")} rules={required}><Input /></Form.Item>
        <Form.Item name={["catalogContext", "region"]} label={t("地区", "Region")} rules={required}><Input /></Form.Item>
        <Form.Item name={["catalogContext", "universe"]} label={t("股票池", "Universe")} rules={required}><Input /></Form.Item>
        <Form.Item name={["catalogContext", "delay"]} label={t("延迟（天）", "Delay (days)")} rules={[{ required: true, type: "integer", min: 0 }]}><InputNumber min={0} precision={0} /></Form.Item>
      </Section>
      <Section title={t("信号处理", "Signal processing")} description={t("中性化与截断由研究引擎按公式特征选择。", "The engine selects neutralization and truncation based on each formula.")}>
        <Form.Item name="decay" label={t("衰减（天）", "Decay (days)")} rules={[{ required: true, type: "integer", min: 0 }]}><InputNumber min={0} precision={0} /></Form.Item>
        {select(["neutralization", "default"], t("默认中性化", "Default neutralization"), neutralizations)}
        <Form.Item name={["truncation", "default"]} label={t("默认截断", "Default truncation")} rules={[{ required: true, type: "number", min: 0, max: 1 }]}><InputNumber min={0} max={1} step={0.01} /></Form.Item>
        <Form.Item name={["truncation", "tailRisk"]} label={t("尾部风险截断", "Tail-risk truncation")} dependencies={[["truncation", "default"]]} rules={[{ required: true, type: "number", min: 0, max: 1 }, ({ getFieldValue }) => ({ validator: (_, value) => value <= getFieldValue(["truncation", "default"]) ? Promise.resolve() : Promise.reject(new Error(t("不能大于默认截断", "Cannot exceed default truncation"))) })]}><InputNumber min={0} max={1} step={0.01} /></Form.Item>
        <div className="ag-settings-wide"><Collapse ghost items={[{ key: "neutralization", forceRender: true, label: t("分类中性化规则", "Neutralization by category"), children: <div className="ag-settings-rule-grid">
          {select(["neutralization", "rootGroupNeutralize"], t("公式已分组中性化时", "Already group-neutralized"), neutralizations)}
          {Object.keys((initial as BacktestPolicy).neutralization.byFieldCategory).map(category => <div key={category}>{select(["neutralization", "byFieldCategory", category], zh ? `${categories[category] || category} · ${category}` : category, neutralizations)}</div>)}
        </div> }]} /></div>
      </Section>
      <Section title={t("执行选项", "Execution options")}>
        {select("pasteurization", t("净化", "Pasteurization"), ["ON", "OFF"])}
        {select("nanHandling", t("缺失值处理", "NaN handling"), ["OFF", "ON"])}
        {select("unitHandling", t("单位处理", "Unit handling"), ["VERIFY", "OFF"])}
        {select("language", t("公式语言", "Formula language"), ["FASTEXPR"])}
        {select("maxTrade", t("最大交易限制", "Max trade"), ["OFF", "ON"])}
        {select("maxPosition", t("最大持仓限制", "Max position"), ["OFF", "ON"])}
        <Form.Item name="visualization" label={t("平台可视化", "Platform visualization")} valuePropName="checked"><Switch checkedChildren={t("开启", "On")} unCheckedChildren={t("关闭", "Off")} /></Form.Item>
      </Section>
    </> : <>
      <Section title={t("每轮预算", "Cycle budget")} description={t("控制每轮生成与实际回测的数量。", "Control generation and actual backtests per cycle.")}>
        {number("generationCount", t("生成数量", "Candidates generated"))}
        <Form.Item name="backtestCount" label={t("回测数量", "Backtests")} dependencies={["generationCount"]} rules={[...integer, ({ getFieldValue }) => ({ validator: (_, value) => value <= getFieldValue("generationCount") ? Promise.resolve() : Promise.reject(new Error(t("不能大于生成数量", "Cannot exceed generated candidates"))) })]}><InputNumber min={1} precision={0} /></Form.Item>
        {number("explorationSeedAttemptMultiplier", t("探索生成倍数", "Exploration attempt multiplier"))}
      </Section>
      <Section title={t("研究分配", "Research allocation")} description={t("探索、SC 治理、变异三项合计为 100%。反转验证包含在变异额度内。", "Exploration, SC repair and mutation must total 100%. Direction validation is part of the mutation allocation.")}>
        {([["explorationPercent", t("探索", "Exploration")], ["selfCorrelationPercent", t("SC 治理", "SC repair")], ["mutationPercent", t("变异", "Mutation")]]).map(([name, label]) => <Form.Item key={name} name={name} label={`${label} (%)`} dependencies={["explorationPercent", "selfCorrelationPercent", "mutationPercent"].filter(key => key !== name)} rules={[{ required: true, type: "integer", min: name === "explorationPercent" ? 1 : 0, max: 100 }, ({ getFieldValue }) => ({ validator: () => ["explorationPercent", "selfCorrelationPercent", "mutationPercent"].reduce((sum, key) => sum + (getFieldValue(key) ?? 0), 0) === 100 ? Promise.resolve() : Promise.reject(new Error(t("三项分配比例之和必须为 100%", "Allocation must total 100%"))) })]}><InputNumber min={name === "explorationPercent" ? 1 : 0} max={100} precision={0} /></Form.Item>)}
        <Form.Item name="directionValidationPercent" label={t("反转验证 (%)", "Direction validation (%)")} dependencies={["mutationPercent"]} rules={[{ required: true, type: "integer", min: 0, max: 100 }, ({ getFieldValue }) => ({ validator: (_, value) => value <= getFieldValue("mutationPercent") ? Promise.resolve() : Promise.reject(new Error(t("不能大于变异比例", "Cannot exceed mutation allocation"))) })]}><InputNumber min={0} max={100} precision={0} /></Form.Item>
      </Section>
      <Section title={t("运行保护", "Run protection")} description={t("达到边界后按研究计划暂停或停止。并发额度仍受平台限制。", "The plan pauses or stops at its limits. Platform concurrency limits still apply.")}>
        {number("maxInFlightBacktests", t("最大在途回测数", "Maximum in-flight backtests"))}
        {number("maxPendingSeconds", t("等待上限（秒）", "Maximum wait (seconds)"))}
        {number("maxConsecutiveFailures", t("连续失败上限", "Consecutive failure limit"))}
        {number("maxRequestFailures", t("连续请求故障上限", "Consecutive request failure limit"))}
      </Section>
    </>}
  </Form>;
}

export default function SettingsPage() {
  const { locale } = useOutletContext<{ locale: ConsoleLocale }>();
  const zh = locale === "zh-CN";
  const settings = useApi<Settings>("/api/settings");
  const session = useApi<Session>("/api/session");
  const jobs = useApi<Jobs>("/api/jobs");
  return <ModulePage>
    {settings.error && <Alert type="error" showIcon message={errorMessage(settings.error, zh)} />}
    {session.data?.read_only && <Alert type="info" showIcon message={zh ? "当前为只读模式，无法修改设置。" : "Read-only mode. Settings cannot be changed."} />}
    {!settings.data ? settings.loading && <AlphaLoading /> : !settings.data.revisions || !settings.data.run_config
      ? <Alert type="info" showIcon message={zh ? "新版运行设置需要重启 Web 服务后启用，请先等待当前运行结束或将其暂停。" : "Restart the web service to enable the updated settings. Wait for the current run to finish or pause it first."} />
      : <Tabs className="ag-settings-tabs" items={[
      { key: "backtest" as const, label: zh ? "回测配置" : "Backtest settings", icon: <SettingOutlined />, value: settings.data.policy },
      { key: "run" as const, label: zh ? "研究与运行" : "Research & runs", icon: <SlidersOutlined />, value: settings.data.run_config },
    ].map(tab => ({ key: tab.key, label: tab.label, icon: tab.icon, forceRender: true, children: <SettingsEditor section={tab.key} initial={tab.value} revision={settings.data!.revisions[tab.key]} zh={zh}
      token={session.data?.csrf_token} disabled={!session.data || session.data.read_only || !!session.error || !!settings.error || !jobs.data || !!jobs.error} busy={!!jobs.data?.busy} /> }))} />}
  </ModulePage>;
}
