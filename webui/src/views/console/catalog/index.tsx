import { ClockCircleOutlined, SettingOutlined } from "@ant-design/icons";
import {
  Alert,
  Button,
  Descriptions,
  Drawer,
  Empty,
  Input,
  Select,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
  theme,
} from "antd";
import { useState } from "react";
import { useOutletContext, useSearchParams } from "react-router-dom";
import type { ConsoleLocale } from "@/layouts/console/locale";
import { useApi } from "@/api/use-api";
import { errorMessage } from "@/api/console";
import type { Catalog, DataField, Operator } from "@/api/research";
import { metric, timestamp } from "@/api/presentation";
import ModulePage from "../components/module-page";

export default function CatalogPage({ kind }: { kind: "fields" | "operators" }) {
  const { locale } = useOutletContext<{ locale: ConsoleLocale }>();
  const { token } = theme.useToken();
  const zh = locale === "zh-CN",
    fields = kind === "fields";
  const [params, setParams] = useSearchParams();
  const [page, setPage] = useState(1),
    [pageSize, setPageSize] = useState(10);
  const [selected, setSelected] = useState<DataField | Operator | null>(null);
  const search = params.get("search") || "",
    category = params.get("category") || "",
    dataset = params.get("dataset") || "";
  const query = new URLSearchParams({
    search,
    category,
    dataset,
    page: String(page),
    page_size: String(pageSize),
  });
  const { data, error, loading } = useApi<Catalog<DataField | Operator>>(
    `/api/catalog/${kind}?${query}`,
  );
  const filter = (key: string, value?: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next);
    setPage(1);
    setSelected(null);
  };
  const pagination = {
    current: page,
    pageSize,
    total: data?.total || 0,
    showSizeChanger: true,
    pageSizeOptions: [10, 20, 50, 100],
    showTotal: (total: number) => (zh ? `共 ${total} 条` : `${total} records`),
    onChange: (next: number, size: number) => {
      setPage(size === pageSize ? next : 1);
      setPageSize(size);
    },
  };
  const empty = (
    <Empty
      description={
        data?.available === false
          ? zh
            ? "当前账号暂无已同步目录"
            : "No synced catalog for this account"
          : zh
            ? "没有匹配的记录"
            : "No matching records"
      }
    />
  );
  return (
    <ModulePage>
      <Space direction="vertical" size={12} style={{ width: "100%" }}>
          <Space wrap>
            <Input.Search
              key={search}
              defaultValue={search}
              allowClear
              maxLength={128}
              style={{ width: 280 }}
              aria-label={zh ? "搜索目录" : "Search catalog"}
              placeholder={
                fields
                  ? zh
                    ? "字段名称或描述"
                    : "Field name or description"
                  : zh
                    ? "算子名称或描述"
                    : "Operator name or description"
              }
              onSearch={(value) => filter("search", value)}
            />
            <Select
              allowClear
              showSearch
              optionFilterProp="label"
              style={{ width: 210 }}
              aria-label={zh ? "分类" : "Category"}
              placeholder={zh ? "全部分类" : "All categories"}
              value={category || undefined}
              options={data?.categories.map((value) => ({ label: value, value }))}
              onChange={(value) => filter("category", value)}
            />
            {fields && (
              <Select
                allowClear
                showSearch
                optionFilterProp="label"
                style={{ width: 270 }}
                aria-label={zh ? "数据集" : "Dataset"}
                placeholder={zh ? "全部数据集" : "All datasets"}
                value={dataset || undefined}
                options={data?.datasets.map((item) => ({
                  value: item.dataset_id,
                  label: item.dataset_name
                    ? `${item.dataset_id} · ${item.dataset_name}`
                    : item.dataset_id,
                }))}
                onChange={(value) => filter("dataset", value)}
              />
            )}
          </Space>
          {data?.context && (
            <Space wrap size={[16, 8]}>
              <Space size={8}>
                <SettingOutlined aria-label={zh ? "配置" : "Configuration"} style={{ color: token.colorPrimary, fontSize: 16 }} />
                <Space size={0} wrap>
                  <Tag>{data.context.instrument_type}</Tag>
                  <Tag>{data.context.region}</Tag>
                  <Tag>{data.context.universe}</Tag>
                  <Tag>Delay {data.context.delay}</Tag>
                </Space>
              </Space>
              <Space size={8}>
                <ClockCircleOutlined aria-hidden style={{ color: token.colorPrimary, fontSize: 16 }} />
                <Typography.Text type="secondary">
                  {zh ? "同步于" : "Synced"} {timestamp(data.synced_at)}
                </Typography.Text>
              </Space>
            </Space>
          )}
      </Space>
      {error ? (
        <Alert type="error" showIcon message={errorMessage(error, zh)} />
      ) : fields ? (
        <Table<DataField>
          bordered
          className="ag-catalog-table"
          tableLayout="fixed"
          rowKey="field_id"
          loading={loading}
          dataSource={(data?.items as DataField[]) || []}
          pagination={pagination}
          locale={{ emptyText: empty }}
          columns={[
            {
              title: zh ? "字段" : "Field",
              dataIndex: "field_id",
              width: "26%",
              ellipsis: true,
              render: (value, row) => (
                <Button type="link" className="ag-table-action" onClick={() => setSelected(row)}>
                  {value}
                </Button>
              ),
            },
            {
              title: zh ? "数据集" : "Dataset",
              dataIndex: "dataset_id",
              width: "11%",
              ellipsis: true,
              render: (value) => value || "—",
            },
            {
              title: zh ? "分类" : "Category",
              dataIndex: "category",
              width: "11%",
              ellipsis: true,
              render: (value) => value || "—",
            },
            {
              title: zh ? "类型" : "Type",
              dataIndex: "field_type",
              width: "10%",
              ellipsis: true,
              render: (value) => <Tag color="blue" style={{ marginInlineEnd: 0 }}>{value || "—"}</Tag>,
            },
            {
              title: zh ? "覆盖率" : "Coverage",
              dataIndex: "coverage",
              width: "9%",
              ellipsis: true,
              render: (value) => metric(value, true),
            },
            {
              title: zh ? "描述" : "Description",
              dataIndex: "description",
              ellipsis: { showTitle: false },
              render: (value: string) => value ? (
                <Tooltip title={value} trigger={["hover", "focus"]} styles={{
                  root: { maxWidth: "min(640px, calc(100vw - 32px))" },
                  body: { whiteSpace: "pre-wrap", overflowWrap: "anywhere" },
                }}>
                  <span className="ag-catalog-description" tabIndex={0}>{value}</span>
                </Tooltip>
              ) : "—",
            },
          ]}
        />
      ) : (
        <Table<Operator>
          bordered
          className="ag-catalog-table"
          tableLayout="fixed"
          rowKey="operator_name"
          loading={loading}
          dataSource={(data?.items as Operator[]) || []}
          pagination={pagination}
          locale={{ emptyText: empty }}
          columns={[
            {
              title: zh ? "算子" : "Operator",
              dataIndex: "operator_name",
              width: "17%",
              ellipsis: true,
              render: (value, row) => (
                <Button type="link" className="ag-table-action" onClick={() => setSelected(row)}>
                  {value}
                </Button>
              ),
            },
            {
              title: zh ? "分类" : "Category",
              dataIndex: "category",
              width: "12%",
              ellipsis: true,
              render: (value) => value || "—",
            },
            {
              title: zh ? "定义" : "Definition",
              dataIndex: "definition",
              width: "26%",
              ellipsis: { showTitle: false },
              render: (value: string) => value ? (
                <Tooltip title={value} trigger={["hover", "focus"]} styles={{
                  root: { maxWidth: "min(640px, calc(100vw - 32px))" },
                  body: { whiteSpace: "pre-wrap", overflowWrap: "anywhere" },
                }}>
                  <code className="ag-catalog-description" tabIndex={0}>{value}</code>
                </Tooltip>
              ) : "—",
            },
            {
              title: zh ? "适用范围" : "Scope",
              dataIndex: "scope",
              width: "14%",
              ellipsis: { showTitle: false },
              render: (value: string[]) =>
                value.length
                  ? <Tooltip title={value.join(", ")} trigger={["hover", "focus"]}>
                      <span className="ag-catalog-description" tabIndex={0}>
                        {value.map((v) => <Tag key={v} color="blue">{v}</Tag>)}
                      </span>
                    </Tooltip>
                  : "—",
            },
            {
              title: zh ? "描述" : "Description",
              dataIndex: "description",
              ellipsis: { showTitle: false },
              render: (value: string) => value ? (
                <Tooltip title={value} trigger={["hover", "focus"]} styles={{
                  root: { maxWidth: "min(640px, calc(100vw - 32px))" },
                  body: { whiteSpace: "pre-wrap", overflowWrap: "anywhere" },
                }}>
                  <span className="ag-catalog-description" tabIndex={0}>{value}</span>
                </Tooltip>
              ) : "—",
            },
          ]}
        />
      )}
      <Drawer
        open={!!selected}
        onClose={() => setSelected(null)}
        width="min(760px, 100vw)"
        title={selected && ("field_id" in selected ? selected.field_id : selected.operator_name)}
      >
        {selected && (
          <Space className="ag-catalog-details" direction="vertical" size={24} style={{ width: "100%" }}>
            <Typography.Paragraph>
              {selected.description || (zh ? "目录未提供描述" : "No description in the catalog")}
            </Typography.Paragraph>
            {"field_id" in selected ? (
              <Descriptions
                bordered
                column={1}
                items={[
                  {
                    key: "id",
                    label: zh ? "字段" : "Field",
                    children: <Typography.Text copyable>{selected.field_id}</Typography.Text>,
                  },
                  {
                    key: "dataset",
                    label: zh ? "数据集" : "Dataset",
                    children: `${selected.dataset_id || "—"} ${selected.dataset_name || ""}`,
                  },
                  {
                    key: "category",
                    label: zh ? "分类 / 子分类" : "Category / subcategory",
                    children: `${selected.category || "—"} / ${selected.subcategory || "—"}`,
                  },
                  {
                    key: "type",
                    label: zh ? "类型" : "Type",
                    children: selected.field_type || "—",
                  },
                  {
                    key: "coverage",
                    label: zh ? "覆盖率" : "Coverage",
                    children: metric(selected.coverage, true),
                  },
                ]}
              />
            ) : (
              <>
                <Typography.Paragraph copyable={!!selected.definition} code>
                  {selected.definition || "—"}
                </Typography.Paragraph>
                <Descriptions
                  bordered
                  column={1}
                  items={[
                    {
                      key: "category",
                      label: zh ? "分类" : "Category",
                      children: selected.category || "—",
                    },
                    {
                      key: "scope",
                      label: zh ? "适用范围" : "Scope",
                      children: selected.scope.join(", ") || "—",
                    },
                    { key: "level", label: zh ? "级别" : "Level", children: selected.level || "—" },
                    {
                      key: "roles",
                      label: zh ? "研究用途" : "Research roles",
                      children: selected.roles.join(", ") || "—",
                    },
                  ]}
                />
                <Typography.Text strong>{zh ? "参数" : "Parameters"}</Typography.Text>
                {selected.parameters.length ? (
                  selected.parameters.map((parameter, i) => (
                    <Descriptions
                      key={i}
                      size="small"
                      column={1}
                      bordered
                      items={Object.entries(parameter).map(([key, value]) => ({
                        key,
                        label: key,
                        children: typeof value === "string" ? value : JSON.stringify(value),
                      }))}
                    />
                  ))
                ) : (
                  <Empty
                    description={zh ? "目录未提供参数说明" : "No parameter details in the catalog"}
                  />
                )}
              </>
            )}
          </Space>
        )}
      </Drawer>
    </ModulePage>
  );
}
