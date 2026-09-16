import { useEffect, useRef } from "react";
import { BarChart, LineChart, PieChart, ScatterChart } from "echarts/charts";
import { GridComponent, TooltipComponent, MarkLineComponent } from "echarts/components";
import * as echarts from "echarts/core";
import type { EChartsCoreOption } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
echarts.use([BarChart, LineChart, PieChart, ScatterChart, GridComponent, TooltipComponent, MarkLineComponent, CanvasRenderer]);

export interface ChartClick { dataIndex: number; seriesIndex: number; data: unknown }
export function Chart({ option, className, label, onClick }: { option: EChartsCoreOption; className: string; label: string; onClick?: (event: ChartClick) => void }) {
  const element = useRef<HTMLDivElement>(null);
  const instance = useRef<ReturnType<typeof echarts.init> | null>(null);
  useEffect(() => {
    const chart = echarts.init(element.current!, undefined, { renderer: "canvas" });
    instance.current = chart;
    const observer = new ResizeObserver(() => {
      const container = element.current;
      if (container && (container.clientWidth !== chart.getWidth() || container.clientHeight !== chart.getHeight())) {
        chart.resize();
      }
    });
    observer.observe(element.current!);
    return () => { observer.disconnect(); chart.dispose(); instance.current = null; };
  }, []);
  useEffect(() => {
    const chart = instance.current;
    if (!chart || !onClick) return;
    const click = (event: unknown) => onClick(event as ChartClick);
    chart.on("click", click);
    return () => { chart.off("click", click); };
  }, [onClick]);
  useEffect(() => {
    const animated = option.animation === true && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    // Keep animated series between updates so live reads do not replay the entrance.
    instance.current?.setOption({ ...option, animation: animated }, { notMerge: !animated });
  }, [option]);
  return <div ref={element} className={className} role="img" aria-label={label} />;
}
