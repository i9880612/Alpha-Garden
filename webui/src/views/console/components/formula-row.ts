import type { HTMLAttributes } from "react";

export function formulaRowProps(
  row: { task_id: string; alpha_id?: string | null },
  zh: boolean,
  onOpen: (taskId: string) => void,
): HTMLAttributes<HTMLTableRowElement> {
  return {
    className: "ag-formula-row",
    tabIndex: 0,
    "aria-label": `${zh ? "查看公式" : "View formula"} ${row.alpha_id || row.task_id}`,
    onClick: (event) => {
      if (
        (event.target as HTMLElement).closest("a, button, input, select, textarea") ||
        window.getSelection()?.toString()
      )
        return;
      onOpen(row.task_id);
    },
    onKeyDown: (event) => {
      if (event.target !== event.currentTarget || !["Enter", " "].includes(event.key)) return;
      event.preventDefault();
      onOpen(row.task_id);
    },
  };
}
