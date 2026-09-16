import { StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { unstableSetRender } from "antd";
import "antd/dist/reset.css";
import App from "./App.tsx";
import "./index.css";
import "./styles/sidebar-collapse-fix.css";
import "./styles/mobile-responsive.css";
import "./styles/console.css";

// Ant Design 5's render hook keeps its overlays and wave effects on React 19's createRoot API.
// https://github.com/ant-design/ant-design/blob/5.29.3/docs/react/v5-for-19.en-US.md
const componentRoots = new WeakMap<Element | DocumentFragment, Root>();
unstableSetRender((node, container) => {
  const root = componentRoots.get(container) ?? createRoot(container);
  componentRoots.set(container, root);
  root.render(node);
  return async () => {
    await new Promise<void>(resolve => setTimeout(resolve, 0));
    root.unmount();
    componentRoots.delete(container);
  };
});

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
