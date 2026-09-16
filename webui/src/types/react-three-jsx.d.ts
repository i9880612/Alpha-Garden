import "react";
import type { ReactNode, Ref } from "react";

type ThreeElementProps = {
  attach?: string;
  args?: unknown[];
  children?: ReactNode;
  object?: unknown;
  ref?: Ref<unknown>;
  [key: string]: unknown;
};

declare global {
  namespace JSX {
    interface IntrinsicElements {
      mesh: ThreeElementProps;
      planeGeometry: ThreeElementProps;
      primitive: ThreeElementProps;
    }
  }
}

declare module "react" {
  namespace JSX {
    interface IntrinsicElements {
      mesh: ThreeElementProps;
      planeGeometry: ThreeElementProps;
      primitive: ThreeElementProps;
    }
  }
}
