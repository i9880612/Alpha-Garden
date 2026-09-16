declare module "three" {
  export class Vector2 {
    constructor(x?: number, y?: number);
    fromArray(array: number[]): this;
  }

  export class Vector3 {
    constructor(x?: number, y?: number, z?: number);
    fromArray(array: number[]): this;
  }

  export interface ShaderMaterialParameters {
    vertexShader?: string;
    fragmentShader?: string;
    uniforms?: Record<string, unknown>;
    glslVersion?: string;
    blending?: number;
    blendSrc?: number;
    blendDst?: number;
  }

  export class ShaderMaterial {
    constructor(parameters?: ShaderMaterialParameters);
    uniforms: Record<string, { value: unknown }>;
  }

  export interface Mesh {
    material: unknown;
  }

  export const GLSL3: string;
  export const CustomBlending: number;
  export const SrcAlphaFactor: number;
  export const OneFactor: number;
}
