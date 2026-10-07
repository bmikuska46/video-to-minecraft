import type { Axis, Bounds, Vector3 } from "@/api/client";

export type QuarterTurns = { x: number; y: number; z: number };

export const IDENTITY_TURNS: QuarterTurns = { x: 0, y: 0, z: 0 };

const multiply = (left: number[], right: number[]) => {
  const result = new Array<number>(16).fill(0);
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      for (let index = 0; index < 4; index += 1) {
        const position = row * 4 + column;
        result[position] = result[position]! + left[row * 4 + index]! * right[index * 4 + column]!;
      }
    }
  }
  return result;
};

const rotation = (axis: Axis, turns: number) => {
  const angle = ((turns % 4) * Math.PI) / 2;
  const c = Math.round(Math.cos(angle));
  const s = Math.round(Math.sin(angle));
  if (axis === "x") return [1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1];
  if (axis === "y") return [c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1];
  return [c,-s,0,0, s,c,0,0, 0,0,1,0, 0,0,0,1];
};

export function rigidTransform(turns: QuarterTurns): number[] {
  return multiply(multiply(rotation("z", turns.z), rotation("y", turns.y)), rotation("x", turns.x));
}

const transformPoint = (point: Vector3, matrix: number[]): Vector3 => [
  matrix[0]! * point[0] + matrix[1]! * point[1] + matrix[2]! * point[2] + matrix[3]!,
  matrix[4]! * point[0] + matrix[5]! * point[1] + matrix[6]! * point[2] + matrix[7]!,
  matrix[8]! * point[0] + matrix[9]! * point[1] + matrix[10]! * point[2] + matrix[11]!,
];

export function transformedBounds(bounds: Bounds, matrix: number[]): Bounds {
  const points: Vector3[] = [];
  for (const x of [bounds.min[0], bounds.max[0]])
    for (const y of [bounds.min[1], bounds.max[1]])
      for (const z of [bounds.min[2], bounds.max[2]]) points.push(transformPoint([x, y, z], matrix));
  return {
    min: [0, 1, 2].map((axis) => Math.min(...points.map((point) => point[axis]!))) as Vector3,
    max: [0, 1, 2].map((axis) => Math.max(...points.map((point) => point[axis]!))) as Vector3,
  };
}

export function normalizedCrop(bounds: Bounds, values: Bounds): Bounds {
  const crop = { min: [...values.min] as Vector3, max: [...values.max] as Vector3 };
  for (let axis = 0; axis < 3; axis += 1) {
    crop.min[axis] = Math.max(bounds.min[axis]!, Math.min(crop.min[axis]!, crop.max[axis]!));
    crop.max[axis] = Math.min(bounds.max[axis]!, Math.max(crop.max[axis]!, crop.min[axis]!));
  }
  return crop;
}

export const extent = (bounds: Bounds, axis: Axis) =>
  bounds.max[{ x: 0, y: 1, z: 2 }[axis]]! - bounds.min[{ x: 0, y: 1, z: 2 }[axis]]!;
