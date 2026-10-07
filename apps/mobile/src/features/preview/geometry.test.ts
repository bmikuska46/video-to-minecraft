import { describe, expect, it } from "vitest";

import { normalizedCrop, rigidTransform, transformedBounds } from "./geometry";

describe("preview geometry contract", () => {
  it("creates a proper row-major quarter-turn transform", () => {
    expect(rigidTransform({ x: 1, y: 0, z: 0 })).toEqual([
      1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1,
    ]);
  });

  it("recomputes available crop bounds after rotation", () => {
    expect(transformedBounds(
      { min: [-2, 0, -1], max: [2, 3, 1] },
      rigidTransform({ x: 1, y: 0, z: 0 }),
    )).toEqual({ min: [-2, -1, 0], max: [2, 1, 3] });
  });

  it("clamps crop edits to the available bounds and preserves ordering", () => {
    expect(normalizedCrop(
      { min: [0, 0, 0], max: [10, 10, 10] },
      { min: [-2, 4, 3], max: [8, 2, 20] },
    )).toEqual({ min: [0, 2, 3], max: [8, 2, 10] });
  });
});
