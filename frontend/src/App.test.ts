import { describe, expect, it } from "vitest";
import { workspaceViewFromHash } from "./App";

describe("workspaceViewFromHash", () => {
  it("restores the evaluations workspace after refresh", () => {
    expect(workspaceViewFromHash("#evaluations")).toBe("evaluations");
  });

  it("defaults unknown and empty hashes to chat", () => {
    expect(workspaceViewFromHash("#chat")).toBe("chat");
    expect(workspaceViewFromHash("")).toBe("chat");
  });
});
