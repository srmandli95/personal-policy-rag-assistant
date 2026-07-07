import { describe, expect, it } from "vitest";
import { workspaceViewFromHash } from "./App";

describe("workspaceViewFromHash", () => {
  it("restores the evaluations workspace after refresh", () => {
    expect(workspaceViewFromHash("#evaluations")).toBe("evaluations");
  });

  it("restores the knowledge base workspace after refresh", () => {
    expect(workspaceViewFromHash("#knowledge-base")).toBe("knowledge-base");
  });

  it("defaults unknown and empty hashes to chat", () => {
    expect(workspaceViewFromHash("#chat")).toBe("chat");
    expect(workspaceViewFromHash("")).toBe("chat");
  });
});
