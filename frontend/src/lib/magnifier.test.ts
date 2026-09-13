import { describe, expect, it } from "vitest";
import {
  MAGNIFIER_SECONDS,
  MAGNIFIER_WIDTH,
  magnifierPosition,
  magnifierSegmentGeometry,
  magnifierTimeFromClientX,
  magnifierWindow,
} from "./magnifier";
import type { TimelineSegment } from "../types";

const seg = (start: number, end: number): TimelineSegment => ({
  index: 0,
  start,
  end,
});

describe("magnifierWindow", () => {
  it("中段以悬停时间为中心，窗口保持 8 秒", () => {
    expect(magnifierWindow(100, 50)).toEqual({
      windowStart: 46,
      windowEnd: 54,
      windowSeconds: MAGNIFIER_SECONDS,
      centerPct: 50,
    });
  });

  it("靠近片头时窗口整体平移到 0，中心点随之偏移", () => {
    const w = magnifierWindow(100, 2);
    expect(w.windowStart).toBe(0);
    expect(w.windowSeconds).toBe(8);
    expect(w.centerPct).toBeCloseTo(25, 6);
  });

  it("靠近片尾时窗口贴住片尾，不越过总时长", () => {
    const w = magnifierWindow(100, 99);
    expect(w.windowStart).toBe(92);
    expect(w.windowEnd).toBe(100);
    expect(w.windowSeconds).toBe(8);
    expect(w.centerPct).toBeCloseTo(87.5, 6);
  });

  it("总时长不足 8 秒时窗口被总时长夹住", () => {
    expect(magnifierWindow(5, 2)).toEqual({
      windowStart: 0,
      windowEnd: 5,
      windowSeconds: 5,
      centerPct: 40,
    });
  });

  it("未悬停时窗口为空、中心点取 50", () => {
    expect(magnifierWindow(100, null)).toEqual({
      windowStart: 0,
      windowEnd: 0,
      windowSeconds: 0,
      centerPct: 50,
    });
  });
});

describe("magnifierSegmentGeometry", () => {
  const geo = magnifierSegmentGeometry(46, 54, 8);

  it("只保留与窗口相交的时段", () => {
    expect(geo.inWindow(seg(45, 47))).toBe(true);
    expect(geo.inWindow(seg(53, 60))).toBe(true);
    expect(geo.inWindow(seg(40, 46))).toBe(false);
    expect(geo.inWindow(seg(54, 60))).toBe(false);
  });

  it("跨出窗口的时段被夹到 0%–100%，宽度不为负", () => {
    expect(geo.leftPct(seg(45, 47))).toBe(0);
    expect(geo.widthPct(seg(45, 47))).toBeCloseTo(12.5, 6);
    expect(geo.leftPct(seg(52, 60))).toBeCloseTo(75, 6);
    expect(geo.widthPct(seg(52, 60))).toBeCloseTo(25, 6);
    expect(geo.widthPct(seg(40, 60))).toBe(100);
  });
});

describe("magnifierPosition", () => {
  it("左边缘夹到 8px，上边缘夹到 140px", () => {
    expect(magnifierPosition(100, 60, 50, 1280)).toEqual({ left: 8, top: 140 });
  });

  it("右边缘留出 8px 边距", () => {
    expect(magnifierPosition(1200, 300, 50, 1280)).toEqual({
      left: 1280 - MAGNIFIER_WIDTH - 8,
      top: 300,
    });
  });

  it("中线对准鼠标：按 centerPct 补偿 padding box 宽度", () => {
    expect(magnifierPosition(600, 300, 25, 1280).left).toBe(
      600 - 0.25 * (MAGNIFIER_WIDTH - 2),
    );
  });
});

describe("magnifierTimeFromClientX", () => {
  const container = { left: 0, width: 1000 };
  const track = { left: 200, width: 800 };

  it("以轨道列宽度为基准换算时间", () => {
    expect(magnifierTimeFromClientX(600, container, track, 100)).toBeCloseTo(50, 6);
    expect(magnifierTimeFromClientX(200, container, track, 100)).toBe(0);
    expect(magnifierTimeFromClientX(1000, container, track, 100)).toBe(100);
  });

  it("越界夹取到时长两端", () => {
    expect(magnifierTimeFromClientX(-500, container, track, 100)).toBe(0);
    expect(magnifierTimeFromClientX(5000, container, track, 100)).toBe(100);
  });

  it("找不到轨道列时退回整块宽度", () => {
    expect(magnifierTimeFromClientX(250, container, null, 100)).toBe(25);
    expect(magnifierTimeFromClientX(250, container, { left: 0, width: 0 }, 100)).toBe(25);
  });
});
