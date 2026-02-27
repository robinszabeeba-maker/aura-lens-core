"use client";

import React, { useRef, useState } from "react";

type ShareCardProps = {
  imageUrl: string;
  score: number;
  label: string;
  vibeText: string;
};

const techBackgroundSvg = `
<svg width="400" height="600" viewBox="0 0 400 600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg-grad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#020617"/>
      <stop offset="100%" stop-color="#1e1b4b"/>
    </linearGradient>

    <filter id="soft-glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="80" result="blur"/>
    </filter>

    <pattern id="grid" x="0" y="0" width="24" height="24" patternUnits="userSpaceOnUse">
      <path d="M 24 0 L 0 0 0 24" fill="none" stroke="rgba(148,163,184,0.35)" stroke-width="0.5"/>
    </pattern>
    <mask id="grid-mask">
      <rect x="0" y="0" width="400" height="600" fill="white" opacity="0.18"/>
    </mask>
  </defs>

  <rect x="0" y="0" width="400" height="600" fill="url(#bg-grad)"/>

  <g filter="url(#soft-glow)">
    <circle cx="340" cy="80" r="180" fill="#3b82f6" fill-opacity="0.18"/>
  </g>

  <rect x="0" y="0" width="400" height="600" fill="url(#grid)" mask="url(#grid-mask)"/>
</svg>
`;

const techBackgroundDataUri = `data:image/svg+xml;utf8,${encodeURIComponent(
  techBackgroundSvg
)}`;

export const ShareCard: React.FC<ShareCardProps> = ({
  imageUrl,
  score,
  label,
  vibeText,
}) => {
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [saving, setSaving] = useState(false);

  const handleSaveImage = async () => {
    const cardEl = document.getElementById("auralens-share-card");
    if (!cardEl || saving) return;
    setSaving(true);
    try {
      // 确保页面滚动到顶部，避免偏移导致截断
      window.scrollTo(0, 0);

      await new Promise((resolve) => setTimeout(resolve, 300));

      const areaEl = document.getElementById("share-card-area");
      const forcedHeight =
        areaEl && areaEl instanceof HTMLElement
          ? areaEl.scrollHeight
          : undefined;

      const html2canvas = (await import("html2canvas")).default;
      const canvas = await html2canvas(cardEl, {
        backgroundColor: "#0f172a",
        useCORS: true,
        allowTaint: false,
        scale: 2,
        logging: true,
        ignoreElements: (element) =>
          (element as HTMLElement).id === "auralens-save-button",
        height: forcedHeight,
        onclone(clonedDoc, clonedEl) {
          const root = typeof clonedEl === "function" ? clonedEl() : clonedEl;
          if (!root || !(root instanceof HTMLElement)) return;

          const doc = clonedDoc as Document;
          const view = doc.defaultView ?? window;

          const elements = doc.querySelectorAll<HTMLElement>("*");
          elements.forEach((el) => {
            try {
              const style = view.getComputedStyle(el);
              const bg = style.backgroundColor;
              const color = style.color;
              const borderColor = style.borderColor;

              if (bg && (bg.includes("lab(") || bg.includes("oklch("))) {
                el.style.backgroundColor = "#0f172a";
              }
              if (color && (color.includes("lab(") || color.includes("oklch("))) {
                el.style.color = "#e2e8f0";
              }
              if (
                borderColor &&
                (borderColor.includes("lab(") || borderColor.includes("oklch("))
              ) {
                el.style.borderColor = "rgba(148,163,184,0.4)";
              }
            } catch (err) {
              console.warn(
                "[AuraLens] onclone 处理颜色时出错，已跳过该元素",
                err
              );
            }
          });
        },
      });

      const dataUrl = canvas.toDataURL("image/png");
      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = "auralens-report.png";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error("[AuraLens] 保存分享卡片失败", error);
      console.error("[AuraLens] 错误详情:", {
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        name: error instanceof Error ? error.name : undefined,
      });
      alert("保存失败，请打开浏览器控制台 (F12) 查看具体错误信息。");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <div
        id="auralens-share-card"
        ref={cardRef}
        className="relative w-full max-w-sm"
      >
        {/* 海报容器：使用 Base64 SVG 背景，避免 CSS lab/oklch */}
        <div
          id="share-card-area"
          className="relative flex flex-col overflow-hidden p-5"
          style={{
            width: "450px",
            minHeight: "600px",
            borderRadius: "1.5rem",
            border: "1px solid rgba(255,255,255,0.1)",
            backgroundColor: "#020617",
            backgroundImage: `url("${techBackgroundDataUri}")`,
            backgroundSize: "100% 100%",
            backgroundPosition: "center",
            boxShadow: "0 24px 70px rgba(0,0,0,0.9)",
          }}
        >
          {/* 网格底纹：纯 rgba */}
          <div
            className="pointer-events-none absolute inset-0"
            style={{
              opacity: 0.25,
              backgroundImage: "radial-gradient(circle at 1px 1px, rgba(148,163,184,0.22) 1px, transparent 0)",
              backgroundSize: "18px 18px",
            }}
          />

          <div className="relative flex h-full flex-col">
            <header className="flex items-center justify-between text-xs" style={{ color: "#cbd5e1" }}>
              <div className="flex flex-col gap-1">
                <span className="uppercase tracking-[0.18em]" style={{ fontSize: "0.65rem", color: "#94a3b8" }}>
                  AURA LENS 分析报告
                </span>
                <span style={{ fontSize: "0.7rem", color: "#64748b" }}>
                  AI Face Aesthetics Insight
                </span>
              </div>
              <div className="flex items-center gap-1">
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{ background: "#34d399", boxShadow: "0 0 10px rgba(52,211,153,0.9)" }}
                />
                <span style={{ fontSize: "0.65rem", color: "#a7f3d0" }}>REAL-TIME AI</span>
              </div>
            </header>

            <div className="mt-4 flex-1">
              <div className="relative mx-auto h-full w-full" style={{ maxHeight: "55%", maxWidth: "88%" }}>
                <div
                  className="absolute rounded-2xl blur-sm"
                  style={{
                    inset: "-1px",
                    background: "linear-gradient(135deg, rgba(56,189,248,0.8), rgba(52,211,153,0.6), rgba(139,92,246,0.6))",
                    opacity: 0.8,
                  }}
                />
                <div
                  className="relative h-full w-full overflow-hidden rounded-2xl"
                  style={{
                    border: "1px solid rgba(186,230,253,0.4)",
                    background: "rgba(0,0,0,0.6)",
                    boxShadow: "0 0 40px rgba(56,189,248,0.6)",
                  }}
                >
                  <div
                    className="pointer-events-none absolute inset-0"
                    style={{
                      background: "radial-gradient(circle at top, rgba(248,250,252,0.08), transparent 55%)",
                    }}
                  />
                  <img
                    src={imageUrl}
                    alt="AI 分析结果"
                    className="h-full w-full object-cover"
                    {...(imageUrl.startsWith("data:") ? {} : { crossOrigin: "anonymous" })}
                  />
                </div>
              </div>
            </div>

            <div
              className="mt-4 flex items-end justify-between"
              style={{ minHeight: "96px" }}
            >
              <div className="flex flex-col gap-1">
                <span className="uppercase tracking-[0.18em]" style={{ fontSize: "0.7rem", color: "#64748b" }}>
                  BEAUTY SCORE
                </span>
                <div className="flex items-end gap-2">
                  <span
                    className="text-5xl font-semibold leading-none"
                    style={{
                      color: "#facc15",
                    }}
                  >
                    {score}
                  </span>
                  <span className="pb-1 text-xs" style={{ color: "#94a3b8" }}>/ 100</span>
                </div>
              </div>
              <div className="flex flex-col items-end gap-2">
                <span
                  className="inline-flex items-center gap-1 rounded-full px-3 py-1 text-[0.7rem] font-medium"
                  style={{
                    border: "1px solid rgba(110,231,183,0.6)",
                    background: "rgba(52,211,153,0.1)",
                    color: "#a7f3d0",
                    boxShadow: "0 0 18px rgba(52,211,153,0.45)",
                  }}
                >
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: "#6ee7b7" }} />
                  {label}
                </span>
                <span className="uppercase tracking-[0.18em]" style={{ fontSize: "0.6rem", color: "#64748b" }}>
                  FACE TYPE SIGNATURE
                </span>
              </div>
            </div>

            <div className="mt-4 flex-1">
              <div
                className="rounded-2xl px-4 py-3 text-xs"
                style={{
                  border: "1px solid rgba(255,255,255,0.1)",
                  background: "rgba(15,23,42,0.7)",
                  color: "#e2e8f0",
                  boxShadow: "0 0 24px rgba(15,23,42,0.85)",
                }}
              >
                <div className="mb-1 uppercase tracking-[0.16em]" style={{ fontSize: "0.7rem", color: "#94a3b8" }}>
                  AURA COMMENTARY
                </div>
                <p className="text-[0.8rem] leading-relaxed" style={{ color: "#e2e8f0" }}>
                  {vibeText}
                </p>
              </div>
            </div>

            <footer className="mt-4 flex items-center justify-between" style={{ fontSize: "0.7rem", color: "#94a3b8" }}>
              <div className="flex items-center gap-3">
                <div
                  className="flex h-12 w-12 items-center justify-center rounded-md text-[0.55rem] font-semibold tracking-widest"
                  style={{
                    backgroundColor: "#0ea5e9",
                    color: "#0f172a",
                    boxShadow: "0 0 22px rgba(56,189,248,0.75)",
                  }}
                >
                  QR
                </div>
                <div className="flex flex-col gap-1">
                  <span style={{ fontSize: "0.7rem", color: "#cbd5e1" }}>扫码解锁你的 AI 面相</span>
                  <span style={{ fontSize: "0.6rem", color: "#64748b" }}>https://your-website.com</span>
                </div>
              </div>
              <div className="flex flex-col items-end text-[0.6rem]" style={{ color: "#64748b" }}>
                <span>AuraLens · Face Insight</span>
                <span>Powered by Private AI</span>
              </div>
            </footer>
          </div>
        </div>
      </div>

      <button
        type="button"
        onClick={handleSaveImage}
        disabled={saving}
        id="auralens-save-button"
        className="inline-flex items-center justify-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium transition disabled:cursor-not-allowed"
        style={{
          background: saving ? "#334155" : "#0ea5e9",
          color: saving ? "#94a3b8" : "#020617",
          boxShadow: "0 18px 45px rgba(56,189,248,0.65)",
        }}
      >
        {saving ? "生成中…" : "保存海报"}
      </button>
    </div>
  );
};
