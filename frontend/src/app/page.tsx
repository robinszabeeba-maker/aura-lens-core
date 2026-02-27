"use client";

import { useCallback, useEffect, useState } from "react";
import type React from "react";

import { ShareCard } from "./components/ShareCard";

const ANALYZE_API_URL = "http://127.0.0.1:8000/analyze";
const MAX_UPLOAD_WIDTH = 1000;

async function compressImageFile(
  file: File,
  maxWidth: number = MAX_UPLOAD_WIDTH
): Promise<File> {
  return new Promise((resolve) => {
    try {
      const img = new Image();
      const url = URL.createObjectURL(file);
      img.onload = () => {
        URL.revokeObjectURL(url);
        const canvas = document.createElement("canvas");
        const ratio = img.width > maxWidth ? maxWidth / img.width : 1;
        const targetWidth = img.width * ratio;
        const targetHeight = img.height * ratio;
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          resolve(file);
          return;
        }
        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
        canvas.toBlob(
          (blob) => {
            if (!blob) {
              resolve(file);
              return;
            }
            const compressed = new File([blob], file.name, {
              type: "image/jpeg",
              lastModified: Date.now(),
            });
            resolve(compressed);
          },
          "image/jpeg",
          0.8
        );
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(file);
      };
      img.src = url;
    } catch {
      resolve(file);
    }
  });
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
  const [compliment, setCompliment] = useState<string | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [displayScore, setDisplayScore] = useState<number>(0);
  const [label, setLabel] = useState<string | null>(null);
  const [showShareCard, setShowShareCard] = useState(false);

  useEffect(() => {
    if (score == null) {
      setDisplayScore(0);
      return;
    }
    let start: number | null = null;
    const duration = 800;
    const from = 0;
    const to = score;
    let frameId: number;

    const step = (timestamp: number) => {
      if (start === null) start = timestamp;
      const progress = Math.min((timestamp - start) / duration, 1);
      const value = Math.round(from + (to - from) * progress);
      setDisplayScore(value);
      if (progress < 1) {
        frameId = requestAnimationFrame(step);
      }
    };

    frameId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frameId);
  }, [score]);

  const handleFile = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const selected = files[0];
    if (!selected.type.startsWith("image/")) {
      setError("只能上传图片文件哦。");
      return;
    }
    setError(null);
    setResultImageUrl(null);
    setCompliment(null);
    setScore(null);
    setLabel(null);

    const compressed = await compressImageFile(selected);
    setFile(compressed);

    const url = URL.createObjectURL(compressed);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
  }, []);

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    handleFile(event.dataTransfer.files);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError("请先上传一张人脸照片。");
      return;
    }
    console.log("[AuraLens] 即将发起分析请求", {
      api: ANALYZE_API_URL,
      fileName: file.name,
      fileType: file.type,
      fileSize: file.size,
    });
    setLoading(true);
    setError(null);
    setResultImageUrl(null);
    setCompliment(null);
    setScore(null);
    setLabel(null);

    try {
      const formData = new FormData();
      // FastAPI 后端参数名为 `file`，键名保持完全一致
      formData.append("file", file);

      const response = await fetch(ANALYZE_API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        console.error("[AuraLens] 后端返回非 2xx 响应", {
          status: response.status,
          statusText: response.statusText,
          body: text,
        });
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      console.log("[AuraLens] 后端返回数据", data);

      // 针对后端业务错误做友好提示
      if (data?.error) {
        console.warn("[AuraLens] 后端业务错误", data.error);
        if (data.error === "NO_FACE_DETECTED") {
          setError(data.message || "未检测到人脸，请换个角度再试。");
        } else {
          setError(data.message || `后端返回错误：${data.error}`);
        }
        setResultImageUrl(null);
        setCompliment(null);
        setScore(null);
        setLabel(null);
        return;
      }

      // FastAPI 当前返回字段：image_data / vibe_text / score
      const imageUrl =
        data.image_url ||
        data.result_image_url ||
        data.image ||
        data.image_data ||
        null;
      const complimentText =
        data.compliment ||
        data.praise ||
        data.text ||
        data.vibe_text ||
        null;

      const receivedScore =
        typeof data.score === "number" ? data.score : null;
      if (receivedScore != null) {
        setScore(receivedScore);
      }

      const receivedLabel =
        typeof data.label === "string" ? data.label : null;
      if (receivedLabel != null) {
        setLabel(receivedLabel);
      }

      if (imageUrl) {
        setResultImageUrl(imageUrl);
      }
      setCompliment(
        complimentText ??
          "你的颜值自带主角光环，走到哪里都是焦点。"
      );
    } catch (err) {
      console.error("[AuraLens] 调用 /analyze 接口异常", err);
      setError("分析失败，请确认后端服务已在 8000 端口启动，再重试一次。");
    } finally {
      setLoading(false);
    }
  };

  const onFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    handleFile(event.target.files);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-slate-950 to-neutral-900 text-slate-100 font-sans">
      <main className="mx-auto flex min-h-screen max-w-5xl flex-col px-4 py-10 md:px-8 md:py-14">
        <header className="mb-10 flex items-center justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.9)]" />
              AuraLens · Face Insight
            </div>
            <h1 className="mt-4 bg-gradient-to-r from-slate-50 via-sky-300 to-emerald-300 bg-clip-text text-3xl font-semibold tracking-tight text-transparent md:text-4xl">
              颜值打分器 · AuraLens
            </h1>
            <p className="mt-3 max-w-xl text-sm text-slate-400 md:text-base">
              拖拽或上传一张人像照片，AuraLens 将为你生成一张带标注的分析图，
              再配上一段专属夸夸文案，让自信感瞬间拉满。
            </p>
          </div>
        </header>

        <section className="grid flex-1 gap-6 md:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
          <div className="flex flex-col rounded-3xl border border-white/10 bg-white/5/10 bg-gradient-to-b from-white/5 to-white/[0.02] p-5 shadow-[0_18px_45px_rgba(0,0,0,0.6)] backdrop-blur-xl md:p-6">
            <h2 className="text-sm font-medium text-slate-200">
              上传人像 · Drag & Drop
            </h2>
            <p className="mt-1 text-xs text-slate-400">
              支持拖拽图片到下方区域，或点击选择文件。建议正脸清晰照片。
            </p>

            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() =>
                document.getElementById("file-input")?.click()
              }
              className={`mt-4 flex flex-1 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-10 text-center transition-all duration-200 ${
                isDragging
                  ? "border-sky-400 bg-sky-500/10"
                  : "border-white/15 hover:border-sky-400/80 hover:bg-white/[0.04]"
              }`}
            >
              {previewUrl ? (
                <>
                  <div className="relative mb-4 w-full max-w-sm overflow-hidden rounded-2xl border border-white/10 bg-black/40">
                    <img
                      src={previewUrl}
                      alt="预览人像"
                      className="h-64 w-full object-cover"
                    />
                  </div>
                  <p className="text-xs text-slate-300">
                    已选择图片，点击下方按钮开始分析。
                  </p>
                </>
              ) : (
                <>
                  <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-sky-500/15 text-sky-300 shadow-[0_0_25px_rgba(56,189,248,0.5)]">
                    <span className="text-xl">⤓</span>
                  </div>
                  <p className="text-sm font-medium text-slate-100">
                    拖拽图片到这里，或点击上传
                  </p>
                  <p className="mt-1 text-xs text-slate-400">
                    支持 JPG / PNG / WEBP，文件仅在本地会话中使用。
                  </p>
                </>
              )}
              <input
                id="file-input"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={onFileInputChange}
              />
            </div>

            <div className="mt-4 flex items-center justify-between gap-3">
              <button
                onClick={handleAnalyze}
                disabled={!file || loading}
                className="inline-flex min-w-[140px] items-center justify-center rounded-full bg-sky-500 px-5 py-2.5 text-sm font-medium text-slate-950 shadow-[0_18px_45px_rgba(56,189,248,0.55)] transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                {loading ? "分析中…" : "开始分析颜值"}
              </button>
              <p className="text-[11px] text-slate-500">
                所有处理均在你的私有后端完成，前端不做持久化存储。
              </p>
            </div>

            {error && (
              <p className="mt-3 rounded-xl border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200">
                {error}
              </p>
            )}
          </div>

          <div className="flex flex-col rounded-3xl border border-white/10 bg-white/5/10 bg-gradient-to-b from-white/5 to-white/[0.02] p-5 shadow-[0_18px_45px_rgba(0,0,0,0.6)] backdrop-blur-xl md:p-6">
            <h2 className="text-sm font-medium text-slate-200">
              分析结果 · Aura Insight
            </h2>
            <p className="mt-1 text-xs text-slate-400">
              完成分析后，将在这里展示带标注的人像图，以及为你量身打造的夸夸文案。
            </p>

            <div className="mt-4 flex-1 space-y-4">
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <div className="flex items-baseline gap-2">
                  <span className="text-[0.7rem] uppercase tracking-[0.2em] text-slate-500">
                    BEAUTY SCORE
                  </span>
                  {score != null && (
                    <span className="text-[0.7rem] rounded-full border border-emerald-400/40 px-2 py-[2px] text-emerald-200">
                      AI 颜值评估
                    </span>
                  )}
                </div>
                {score != null && (
                  <div className="flex items-end gap-1">
                    <span className="bg-gradient-to-r from-emerald-300 via-sky-300 to-slate-50 bg-clip-text text-3xl font-semibold text-transparent md:text-4xl transition-transform duration-500">
                      {displayScore}
                    </span>
                    <span className="pb-0.5 text-xs text-slate-400">
                      / 100
                    </span>
                  </div>
                )}
              </div>

              <div
                className={`relative flex min-h-[180px] items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-black/40 transition-all duration-500 ease-out ${
                  resultImageUrl ? "opacity-100 translate-y-0" : "opacity-70"
                }`}
              >
                {loading ? (
                  <div className="flex flex-col items-center gap-3">
                    <div className="h-10 w-10 animate-spin rounded-full border-2 border-sky-500/60 border-t-transparent" />
                    <p className="text-xs text-slate-300">
                      正在为你的颜值生成「专业打光 + 高情商夸夸」…
                    </p>
                  </div>
                ) : resultImageUrl ? (
                  <img
                    key={resultImageUrl}
                    src={resultImageUrl}
                    alt="分析结果图"
                    className="max-h-72 w-full object-contain"
                  />
                ) : (
                  <p className="max-w-xs text-center text-xs text-slate-500">
                    还没有结果。上传一张照片并点击「开始分析颜值」，我们会为你生成一张高亮优点的分析图。
                  </p>
                )}
              </div>

              <div
                className={`rounded-2xl border border-emerald-400/25 bg-emerald-500/8 px-4 py-3 text-sm text-emerald-100 shadow-[0_0_24px_rgba(52,211,153,0.25)] transition-all duration-500 ease-out ${
                  compliment ? "opacity-100 translate-y-0" : "opacity-80"
                }`}
              >
                {compliment ? (
                  <p className="leading-relaxed">{compliment}</p>
                ) : (
                  <p className="text-xs text-emerald-200/80">
                    分析完成后，这里会生成一段专属的夸夸文案，从五官精致度、整体气质、
                    镜头表现力等多个维度，帮你把「不好意思说出口的夸奖」一次性说完。
                  </p>
                )}
              </div>

              {resultImageUrl && score != null && compliment && (
                <button
                  type="button"
                  onClick={() => setShowShareCard(true)}
                  className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-full border border-sky-400/60 bg-sky-500/10 px-4 py-2.5 text-sm font-medium text-sky-200 transition hover:bg-sky-500/20"
                >
                  <span className="text-base">📤</span>
                  生成分享卡片
                </button>
              )}
            </div>
          </div>
        </section>
      </main>

      {showShareCard && resultImageUrl && score != null && compliment && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
          <div className="relative w-full max-w-md rounded-3xl border border-white/10 bg-slate-950/95 p-4 shadow-[0_24px_80px_rgba(0,0,0,0.9)]">
            <button
              type="button"
              onClick={() => setShowShareCard(false)}
              className="absolute right-4 top-4 rounded-full p-1.5 text-slate-400 transition hover:bg-white/10 hover:text-slate-200"
              aria-label="关闭"
            >
              <span className="text-lg">×</span>
            </button>

            <ShareCard
              imageUrl={resultImageUrl}
              score={score}
              label={label ?? "AI 颜值风格"}
              vibeText={compliment}
            />
          </div>
        </div>
      )}
    </div>
  );
}
