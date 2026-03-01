"""
AuraLens 颜值分析 API
- 所有 import 置于顶部，避免命名空间冲突（确保 import mediapipe 加载的是 pip 安装的官方库）。
- 不修改 sys.path。
"""
import base64
import io
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont

# ----- 自检：Render 日志中可一眼看出 mediapipe 加载路径 -----
print("RENDER_DEBUG mediapipe.__file__ =", getattr(mp, "__file__", "N/A"))
print("RENDER_DEBUG os.listdir('.') =", os.listdir("."))

# ----- 启动日志：Render 健康检查依赖服务绑定到 $PORT -----
_bind_port = os.environ.get("PORT", "80")
print("STARTUP BIND_PORT (from env):", _bind_port)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mem_log(label: str) -> None:
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        print(f"MEMORY_LOG {label} (RSS max approx KB): {usage.ru_maxrss}")
    except Exception:
        print(f"MEMORY_LOG {label} (resource not available)")


face_mesh = None
try:
    _mem_log("before model load")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    _mem_log("after model load")
except Exception as e:
    print("INIT_ERROR", str(e), file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise


@app.get("/")
def health():
    return {"status": "ok", "message": "Aura Lens API is running"}


def add_chinese_text(img, text, position, size=30, color=(255, 255, 255)):
    """绘制中文文案；Linux/Docker 下无 PingFang 时使用默认字体。"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = None
    for path in (
        "/System/Library/Fonts/PingFang.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ):
        try:
            font = ImageFont.truetype(path, size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)


@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    if face_mesh is None:
        return JSONResponse(
            content={"error": "SERVICE_UNAVAILABLE", "message": "模型未就绪，请稍后重试。"},
            status_code=503,
        )
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "INVALID_IMAGE", "message": "无法解析图片，请上传有效的 JPG/PNG。"}
    h, w, _ = image.shape

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return {
            "error": "NO_FACE_DETECTED",
            "message": "未检测到人脸，请换个角度再试",
        }

    landmarks = results.multi_face_landmarks[0].landmark
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    score = int(min(100, max(0, 90 - abs(left_eye.y - right_eye.y) * 1000)))

    if score > 85:
        label = "未来猫系脸"
        vibe_text = "你的五官比例极具电影感，是非常上镜的高级脸。"
    else:
        label = "治愈犬系脸"
        vibe_text = "你的面部线条柔和，给人一种非常亲切的温暖感。"

    for lm in landmarks:
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 1, (0, 0, 255), -1)
    image = add_chinese_text(image, f"评分: {score}", (20, 30), 40, (0, 255, 255))
    image = add_chinese_text(image, f"风格: {label}", (20, 80), 30, (255, 200, 0))
    image = add_chinese_text(image, vibe_text, (20, h - 60), 25, (255, 255, 255))

    _, buffer = cv2.imencode(".jpg", image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "score": score,
        "label": label,
        "vibe_text": vibe_text,
        "image_data": f"data:image/jpeg;base64,{img_base64}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
