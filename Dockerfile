# 1. 继续使用全量版，确保环境底座厚实
FROM python:3.10

# 2. 设置不弹出交互窗口
ENV DEBIAN_FRONTEND=noninteractive

# 3. 核心修复：安装 Mediapipe 必须的系统库 + 中文渲染字体
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. 升级 pip 并安装依赖
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制代码并启动
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]