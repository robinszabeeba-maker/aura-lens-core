# 1. 使用官方 Python 3.10 镜像（为了兼容 Mediapipe）
FROM python:3.10-slim

# 2. 安装 Mediapipe 必须的 Linux 系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. 设置工作目录
WORKDIR /app

# 4. 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制你的 main.py 到容器中
COPY . .

# 6. 启动命令（Render 会自动分配端口）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]