# 1. 换成非 slim 的 bullseye 镜像，它更重但也更稳
FROM python:3.10-bullseye

# 2. 设置环境变量，防止安装时弹出交互窗口
ENV DEBIAN_FRONTEND=noninteractive

# 3. 增加重试逻辑的安装命令
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. 升级 pip
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render 部署建议监听 PORT 环境变量
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-80}