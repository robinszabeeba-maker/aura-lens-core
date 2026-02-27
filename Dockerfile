# 1. 换成最新的 Debian 12 (bookworm) 底座，它更现代，源也更稳
FROM python:3.10-bookworm

# 2. 设置不弹出交互窗口
ENV DEBIAN_FRONTEND=noninteractive

# 3. 这种写法将所有指令合并，并增加了“清理”动作，成功率更高
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. 剩下的逻辑保持不变
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 适配 Render 端口
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-80}