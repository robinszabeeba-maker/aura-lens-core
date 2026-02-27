# 删掉那个 dockerpull.com 的前缀，回归官方镜像
FROM python:3.10-slim

# 后面的安装依赖逻辑保持不变
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-80}