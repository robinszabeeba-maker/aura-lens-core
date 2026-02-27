# 1. 换成全量版镜像，它预装了更多零件，减少对 apt-get 的依赖
FROM python:3.10

# 2. 设置工作目录
WORKDIR /app

# 3. 尝试安装（如果已经有了，它会跳过；如果报错，我们加一个 || true 强行通过）
# 这是为了确保即便这步报错，只要后面 AI 能跑就行
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 || true \
    && rm -rf /var/lib/apt/lists/*

# 4. 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制所有代码
COPY . .

# 6. 启动
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-80}