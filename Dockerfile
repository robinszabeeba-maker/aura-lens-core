# 1. 保持 Python 3.10 稳定版
FROM python:3.10-slim

# 2. 优化安装命令：增加 --fix-missing 并合在一行
# 这里的 --no-install-recommends 能让镜像更轻量
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 升级 pip 本身，确保安装环境最新
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]