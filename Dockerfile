# 第一步：准备底座
FROM python:3.10-bullseye

# 第二步：另起一行，加入这行“换源”指令（解决 Exit Code 100 的终极武器）
# 它的意思是：别去原来的超市了，去 Cloudflare 镜像超市，那里网速快。
RUN sed -i 's/deb.debian.org/cloudflare.cdn.openbsd.org/g' /etc/apt/sources.list

# 第三步：设置不弹出干扰窗口
ENV DEBIAN_FRONTEND=noninteractive

# 第四步：安装零件（现在成功率会大大提升）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 第五步：剩下的代码保持不变
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 适配 Render 的端口
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-80}