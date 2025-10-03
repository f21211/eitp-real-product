# EIT-P 生产级Docker镜像
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    hdf5-tools \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建应用目录
WORKDIR /app

# 复制项目文件
COPY . /app/

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 安装EIT-P包
RUN pip3 install -e .

# 创建非root用户
RUN useradd -m -u 1000 eitp && chown -R eitp:eitp /app
USER eitp

# 设置工作目录权限
RUN chmod +x /app/scripts/*.sh

# 暴露端口（用于监控和API）
EXPOSE 8080 8081

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import eit_p; print('EIT-P is healthy')" || exit 1

# 默认启动命令
CMD ["python3", "ultra_safe_train.py"]
