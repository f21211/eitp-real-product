# Enhanced CEP-EIT-P 部署和运维指南

## 概述

本指南详细介绍了Enhanced CEP-EIT-P框架的部署、运维和监控方法，包括生产环境配置、性能优化、故障排除等内容。

## 目录

1. [系统要求](#系统要求)
2. [部署准备](#部署准备)
3. [生产环境部署](#生产环境部署)
4. [API服务部署](#api服务部署)
5. [监控和告警](#监控和告警)
6. [性能优化](#性能优化)
7. [故障排除](#故障排除)
8. [运维管理](#运维管理)
9. [安全配置](#安全配置)
10. [备份和恢复](#备份和恢复)

## 系统要求

### 硬件要求

#### 最低配置
- **CPU**: 4核心 2.0GHz
- **内存**: 8GB RAM
- **存储**: 50GB SSD
- **网络**: 100Mbps

#### 推荐配置
- **CPU**: 8核心 3.0GHz+
- **内存**: 32GB RAM
- **存储**: 500GB NVMe SSD
- **网络**: 1Gbps
- **GPU**: NVIDIA RTX 3080+ (可选，用于加速推理)

#### 生产环境配置
- **CPU**: 16核心 3.5GHz+
- **内存**: 64GB RAM
- **存储**: 1TB NVMe SSD
- **网络**: 10Gbps
- **GPU**: NVIDIA A100 80GB (推荐)

### 软件要求

#### 操作系统
- Ubuntu 20.04 LTS (推荐)
- Ubuntu 22.04 LTS
- CentOS 8+
- RHEL 8+

#### Python环境
- Python 3.9+
- pip 21.0+

#### 依赖库
```bash
torch>=1.12.0
numpy>=1.21.0
flask>=2.0.0
requests>=2.25.0
matplotlib>=3.5.0
psutil>=5.8.0
```

## 部署准备

### 1. 环境准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y git curl wget vim htop tree

# 安装Python和pip
sudo apt install -y python3 python3-pip python3-venv

# 安装系统监控工具
sudo apt install -y htop iotop nethogs
```

### 2. 创建用户和目录

```bash
# 创建专用用户
sudo useradd -m -s /bin/bash eitp
sudo usermod -aG sudo eitp

# 创建应用目录
sudo mkdir -p /opt/eitp
sudo chown eitp:eitp /opt/eitp

# 创建日志目录
sudo mkdir -p /var/log/eitp
sudo chown eitp:eitp /var/log/eitp
```

### 3. 克隆项目

```bash
# 切换到应用用户
sudo su - eitp

# 克隆项目
cd /opt/eitp
git clone https://github.com/f21211/eitp-real-product.git
cd eitp-real-product
```

### 4. 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

## 生产环境部署

### 1. 系统服务配置

#### 创建systemd服务文件

```bash
sudo vim /etc/systemd/system/eitp-api.service
```

```ini
[Unit]
Description=Enhanced CEP-EIT-P API Server
After=network.target

[Service]
Type=simple
User=eitp
Group=eitp
WorkingDirectory=/opt/eitp/eitp-real-product
Environment=PATH=/opt/eitp/eitp-real-product/venv/bin
ExecStart=/opt/eitp/eitp-real-product/venv/bin/python enhanced_api_server_v2.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### 创建监控服务文件

```bash
sudo vim /etc/systemd/system/eitp-monitor.service
```

```ini
[Unit]
Description=Enhanced CEP-EIT-P Production Monitor
After=network.target eitp-api.service

[Service]
Type=simple
User=eitp
Group=eitp
WorkingDirectory=/opt/eitp/eitp-real-product
Environment=PATH=/opt/eitp/eitp-real-product/venv/bin
ExecStart=/opt/eitp/eitp-real-product/venv/bin/python production_monitor.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 2. 启动服务

```bash
# 重新加载systemd配置
sudo systemctl daemon-reload

# 启用服务
sudo systemctl enable eitp-api
sudo systemctl enable eitp-monitor

# 启动服务
sudo systemctl start eitp-api
sudo systemctl start eitp-monitor

# 检查服务状态
sudo systemctl status eitp-api
sudo systemctl status eitp-monitor
```

### 3. 配置Nginx反向代理

#### 安装Nginx

```bash
sudo apt install -y nginx
```

#### 配置Nginx

```bash
sudo vim /etc/nginx/sites-available/eitp
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # 缓冲设置
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
}
```

#### 启用配置

```bash
sudo ln -s /etc/nginx/sites-available/eitp /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## API服务部署

### 1. 基础API服务

```bash
# 启动基础API服务
cd /opt/eitp/eitp-real-product
source venv/bin/activate
./start_enhanced_production.sh
```

### 2. 增强版API服务V2

```bash
# 启动增强版API服务V2
./start_enhanced_v2.sh
```

### 3. 服务管理脚本

#### 创建统一管理脚本

```bash
vim /opt/eitp/eitp-real-product/manage_services.sh
```

```bash
#!/bin/bash
# Enhanced CEP-EIT-P 服务管理脚本

SERVICES=("enhanced_api_server" "enhanced_api_server_v2" "production_monitor")

case "$1" in
    start)
        echo "🚀 启动所有服务..."
        for service in "${SERVICES[@]}"; do
            echo "启动 $service..."
            if [ "$service" = "enhanced_api_server" ]; then
                ./start_enhanced_production.sh
            elif [ "$service" = "enhanced_api_server_v2" ]; then
                ./start_enhanced_v2.sh
            elif [ "$service" = "production_monitor" ]; then
                nohup python3 production_monitor.py > monitor.log 2>&1 &
            fi
        done
        ;;
    stop)
        echo "⏹️ 停止所有服务..."
        for service in "${SERVICES[@]}"; do
            echo "停止 $service..."
            if [ "$service" = "enhanced_api_server" ]; then
                ./stop_enhanced_production.sh
            elif [ "$service" = "enhanced_api_server_v2" ]; then
                ./stop_enhanced_v2.sh
            elif [ "$service" = "production_monitor" ]; then
                pkill -f "production_monitor"
            fi
        done
        ;;
    restart)
        echo "🔄 重启所有服务..."
        $0 stop
        sleep 5
        $0 start
        ;;
    status)
        echo "📊 服务状态..."
        for service in "${SERVICES[@]}"; do
            if pgrep -f "$service" > /dev/null; then
                echo "✅ $service: 运行中"
            else
                echo "❌ $service: 已停止"
            fi
        done
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
```

```bash
chmod +x manage_services.sh
```

## 监控和告警

### 1. 系统监控

#### 使用内置监控器

```bash
# 启动生产监控器
python3 production_monitor.py
```

#### 监控指标

- **系统指标**: CPU使用率、内存使用率、磁盘使用率
- **API指标**: 响应时间、成功率、请求量
- **模型指标**: 意识水平、推理时间、能量效率

### 2. 日志管理

#### 配置日志轮转

```bash
sudo vim /etc/logrotate.d/eitp
```

```
/var/log/eitp/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 eitp eitp
    postrotate
        systemctl reload eitp-api
    endscript
}
```

#### 日志分析

```bash
# 查看API日志
tail -f /var/log/eitp/enhanced_api_v2.log

# 查看错误日志
grep "ERROR" /var/log/eitp/*.log

# 查看性能日志
grep "inference_time" /var/log/eitp/*.log
```

### 3. 告警配置

#### 邮件告警

```bash
# 安装邮件工具
sudo apt install -y mailutils

# 配置邮件
sudo vim /etc/postfix/main.cf
```

#### 告警规则

```python
# 在production_monitor.py中配置告警阈值
alert_thresholds = {
    'cpu_percent': 80.0,           # CPU使用率告警
    'memory_percent': 85.0,        # 内存使用率告警
    'disk_usage_percent': 90.0,    # 磁盘使用率告警
    'response_time': 5.0,          # 响应时间告警
    'error_rate': 0.1,             # 错误率告警
    'consciousness_level': 1,      # 意识水平告警
    'constraint_satisfaction': 0.3 # 约束满足率告警
}
```

## 性能优化

### 1. 系统优化

#### 内核参数优化

```bash
sudo vim /etc/sysctl.conf
```

```
# 网络优化
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# 文件描述符限制
fs.file-max = 65536

# 内存管理
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
```

```bash
sudo sysctl -p
```

#### 文件系统优化

```bash
# 挂载选项优化
sudo vim /etc/fstab
```

```
/dev/sda1 / ext4 defaults,noatime,nodiratime 0 1
```

### 2. 应用优化

#### Python优化

```bash
# 设置Python环境变量
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
```

#### 模型优化

```python
# 在enhanced_api_server_v2.py中优化模型
def optimize_model():
    # 启用优化模式
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 设置线程数
    torch.set_num_threads(4)
    
    # 启用混合精度
    if torch.cuda.is_available():
        model = model.half()
```

### 3. 数据库优化

#### 如果使用数据库存储历史数据

```python
# 数据库连接池配置
DATABASE_CONFIG = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600
}
```

## 故障排除

### 1. 常见问题

#### 服务无法启动

```bash
# 检查端口占用
lsof -i :5000

# 检查日志
journalctl -u eitp-api -f

# 检查权限
ls -la /opt/eitp/
```

#### 内存不足

```bash
# 检查内存使用
free -h
ps aux --sort=-%mem | head

# 清理内存
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

#### 性能问题

```bash
# 检查CPU使用
top -p $(pgrep -f enhanced_api)

# 检查I/O
iotop -p $(pgrep -f enhanced_api)

# 检查网络
nethogs
```

### 2. 调试工具

#### 性能分析

```bash
# 安装性能分析工具
pip install py-spy memory-profiler

# CPU性能分析
py-spy top --pid $(pgrep -f enhanced_api)

# 内存分析
python -m memory_profiler enhanced_api_server_v2.py
```

#### 网络调试

```bash
# 检查网络连接
netstat -tulpn | grep :5000

# 测试API连接
curl -v http://localhost:5000/api/health

# 压力测试
ab -n 1000 -c 10 http://localhost:5000/api/health
```

## 运维管理

### 1. 日常运维

#### 健康检查脚本

```bash
vim /opt/eitp/eitp-real-product/health_check.sh
```

```bash
#!/bin/bash
# 健康检查脚本

API_URL="http://localhost:5000"
LOG_FILE="/var/log/eitp/health_check.log"

check_api() {
    response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/api/health)
    if [ "$response" = "200" ]; then
        echo "$(date): API健康检查通过" >> $LOG_FILE
        return 0
    else
        echo "$(date): API健康检查失败 (HTTP $response)" >> $LOG_FILE
        return 1
    fi
}

check_system() {
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    
    if (( $(echo "$cpu_usage > 90" | bc -l) )); then
        echo "$(date): CPU使用率过高: $cpu_usage%" >> $LOG_FILE
        return 1
    fi
    
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        echo "$(date): 内存使用率过高: $memory_usage%" >> $LOG_FILE
        return 1
    fi
    
    return 0
}

# 执行检查
if check_api && check_system; then
    echo "系统健康"
    exit 0
else
    echo "系统异常"
    exit 1
fi
```

#### 定时任务

```bash
# 添加定时任务
crontab -e
```

```
# 每5分钟执行健康检查
*/5 * * * * /opt/eitp/eitp-real-product/health_check.sh

# 每天凌晨2点清理日志
0 2 * * * find /var/log/eitp -name "*.log" -mtime +7 -delete

# 每周日凌晨3点重启服务
0 3 * * 0 systemctl restart eitp-api
```

### 2. 备份和恢复

#### 数据备份

```bash
vim /opt/eitp/eitp-real-product/backup.sh
```

```bash
#!/bin/bash
# 备份脚本

BACKUP_DIR="/opt/backups/eitp"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份应用代码
tar -czf $BACKUP_DIR/eitp_code_$DATE.tar.gz /opt/eitp/eitp-real-product

# 备份日志
tar -czf $BACKUP_DIR/eitp_logs_$DATE.tar.gz /var/log/eitp

# 备份配置文件
tar -czf $BACKUP_DIR/eitp_config_$DATE.tar.gz /etc/systemd/system/eitp-*.service

# 清理旧备份（保留30天）
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

#### 恢复流程

```bash
# 停止服务
systemctl stop eitp-api eitp-monitor

# 恢复代码
tar -xzf /opt/backups/eitp/eitp_code_YYYYMMDD_HHMMSS.tar.gz -C /

# 恢复配置
tar -xzf /opt/backups/eitp/eitp_config_YYYYMMDD_HHMMSS.tar.gz -C /

# 重新加载配置
systemctl daemon-reload

# 启动服务
systemctl start eitp-api eitp-monitor
```

## 安全配置

### 1. 网络安全

#### 防火墙配置

```bash
# 安装UFW
sudo apt install -y ufw

# 配置防火墙规则
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 5000/tcp  # 禁止直接访问API端口

# 启用防火墙
sudo ufw enable
```

#### SSL/TLS配置

```bash
# 安装Certbot
sudo apt install -y certbot python3-certbot-nginx

# 获取SSL证书
sudo certbot --nginx -d your-domain.com

# 自动续期
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

### 2. 应用安全

#### API认证

```python
# 在enhanced_api_server_v2.py中添加认证
from functools import wraps
import jwt

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': '缺少认证令牌'}), 401
        
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except:
            return jsonify({'error': '无效的认证令牌'}), 401
        
        return f(*args, **kwargs)
    return decorated_function
```

#### 访问控制

```bash
# 配置Nginx访问控制
sudo vim /etc/nginx/sites-available/eitp
```

```nginx
# 限制访问频率
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    # 限制请求频率
    limit_req zone=api burst=20 nodelay;
    
    # 限制连接数
    limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;
    limit_conn conn_limit_per_ip 10;
    
    # 隐藏敏感信息
    server_tokens off;
    
    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
}
```

## 监控仪表板

### 1. 创建监控仪表板

```python
# 创建Grafana仪表板配置
dashboard_config = {
    "dashboard": {
        "title": "Enhanced CEP-EIT-P Monitoring",
        "panels": [
            {
                "title": "API响应时间",
                "type": "graph",
                "targets": [
                    {
                        "expr": "avg(api_response_time)",
                        "legendFormat": "平均响应时间"
                    }
                ]
            },
            {
                "title": "意识水平分布",
                "type": "piechart",
                "targets": [
                    {
                        "expr": "consciousness_level",
                        "legendFormat": "意识水平"
                    }
                ]
            }
        ]
    }
}
```

### 2. 告警规则

```yaml
# Prometheus告警规则
groups:
- name: eitp_alerts
  rules:
  - alert: HighCPUUsage
    expr: cpu_usage > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "CPU使用率过高"
      
  - alert: HighMemoryUsage
    expr: memory_usage > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "内存使用率过高"
```

## 总结

本指南提供了Enhanced CEP-EIT-P框架的完整部署和运维方案，包括：

1. **系统要求**: 详细的硬件和软件要求
2. **部署准备**: 环境配置和依赖安装
3. **生产部署**: systemd服务配置和Nginx反向代理
4. **监控告警**: 全面的监控和告警系统
5. **性能优化**: 系统和应用层面的优化
6. **故障排除**: 常见问题的诊断和解决
7. **运维管理**: 日常运维和自动化脚本
8. **安全配置**: 网络安全和应用安全
9. **备份恢复**: 数据备份和灾难恢复

通过遵循本指南，可以确保Enhanced CEP-EIT-P框架在生产环境中的稳定运行和高性能表现。
