#!/bin/bash

# EIT-P GitHub推送监控脚本
echo "🚀 开始监控EIT-P项目GitHub推送..."

# 设置代理
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897

# 配置Git
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
git config --global http.sslVerify true
git config --global http.version HTTP/1.1
git config --global http.postBuffer 1048576000

# 监控循环
while true; do
    echo "$(date): 检查网络连接..."
    
    # 检查网络连接
    if curl -s --connect-timeout 10 https://api.github.com > /dev/null; then
        echo "$(date): ✅ 网络正常，尝试推送..."
        
        # 尝试推送
        if git push origin main; then
            echo "$(date): 🎉 推送成功！EIT-P项目已成功推送到GitHub！"
            break
        else
            echo "$(date): ❌ 推送失败，等待60秒后重试..."
        fi
    else
        echo "$(date): ❌ 网络连接失败，等待60秒后重试..."
    fi
    
    # 等待60秒
    sleep 60
done

echo "✅ 监控脚本完成！"
