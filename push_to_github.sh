#!/bin/bash

# EIT-P GitHub推送脚本
echo "🚀 开始推送EIT-P项目到GitHub..."

# 设置代理
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897

# 配置Git
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
git config --global http.sslVerify true
git config --global http.version HTTP/2
git config --global http.postBuffer 1048576000

# 检查网络连接
echo "📡 检查网络连接..."
curl -s --connect-timeout 10 https://api.github.com > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ 网络连接正常"
else
    echo "❌ 网络连接失败"
    exit 1
fi

# 检查Git状态
echo "📋 检查Git状态..."
git status

# 尝试推送
echo "📤 开始推送..."
for i in {1..3}; do
    echo "🔄 尝试第 $i 次推送..."
    if git push origin main; then
        echo "✅ 推送成功！"
        exit 0
    else
        echo "❌ 第 $i 次推送失败，等待5秒后重试..."
        sleep 5
    fi
done

echo "❌ 所有推送尝试都失败了"
echo "💡 建议："
echo "   1. 检查网络连接"
echo "   2. 检查GitHub仓库权限"
echo "   3. 尝试使用SSH方式"
echo "   4. 使用备份文件手动上传"

exit 1
