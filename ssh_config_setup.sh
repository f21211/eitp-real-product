#!/bin/bash

# 配置SSH使用f21211用户的密钥
echo "🔑 配置SSH使用f21211用户的密钥..."

# 创建SSH config文件
cat > ~/.ssh/config << 'EOF'
# f21211 GitHub配置
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_f21211
    IdentitiesOnly yes

# 默认配置
Host *
    AddKeysToAgent yes
    IdentitiesOnly yes
EOF

echo "✅ SSH配置完成！"
echo "📋 请将以下公钥添加到GitHub账户 f21211:"
echo ""
cat ~/.ssh/id_ed25519_f21211.pub
echo ""
echo "🔗 添加步骤："
echo "1. 访问 https://github.com/settings/keys"
echo "2. 点击 'New SSH key'"
echo "3. 复制上面的公钥内容"
echo "4. 粘贴到 'Key' 字段"
echo "5. 点击 'Add SSH key'"
echo ""
echo "完成后运行: git push origin main"
