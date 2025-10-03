#!/usr/bin/env python3
"""
EIT-P 认证服务测试脚本
"""

import requests
import json
import time

def test_auth_service():
    """测试认证服务"""
    base_url = 'http://localhost:8087'
    
    print('🧪 测试EIT-P认证服务...')
    print('=' * 80)
    
    # 测试健康检查
    print('📋 测试健康检查...')
    try:
        response = requests.get(f'{base_url}/health')
        if response.status_code == 200:
            print('✅ 健康检查通过')
            print(f'  响应: {response.json()}')
        else:
            print(f'❌ 健康检查失败: {response.status_code}')
    except Exception as e:
        print(f'❌ 健康检查错误: {e}')
    
    print()
    
    # 测试用户注册
    print('📋 测试用户注册...')
    try:
        data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpass123',
            'role': 'user'
        }
        response = requests.post(f'{base_url}/api/auth/register', json=data)
        if response.status_code == 200:
            print('✅ 用户注册测试通过')
            result = response.json()
            print(f'  用户ID: {result["data"]["user_id"]}')
            print(f'  用户名: {result["data"]["username"]}')
        else:
            print(f'❌ 用户注册测试失败: {response.status_code}')
            print(f'  响应: {response.json()}')
    except Exception as e:
        print(f'❌ 用户注册测试错误: {e}')
    
    print()
    
    # 测试用户登录
    print('📋 测试用户登录...')
    token = None
    try:
        data = {
            'username': 'testuser',
            'password': 'testpass123'
        }
        response = requests.post(f'{base_url}/api/auth/login', json=data)
        if response.status_code == 200:
            print('✅ 用户登录测试通过')
            result = response.json()
            token = result['data']['token']
            print(f'  令牌: {token[:20]}...')
            print(f'  会话ID: {result["data"]["session_id"]}')
        else:
            print(f'❌ 用户登录测试失败: {response.status_code}')
            print(f'  响应: {response.json()}')
    except Exception as e:
        print(f'❌ 用户登录测试错误: {e}')
    
    print()
    
    # 测试令牌验证
    if token:
        print('📋 测试令牌验证...')
        try:
            data = {'token': token}
            response = requests.post(f'{base_url}/api/auth/verify', json=data)
            if response.status_code == 200:
                print('✅ 令牌验证测试通过')
                result = response.json()
                print(f'  用户: {result["data"]["username"]}')
                print(f'  角色: {result["data"]["role"]}')
                print(f'  权限: {result["data"]["permissions"]}')
            else:
                print(f'❌ 令牌验证测试失败: {response.status_code}')
        except Exception as e:
            print(f'❌ 令牌验证测试错误: {e}')
    
    print()
    
    # 测试管理员登录
    print('📋 测试管理员登录...')
    admin_token = None
    try:
        data = {
            'username': 'admin',
            'password': 'admin123'
        }
        response = requests.post(f'{base_url}/api/auth/login', json=data)
        if response.status_code == 200:
            print('✅ 管理员登录测试通过')
            result = response.json()
            admin_token = result['data']['token']
            print(f'  管理员令牌: {admin_token[:20]}...')
        else:
            print(f'❌ 管理员登录测试失败: {response.status_code}')
    except Exception as e:
        print(f'❌ 管理员登录测试错误: {e}')
    
    print()
    
    # 测试用户列表（需要管理员权限）
    if admin_token:
        print('📋 测试用户列表...')
        try:
            headers = {'Authorization': f'Bearer {admin_token}'}
            response = requests.get(f'{base_url}/api/auth/users', headers=headers)
            if response.status_code == 200:
                print('✅ 用户列表测试通过')
                result = response.json()
                print(f'  用户总数: {result["count"]}')
                for user in result['data'][:3]:  # 显示前3个用户
                    print(f'    - {user["username"]} ({user["role"]})')
            else:
                print(f'❌ 用户列表测试失败: {response.status_code}')
        except Exception as e:
            print(f'❌ 用户列表测试错误: {e}')
    
    print()
    print('🎉 认证服务测试完成！')

if __name__ == '__main__':
    test_auth_service()
