#!/usr/bin/env python3
"""
EIT-P 模型管理测试脚本
"""

import requests
import json
import time
import os

def test_model_management():
    """测试模型管理服务"""
    base_url = 'http://localhost:8089'
    
    print('🧪 测试EIT-P模型管理服务...')
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
    
    # 测试获取模型列表
    print('📋 测试获取模型列表...')
    try:
        response = requests.get(f'{base_url}/api/models')
        if response.status_code == 200:
            print('✅ 获取模型列表测试通过')
            result = response.json()
            print(f'  模型总数: {result["count"]}')
        else:
            print(f'❌ 获取模型列表测试失败: {response.status_code}')
    except Exception as e:
        print(f'❌ 获取模型列表测试错误: {e}')
    
    print()
    
    # 测试创建A/B测试
    print('📋 测试创建A/B测试...')
    try:
        data = {
            'name': '测试A/B测试',
            'model_a_id': 'model_a_test',
            'model_b_id': 'model_b_test',
            'traffic_split': 0.5
        }
        response = requests.post(f'{base_url}/api/ab-tests', json=data)
        if response.status_code == 200:
            print('✅ 创建A/B测试测试通过')
            result = response.json()
            test_id = result['data']['test_id']
            print(f'  测试ID: {test_id}')
        else:
            print(f'❌ 创建A/B测试测试失败: {response.status_code}')
    except Exception as e:
        print(f'❌ 创建A/B测试测试错误: {e}')
    
    print()
    
    # 测试获取A/B测试列表
    print('📋 测试获取A/B测试列表...')
    try:
        response = requests.get(f'{base_url}/api/ab-tests')
        if response.status_code == 200:
            print('✅ 获取A/B测试列表测试通过')
            result = response.json()
            print(f'  A/B测试总数: {result["count"]}')
        else:
            print(f'❌ 获取A/B测试列表测试失败: {response.status_code}')
    except Exception as e:
        print(f'❌ 获取A/B测试列表测试错误: {e}')
    
    print()
    print('🎉 模型管理服务测试完成！')

if __name__ == '__main__':
    test_model_management()
