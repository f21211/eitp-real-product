#!/usr/bin/env python3
"""
EIT-P 推理服务测试脚本
"""

import requests
import json
import time

def test_inference_service():
    """测试推理服务"""
    base_url = 'http://localhost:8086'
    
    print('🧪 测试EIT-P推理服务...')
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
    
    # 测试文本生成
    print('�� 测试文本生成...')
    try:
        data = {
            'text': '人工智能的未来',
            'max_length': 50,
            'temperature': 0.7
        }
        response = requests.post(f'{base_url}/api/inference/text_generation', json=data)
        if response.status_code == 200:
            print('✅ 文本生成测试通过')
            result = response.json()
            print(f'  输入: {result["data"]["input_text"]}')
            print(f'  输出: {result["data"]["generated_text"]}')
        else:
            print(f'❌ 文本生成测试失败: {response.status_code}')
    except Exception as e:
        print(f'❌ 文本生成测试错误: {e}')
    
    print()
    
    # 测试文本分类
    print('📋 测试文本分类...')
    try:
        data = {
            'text': '这是一个很好的产品',
            'labels': ['positive', 'negative', 'neutral']
        }
        response = requests.post(f'{base_url}/api/inference/classification', json=data)
        if response.status_code == 200:
            print('✅ 文本分类测试通过')
            result = response.json()
            print(f'  输入: {result["data"]["input_text"]}')
            print(f'  预测: {result["data"]["predictions"]}')
        else:
            print(f'❌ 文本分类测试失败: {response.status_code}')
    except Exception as e:
        print(f'❌ 文本分类测试错误: {e}')
    
    print()
    
    # 测试问答
    print('📋 测试问答...')
    try:
        data = {
            'question': '什么是人工智能？',
            'context': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。'
        }
        response = requests.post(f'{base_url}/api/inference/qa', json=data)
        if response.status_code == 200:
            print('✅ 问答测试通过')
            result = response.json()
            print(f'  问题: {result["data"]["question"]}')
            print(f'  回答: {result["data"]["answer"]}')
        else:
            print(f'❌ 问答测试失败: {response.status_code}')
    except Exception as e:
        print(f'❌ 问答测试错误: {e}')
    
    print()
    
    # 测试文本嵌入
    print('📋 测试文本嵌入...')
    try:
        data = {
            'text': '测试文本嵌入功能'
        }
        response = requests.post(f'{base_url}/api/inference/embedding', json=data)
        if response.status_code == 200:
            print('✅ 文本嵌入测试通过')
            result = response.json()
            print(f'  输入: {result["data"]["input_text"]}')
            print(f'  维度: {result["data"]["dimension"]}')
            print(f'  嵌入: {result["data"]["embedding"][:5]}...')
        else:
            print(f'❌ 文本嵌入测试失败: {response.status_code}')
    except Exception as e:
        print(f'❌ 文本嵌入测试错误: {e}')
    
    print()
    print('🎉 推理服务测试完成！')

if __name__ == '__main__':
    test_inference_service()
