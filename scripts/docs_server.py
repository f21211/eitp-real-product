#!/usr/bin/env python3
"""
EIT-P API文档服务器
提供完整的API文档、交互式测试界面和用户指南
"""

import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, render_template_string, jsonify, request
    from flask_cors import CORS
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install flask flask-cors pyyaml")
    sys.exit(1)

from eit_p.utils import get_global_logger


class DocsServer:
    """文档服务器"""
    
    def __init__(self, host='0.0.0.0', port=8088, config_path=None):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.host = host
        self.port = port
        self.logger = get_global_logger()
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设置路由
        self._setup_routes()
    
    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return {
                'services': {
                    'api_server': {'port': 8085},
                    'inference_service': {'port': 8086},
                    'auth_service': {'port': 8087}
                }
            }
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'service': 'EIT-P Docs Service',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/', methods=['GET'])
        def index():
            """文档首页"""
            return render_template_string(self._get_docs_template())
        
        @self.app.route('/api/docs', methods=['GET'])
        def get_api_docs():
            """获取API文档JSON"""
            return jsonify(self._generate_api_docs())
    
    def _generate_api_docs(self):
        """生成API文档数据"""
        return {
            'info': {
                'title': 'EIT-P API Documentation',
                'version': '2.0.0',
                'description': 'EIT-P (Emergent Intelligence Theory - PyTorch) 完整API文档'
            },
            'servers': [
                {
                    'url': f'http://localhost:{self.config["services"]["api_server"]["port"]}',
                    'description': 'API服务器'
                },
                {
                    'url': f'http://localhost:{self.config["services"]["inference_service"]["port"]}',
                    'description': '推理服务'
                },
                {
                    'url': f'http://localhost:{self.config["services"]["auth_service"]["port"]}',
                    'description': '认证服务'
                }
            ],
            'paths': {
                '/api/auth/login': {
                    'post': {
                        'tags': ['认证'],
                        'summary': '用户登录',
                        'requestBody': {
                            'required': True,
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'object',
                                        'properties': {
                                            'username': {'type': 'string'},
                                            'password': {'type': 'string'}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                '/api/inference/text_generation': {
                    'post': {
                        'tags': ['推理'],
                        'summary': '文本生成',
                        'requestBody': {
                            'required': True,
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'object',
                                        'properties': {
                                            'model_id': {'type': 'string'},
                                            'text': {'type': 'string'}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def _get_docs_template(self):
        """获取文档模板"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EIT-P API 文档</title>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css" />
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
        .header h1 { margin: 0; font-size: 2.5em; }
        .swagger-ui { max-width: 1200px; margin: 0 auto; padding: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 EIT-P API 文档</h1>
        <p>Emergent Intelligence Theory - PyTorch 完整API文档</p>
    </div>
    <div class="swagger-ui" id="swagger-ui"></div>
    <script>
        SwaggerUIBundle({
            url: '/api/docs',
            dom_id: '#swagger-ui',
            presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.presets.standalone],
            layout: "StandaloneLayout"
        });
    </script>
</body>
</html>
        """
    
    def run(self, debug=False):
        """运行文档服务"""
        self.logger.info(f"启动EIT-P文档服务: http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)


def main():
    """主函数"""
    print("🚀 启动EIT-P文档服务...")
    
    # 检查配置文件
    config_path = project_root / "config" / "production.yaml"
    if not config_path.exists():
        print(f"警告: 配置文件不存在 {config_path}，使用默认配置")
        config_path = None
    
    # 启动文档服务
    server = DocsServer(config_path=str(config_path) if config_path else None)
    server.run(debug=False)


if __name__ == "__main__":
    main()