#!/usr/bin/env python3
"""
EIT-P 模型管理服务
提供模型上传、下载、版本管理、A/B测试和模型压缩功能
"""

import os
import sys
import json
import time
import hashlib
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import wraps

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, request, jsonify, send_file, g
    from flask_cors import CORS
    import torch
    import yaml
    from werkzeug.utils import secure_filename
    import sqlite3
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install flask flask-cors torch pyyaml")
    sys.exit(1)

from eit_p.utils import get_global_logger


class ModelManager:
    """模型管理器"""
    
    def __init__(self, models_dir: str = "models", db_path: str = "data/models.db"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.db_path = db_path
        self.logger = get_global_logger()
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建模型表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                version TEXT NOT NULL,
                model_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                file_hash TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        ''')
        
        # 创建模型版本表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                file_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_current BOOLEAN DEFAULT 0,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        ''')
        
        # 创建A/B测试表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                model_a_id TEXT NOT NULL,
                model_b_id TEXT NOT NULL,
                traffic_split REAL DEFAULT 0.5,
                status TEXT NOT NULL DEFAULT 'active',
                start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_date TIMESTAMP,
                results TEXT DEFAULT '{}',
                FOREIGN KEY (model_a_id) REFERENCES models (model_id),
                FOREIGN KEY (model_b_id) REFERENCES models (model_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def upload_model(self, file, model_id: str, name: str, description: str, 
                    model_type: str, version: str = "1.0.0", created_by: str = "system",
                    tags: List[str] = None, metadata: Dict[str, Any] = None):
        """上传模型"""
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}
        
        # 保存文件
        filename = secure_filename(file.filename)
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        file_path = model_dir / f"{version}_{filename}"
        file.save(str(file_path))
        
        # 计算文件信息
        file_size = file_path.stat().st_size
        file_hash = self._calculate_file_hash(file_path)
        
        # 保存到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO models (model_id, name, description, version, model_type, 
                                  file_path, file_size, file_hash, created_by, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (model_id, name, description, version, model_type, 
                  str(file_path), file_size, file_hash, created_by, 
                  json.dumps(tags), json.dumps(metadata)))
            
            # 保存版本信息
            cursor.execute('''
                INSERT INTO model_versions (model_id, version, file_path, file_size, file_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_id, version, str(file_path), file_size, file_hash))
            
            conn.commit()
            
            self.logger.info(f"模型上传成功: {model_id} v{version}")
            return True
            
        except sqlite3.IntegrityError as e:
            if 'UNIQUE constraint failed' in str(e):
                raise ValueError("模型ID已存在")
            raise
        finally:
            conn.close()
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def list_models(self, status: str = None, model_type: str = None) -> List[Dict[str, Any]]:
        """列出模型"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT model_id, name, description, version, model_type, file_size, 
                   status, created_at, updated_at, created_by, tags, metadata
            FROM models WHERE 1=1
        '''
        params = []
        
        if status:
            query += ' AND status = ?'
            params.append(status)
        
        if model_type:
            query += ' AND model_type = ?'
            params.append(model_type)
        
        query += ' ORDER BY created_at DESC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        models = []
        for row in rows:
            models.append({
                'model_id': row[0],
                'name': row[1],
                'description': row[2],
                'version': row[3],
                'model_type': row[4],
                'file_size': row[5],
                'status': row[6],
                'created_at': row[7],
                'updated_at': row[8],
                'created_by': row[9],
                'tags': json.loads(row[10]),
                'metadata': json.loads(row[11])
            })
        
        return models
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型详情"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT model_id, name, description, version, model_type, file_path, 
                   file_size, file_hash, status, created_at, updated_at, 
                   created_by, tags, metadata
            FROM models WHERE model_id = ?
        ''', (model_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'model_id': row[0],
                'name': row[1],
                'description': row[2],
                'version': row[3],
                'model_type': row[4],
                'file_path': row[5],
                'file_size': row[6],
                'file_hash': row[7],
                'status': row[8],
                'created_at': row[9],
                'updated_at': row[10],
                'created_by': row[11],
                'tags': json.loads(row[12]),
                'metadata': json.loads(row[13])
            }
        return None
    
    def download_model(self, model_id: str, version: str = None) -> Optional[Path]:
        """下载模型文件"""
        model = self.get_model(model_id)
        if not model:
            return None
        
        if version:
            # 获取指定版本
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_path FROM model_versions 
                WHERE model_id = ? AND version = ?
            ''', (model_id, version))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return Path(row[0])
        else:
            # 获取当前版本
            return Path(model['file_path'])
        
        return None
    
    def update_model_status(self, model_id: str, status: str) -> bool:
        """更新模型状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE models SET status = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE model_id = ?
        ''', (status, model_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            self.logger.info(f"模型状态更新成功: {model_id} -> {status}")
        
        return success
    
    def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        model = self.get_model(model_id)
        if not model:
            return False
        
        # 删除文件
        model_dir = self.models_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # 删除数据库记录
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM models WHERE model_id = ?', (model_id,))
        cursor.execute('DELETE FROM model_versions WHERE model_id = ?', (model_id,))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            self.logger.info(f"模型删除成功: {model_id}")
        
        return success
    
    def create_ab_test(self, test_name: str, model_a_id: str, model_b_id: str, 
                      traffic_split: float = 0.5) -> bool:
        """创建A/B测试"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO ab_tests (test_name, model_a_id, model_b_id, traffic_split)
                VALUES (?, ?, ?, ?)
            ''', (test_name, model_a_id, model_b_id, traffic_split))
            
            conn.commit()
            self.logger.info(f"A/B测试创建成功: {test_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"A/B测试创建失败: {e}")
            return False
        finally:
            conn.close()
    
    def get_ab_tests(self) -> List[Dict[str, Any]]:
        """获取A/B测试列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, test_name, model_a_id, model_b_id, traffic_split, 
                   status, start_date, end_date, results
            FROM ab_tests ORDER BY start_date DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        tests = []
        for row in rows:
            tests.append({
                'id': row[0],
                'test_name': row[1],
                'model_a_id': row[2],
                'model_b_id': row[3],
                'traffic_split': row[4],
                'status': row[5],
                'start_date': row[6],
                'end_date': row[7],
                'results': json.loads(row[8]) if row[8] else {}
            })
        
        return tests


class ModelManagementService:
    """模型管理服务"""
    
    def __init__(self, host='0.0.0.0', port=8090, config_path=None):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.host = host
        self.port = port
        self.logger = get_global_logger()
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化模型管理器
        self.model_manager = ModelManager(
            models_dir=self.config.get('models_dir', 'models'),
            db_path=self.config.get('db_path', 'data/models.db')
        )
        
        # 设置路由
        self._setup_routes()
    
    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return {
                'models_dir': 'models',
                'db_path': 'data/models.db',
                'max_file_size': 1024 * 1024 * 1024,  # 1GB
                'allowed_extensions': {'.pt', '.pth', '.bin', '.safetensors'}
            }
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'service': 'EIT-P Model Management',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/models', methods=['GET'])
        def list_models():
            """获取模型列表"""
            try:
                status = request.args.get('status')
                model_type = request.args.get('model_type')
                
                models = self.model_manager.list_models(status, model_type)
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'models': models,
                        'total_count': len(models)
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/models', methods=['POST'])
        def upload_model():
            """上传模型"""
            try:
                if 'file' not in request.files:
                    return jsonify({'status': 'error', 'message': '没有文件'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'status': 'error', 'message': '没有选择文件'}), 400
                
                # 获取参数
                model_id = request.form.get('model_id')
                name = request.form.get('name')
                description = request.form.get('description', '')
                model_type = request.form.get('model_type')
                version = request.form.get('version', '1.0.0')
                created_by = request.form.get('created_by', 'system')
                
                if not all([model_id, name, model_type]):
                    return jsonify({'status': 'error', 'message': '缺少必需参数'}), 400
                
                # 检查文件扩展名
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in self.config.get('allowed_extensions', {'.pt', '.pth'}):
                    return jsonify({'status': 'error', 'message': '不支持的文件类型'}), 400
                
                # 解析标签和元数据
                tags = json.loads(request.form.get('tags', '[]'))
                metadata = json.loads(request.form.get('metadata', '{}'))
                
                # 上传模型
                success = self.model_manager.upload_model(
                    file, model_id, name, description, model_type, 
                    version, created_by, tags, metadata
                )
                
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': '模型上传成功',
                        'data': {'model_id': model_id, 'version': version}
                    })
                else:
                    return jsonify({'status': 'error', 'message': '模型上传失败'}), 500
                    
            except ValueError as e:
                return jsonify({'status': 'error', 'message': str(e)}), 400
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/models/<model_id>', methods=['GET'])
        def get_model(model_id):
            """获取模型详情"""
            try:
                model = self.model_manager.get_model(model_id)
                if not model:
                    return jsonify({'status': 'error', 'message': '模型不存在'}), 404
                
                return jsonify({
                    'status': 'success',
                    'data': model
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/models/<model_id>/download', methods=['GET'])
        def download_model(model_id):
            """下载模型"""
            try:
                version = request.args.get('version')
                file_path = self.model_manager.download_model(model_id, version)
                
                if not file_path or not file_path.exists():
                    return jsonify({'status': 'error', 'message': '模型文件不存在'}), 404
                
                return send_file(str(file_path), as_attachment=True)
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/models/<model_id>/status', methods=['PUT'])
        def update_model_status(model_id):
            """更新模型状态"""
            try:
                data = request.get_json()
                status = data.get('status')
                
                if not status:
                    return jsonify({'status': 'error', 'message': '缺少状态字段'}), 400
                
                success = self.model_manager.update_model_status(model_id, status)
                if success:
                    return jsonify({'status': 'success', 'message': '状态更新成功'})
                else:
                    return jsonify({'status': 'error', 'message': '模型不存在'}), 404
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/models/<model_id>', methods=['DELETE'])
        def delete_model(model_id):
            """删除模型"""
            try:
                success = self.model_manager.delete_model(model_id)
                if success:
                    return jsonify({'status': 'success', 'message': '模型删除成功'})
                else:
                    return jsonify({'status': 'error', 'message': '模型不存在'}), 404
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/ab-tests', methods=['GET'])
        def get_ab_tests():
            """获取A/B测试列表"""
            try:
                tests = self.model_manager.get_ab_tests()
                return jsonify({
                    'status': 'success',
                    'data': {
                        'tests': tests,
                        'total_count': len(tests)
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/ab-tests', methods=['POST'])
        def create_ab_test():
            """创建A/B测试"""
            try:
                data = request.get_json()
                test_name = data.get('test_name')
                model_a_id = data.get('model_a_id')
                model_b_id = data.get('model_b_id')
                traffic_split = data.get('traffic_split', 0.5)
                
                if not all([test_name, model_a_id, model_b_id]):
                    return jsonify({'status': 'error', 'message': '缺少必需参数'}), 400
                
                success = self.model_manager.create_ab_test(
                    test_name, model_a_id, model_b_id, traffic_split
                )
                
                if success:
                    return jsonify({'status': 'success', 'message': 'A/B测试创建成功'})
                else:
                    return jsonify({'status': 'error', 'message': 'A/B测试创建失败'}), 500
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def run(self, debug=False):
        """运行模型管理服务"""
        self.logger.info(f"启动EIT-P模型管理服务: http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)


def main():
    """主函数"""
    print("🚀 启动EIT-P模型管理服务...")
    
    # 检查配置文件
    config_path = project_root / "config" / "production.yaml"
    if not config_path.exists():
        print(f"警告: 配置文件不存在 {config_path}，使用默认配置")
        config_path = None
    
    # 启动模型管理服务
    service = ModelManagementService(config_path=str(config_path) if config_path else None)
    service.run(debug=False)


if __name__ == "__main__":
    main()