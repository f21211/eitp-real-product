"""
集成测试
测试EIT-P各组件的集成功能
"""

import unittest
import torch
import tempfile
import os
import time
from pathlib import Path
from eit_p.utils.config_manager import ConfigManager
from eit_p.utils.exceptions import ErrorHandler, MemoryOverflowError
from eit_p.utils.logger import EITPLogger, TrainingMonitor


class TestIntegration(unittest.TestCase):
    """集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试配置
        self.config = {
            'training': {
                'batch_size': 1,
                'learning_rate': 5e-5
            },
            'model': {
                'name': 'gpt2',
                'block_size': 16
            },
            'hypernetwork': {
                'input_dim': 1539,
                'hidden_dim': 8,
                'output_dim': 1
            },
            'memory': {
                'max_gpu_usage': 3.0,
                'cleanup_interval': 5
            },
            'loss_weights': {
                'ce_weight': 1.0,
                'path_norm_weight': 1.0
            },
            'regularization': {
                'target_fractal_dim': 2.7,
                'path_norm_weight': 1.0
            },
            'logging': {
                'level': 'INFO',
                'file': str(Path(self.temp_dir) / 'test.log')
            }
        }
        
        # 写入配置文件
        self.config_file = Path(self.temp_dir) / 'test_config.yaml'
        import yaml
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f)
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_config_logger_integration(self):
        """测试配置和日志系统集成"""
        # 创建配置管理器
        config_manager = ConfigManager(str(self.config_file))
        
        # 创建日志器
        logger = EITPLogger('test', config_manager.get_logging_config())
        
        # 测试日志记录
        logger.info("测试信息日志")
        logger.warning("测试警告日志")
        logger.error("测试错误日志")
        
        # 验证日志文件创建
        log_file = Path(self.temp_dir) / 'test.log'
        self.assertTrue(log_file.exists(), "日志文件应该被创建")
        
        # 验证日志内容
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            self.assertIn("测试信息日志", log_content)
            self.assertIn("测试警告日志", log_content)
            self.assertIn("测试错误日志", log_content)
    
    def test_error_handler_logger_integration(self):
        """测试错误处理器和日志系统集成"""
        # 创建日志器
        logger = EITPLogger('test')
        
        # 创建错误处理器
        error_handler = ErrorHandler(logger)
        
        # 测试异常处理
        exception = MemoryOverflowError("GPU", 4.0, 3.0)
        error_handler.handle_exception(exception, "测试上下文")
        
        # 验证错误计数
        error_summary = error_handler.get_error_summary()
        self.assertEqual(error_summary['MemoryOverflowError'], 1)
    
    def test_training_monitor_integration(self):
        """测试训练监控器集成"""
        # 创建日志器
        logger = EITPLogger('test')
        
        # 创建训练监控器
        monitor = TrainingMonitor(logger)
        
        # 模拟训练步骤
        for step in range(10):
            loss = 100.0 - step * 5.0  # 模拟损失下降
            monitor.log_step(step, loss, accuracy=0.8 + step * 0.02)
        
        # 验证监控器状态
        self.assertEqual(monitor.step_count, 9)  # 最后一步
        self.assertEqual(len(monitor.loss_history), 10)
        
        # 测试内存清理日志
        monitor.log_memory_cleanup(2.0, 1.5)
        
        # 测试错误日志
        exception = Exception("测试异常")
        monitor.log_error(exception, "测试上下文")
    
    def test_config_validation_integration(self):
        """测试配置验证集成"""
        # 创建配置管理器
        config_manager = ConfigManager(str(self.config_file))
        
        # 验证有效配置
        self.assertTrue(config_manager.validate_config())
        
        # 测试无效配置
        config_manager.set('training.batch_size', -1)
        self.assertFalse(config_manager.validate_config())
        
        # 测试缺失配置节
        config_manager._config.pop('training')
        self.assertFalse(config_manager.validate_config())
    
    def test_memory_management_integration(self):
        """测试内存管理集成"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA不可用，跳过GPU内存测试")
        
        # 创建配置管理器
        config_manager = ConfigManager(str(self.config_file))
        
        # 获取内存配置
        memory_config = config_manager.get_memory_config()
        self.assertEqual(memory_config['max_gpu_usage'], 3.0)
        self.assertEqual(memory_config['cleanup_interval'], 5)
        
        # 测试GPU内存检查
        from eit_p.utils.exceptions import check_gpu_memory
        memory_ok = check_gpu_memory(threshold=0.9)
        self.assertIsInstance(memory_ok, bool)
    
    def test_logging_configuration_integration(self):
        """测试日志配置集成"""
        # 创建配置管理器
        config_manager = ConfigManager(str(self.config_file))
        
        # 获取日志配置
        logging_config = config_manager.get_logging_config()
        
        # 验证日志配置
        self.assertEqual(logging_config['level'], 'INFO')
        self.assertIn('test.log', logging_config['file'])
        
        # 创建日志器
        logger = EITPLogger('test', logging_config)
        
        # 测试不同级别的日志
        logger.debug("调试日志")
        logger.info("信息日志")
        logger.warning("警告日志")
        logger.error("错误日志")
        logger.critical("严重错误日志")
    
    def test_error_recovery_integration(self):
        """测试错误恢复集成"""
        # 创建日志器
        logger = EITPLogger('test')
        
        # 创建错误处理器
        error_handler = ErrorHandler(logger)
        
        # 模拟多次错误
        for i in range(5):
            exception = MemoryOverflowError("GPU", 4.0 + i, 3.0)
            error_handler.handle_exception(exception, f"测试上下文 {i}")
        
        # 验证错误计数
        error_summary = error_handler.get_error_summary()
        self.assertEqual(error_summary['MemoryOverflowError'], 5)
        
        # 重置错误计数
        error_handler.reset_error_count()
        error_summary = error_handler.get_error_summary()
        self.assertEqual(len(error_summary), 0)
    
    def test_performance_monitoring_integration(self):
        """测试性能监控集成"""
        # 创建日志器
        logger = EITPLogger('test')
        
        # 创建训练监控器
        monitor = TrainingMonitor(logger)
        
        # 模拟长时间训练
        start_time = time.time()
        for step in range(100):
            loss = 100.0 * (0.99 ** step)  # 模拟指数衰减
            monitor.log_step(step, loss, 
                           learning_rate=5e-5,
                           gradient_norm=1.0)
        
        # 验证监控器状态
        self.assertEqual(monitor.step_count, 99)
        self.assertEqual(len(monitor.loss_history), 100)
        
        # 验证损失历史
        self.assertLess(monitor.loss_history[-1], monitor.loss_history[0])
        
        # 验证内存历史（如果有GPU）
        if torch.cuda.is_available():
            self.assertGreater(len(monitor.memory_history), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
