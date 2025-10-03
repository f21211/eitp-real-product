"""
异常处理测试
"""

import unittest
import torch
from eit_p.utils.exceptions import (
    EITPException, MemoryOverflowError, ConvergenceError, 
    ConfigurationError, ModelError, TrainingError, 
    EvaluationError, HardwareError, ErrorHandler,
    check_gpu_memory, check_convergence
)


class TestEITPExceptions(unittest.TestCase):
    """EIT-P异常测试类"""
    
    def test_base_exception(self):
        """测试基础异常类"""
        exception = EITPException("测试消息", "TEST_ERROR", {"key": "value"})
        
        self.assertEqual(exception.message, "测试消息")
        self.assertEqual(exception.error_code, "TEST_ERROR")
        self.assertEqual(exception.details, {"key": "value"})
        self.assertIn("TEST_ERROR", str(exception))
        
        # 测试字典转换
        exception_dict = exception.to_dict()
        self.assertEqual(exception_dict['error_type'], 'EITPException')
        self.assertEqual(exception_dict['message'], "测试消息")
    
    def test_memory_overflow_error(self):
        """测试内存溢出异常"""
        exception = MemoryOverflowError("GPU", 4.0, 3.0)
        
        self.assertEqual(exception.memory_type, "GPU")
        self.assertEqual(exception.current_usage, 4.0)
        self.assertEqual(exception.max_usage, 3.0)
        self.assertIn("内存溢出", str(exception))
    
    def test_convergence_error(self):
        """测试收敛异常"""
        exception = ConvergenceError(1500.0, 1000.0)
        
        self.assertEqual(exception.loss_value, 1500.0)
        self.assertEqual(exception.threshold, 1000.0)
        self.assertIn("损失值异常", str(exception))
    
    def test_configuration_error(self):
        """测试配置错误"""
        exception = ConfigurationError("batch_size", "int", "invalid")
        
        self.assertEqual(exception.config_key, "batch_size")
        self.assertEqual(exception.expected_type, "int")
        self.assertEqual(exception.actual_value, "invalid")
        self.assertIn("配置错误", str(exception))
    
    def test_model_error(self):
        """测试模型错误"""
        exception = ModelError("GPT2", "forward", "CUDA out of memory")
        
        self.assertEqual(exception.model_name, "GPT2")
        self.assertEqual(exception.operation, "forward")
        self.assertIn("模型错误", str(exception))
    
    def test_training_error(self):
        """测试训练错误"""
        exception = TrainingError(100, 500.0, "梯度爆炸")
        
        self.assertEqual(exception.step, 100)
        self.assertEqual(exception.loss_value, 500.0)
        self.assertIn("训练错误", str(exception))
    
    def test_evaluation_error(self):
        """测试评估错误"""
        exception = EvaluationError("coherence", "计算失败")
        
        self.assertEqual(exception.metric_name, "coherence")
        self.assertIn("评估错误", str(exception))
    
    def test_hardware_error(self):
        """测试硬件错误"""
        exception = HardwareError("CUDA", "设备不可用")
        
        self.assertEqual(exception.device, "CUDA")
        self.assertIn("硬件错误", str(exception))


class TestErrorHandler(unittest.TestCase):
    """错误处理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.error_handler = ErrorHandler()
    
    def test_handle_exception(self):
        """测试异常处理"""
        # 测试基础异常处理
        exception = EITPException("测试异常")
        self.error_handler.handle_exception(exception)
        
        # 验证错误计数
        error_summary = self.error_handler.get_error_summary()
        self.assertEqual(error_summary['EITPException'], 1)
    
    def test_handle_memory_overflow(self):
        """测试内存溢出处理"""
        exception = MemoryOverflowError("GPU", 4.0, 3.0)
        self.error_handler.handle_exception(exception)
        
        # 验证错误计数
        error_summary = self.error_handler.get_error_summary()
        self.assertEqual(error_summary['MemoryOverflowError'], 1)
    
    def test_handle_convergence_error(self):
        """测试收敛错误处理"""
        exception = ConvergenceError(1500.0, 1000.0)
        self.error_handler.handle_exception(exception)
        
        # 验证错误计数
        error_summary = self.error_handler.get_error_summary()
        self.assertEqual(error_summary['ConvergenceError'], 1)
    
    def test_reset_error_count(self):
        """测试错误计数重置"""
        exception = EITPException("测试异常")
        self.error_handler.handle_exception(exception)
        
        # 验证有错误计数
        error_summary = self.error_handler.get_error_summary()
        self.assertGreater(len(error_summary), 0)
        
        # 重置错误计数
        self.error_handler.reset_error_count()
        error_summary = self.error_handler.get_error_summary()
        self.assertEqual(len(error_summary), 0)


class TestUtilityFunctions(unittest.TestCase):
    """工具函数测试类"""
    
    def test_check_gpu_memory(self):
        """测试GPU内存检查"""
        # 测试GPU不可用的情况
        if not torch.cuda.is_available():
            result = check_gpu_memory()
            self.assertFalse(result)
        else:
            # 测试正常情况
            result = check_gpu_memory(threshold=0.0)  # 设置极低阈值
            self.assertIsInstance(result, bool)
    
    def test_check_convergence(self):
        """测试收敛检查"""
        # 测试正常收敛
        result = check_convergence(500.0, 1000.0)
        self.assertFalse(result)
        
        # 测试异常收敛
        result = check_convergence(1500.0, 1000.0)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
