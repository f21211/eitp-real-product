"""
超参数优化器
提供多种超参数搜索策略
"""

import numpy as np
import optuna
from typing import Dict, Any, List, Optional, Callable, Union
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils import get_global_logger
from ..experiments import ExperimentManager, MetricsTracker


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_time: float
    total_trials: int
    successful_trials: int
    failed_trials: int
    optimization_history: List[Dict[str, Any]]
    best_trial_id: int


class HyperparameterOptimizer:
    """超参数优化器基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_global_logger()
        
        # 优化状态
        self.optimization_state = {
            'is_running': False,
            'current_trial': 0,
            'total_trials': 0,
            'best_score': float('-inf'),
            'start_time': None
        }
        
        # 优化历史
        self.optimization_history = []
        
        # 实验管理器
        self.experiment_manager = ExperimentManager()
        
        # 优化回调
        self.optimization_callbacks = []
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_trials': 100,
            'timeout': 3600,  # 1小时
            'n_jobs': 1,
            'random_state': 42,
            'early_stopping_patience': 10,
            'min_improvement': 0.001,
            'save_best_model': True,
            'output_dir': './optimization_results'
        }
    
    def optimize(self, 
                 objective_function: Callable[[Dict[str, Any]], float],
                 parameter_space: Dict[str, Any],
                 direction: str = 'maximize') -> OptimizationResult:
        """优化超参数"""
        try:
            self.logger.info("开始超参数优化...")
            
            # 初始化优化状态
            self.optimization_state.update({
                'is_running': True,
                'current_trial': 0,
                'total_trials': self.config['max_trials'],
                'best_score': float('-inf') if direction == 'maximize' else float('inf'),
                'start_time': time.time()
            })
            
            # 创建输出目录
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 执行优化
            best_params, best_score, optimization_history = self._run_optimization(
                objective_function, parameter_space, direction
            )
            
            # 计算优化时间
            optimization_time = time.time() - self.optimization_state['start_time']
            
            # 创建优化结果
            result = OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                optimization_time=optimization_time,
                total_trials=len(optimization_history),
                successful_trials=len([h for h in optimization_history if h['status'] == 'success']),
                failed_trials=len([h for h in optimization_history if h['status'] == 'failed']),
                optimization_history=optimization_history,
                best_trial_id=max(optimization_history, key=lambda x: x['score'])['trial_id'] if optimization_history else 0
            )
            
            # 保存优化结果
            self._save_optimization_result(result, output_dir)
            
            # 更新状态
            self.optimization_state['is_running'] = False
            
            self.logger.info(f"超参数优化完成 - 最佳得分: {best_score:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"超参数优化失败: {e}")
            self.optimization_state['is_running'] = False
            raise
    
    def _run_optimization(self, 
                         objective_function: Callable[[Dict[str, Any]], float],
                         parameter_space: Dict[str, Any],
                         direction: str) -> tuple:
        """运行优化（子类实现）"""
        raise NotImplementedError("子类必须实现_run_optimization方法")
    
    def _evaluate_trial(self, 
                       trial_params: Dict[str, Any],
                       objective_function: Callable[[Dict[str, Any]], float],
                       trial_id: int) -> Dict[str, Any]:
        """评估单个试验"""
        start_time = time.time()
        
        try:
            # 执行目标函数
            score = objective_function(trial_params)
            
            # 记录试验结果
            trial_result = {
                'trial_id': trial_id,
                'params': trial_params.copy(),
                'score': score,
                'status': 'success',
                'evaluation_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
            # 更新最佳得分
            if self._is_better_score(score, self.optimization_state['best_score']):
                self.optimization_state['best_score'] = score
                trial_result['is_best'] = True
            else:
                trial_result['is_best'] = False
            
            # 执行优化回调
            for callback in self.optimization_callbacks:
                try:
                    callback(trial_result)
                except Exception as e:
                    self.logger.warning(f"优化回调执行失败: {e}")
            
            return trial_result
            
        except Exception as e:
            self.logger.error(f"试验评估失败: {e}")
            
            trial_result = {
                'trial_id': trial_id,
                'params': trial_params.copy(),
                'score': float('-inf') if direction == 'maximize' else float('inf'),
                'status': 'failed',
                'error': str(e),
                'evaluation_time': time.time() - start_time,
                'timestamp': time.time(),
                'is_best': False
            }
            
            return trial_result
    
    def _is_better_score(self, new_score: float, current_best: float) -> bool:
        """判断新得分是否更好"""
        if self.optimization_state.get('direction', 'maximize') == 'maximize':
            return new_score > current_best
        else:
            return new_score < current_best
    
    def _save_optimization_result(self, result: OptimizationResult, output_dir: Path):
        """保存优化结果"""
        try:
            # 保存优化结果
            with open(output_dir / "optimization_result.json", 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            
            # 保存优化历史
            with open(output_dir / "optimization_history.json", 'w') as f:
                json.dump(result.optimization_history, f, indent=2, default=str)
            
            # 保存最佳参数
            with open(output_dir / "best_params.json", 'w') as f:
                json.dump(result.best_params, f, indent=2)
            
            self.logger.info(f"优化结果已保存: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"保存优化结果失败: {e}")
    
    def add_optimization_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """添加优化回调"""
        self.optimization_callbacks.append(callback)
        self.logger.debug("添加优化回调")
    
    def remove_optimization_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """移除优化回调"""
        if callback in self.optimization_callbacks:
            self.optimization_callbacks.remove(callback)
            self.logger.debug("移除优化回调")
    
    def get_optimization_state(self) -> Dict[str, Any]:
        """获取优化状态"""
        return self.optimization_state.copy()
    
    def stop_optimization(self):
        """停止优化"""
        self.optimization_state['is_running'] = False
        self.logger.info("优化已停止")
    
    def resume_optimization(self, checkpoint_path: str):
        """恢复优化"""
        try:
            checkpoint_path = Path(checkpoint_path)
            
            # 加载检查点
            with open(checkpoint_path / "optimization_result.json", 'r') as f:
                result_data = json.load(f)
            
            # 恢复状态
            self.optimization_state.update({
                'is_running': True,
                'best_score': result_data['best_score'],
                'current_trial': result_data['total_trials']
            })
            
            # 恢复历史
            with open(checkpoint_path / "optimization_history.json", 'r') as f:
                self.optimization_history = json.load(f)
            
            self.logger.info(f"优化已恢复: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"恢复优化失败: {e}")
            raise
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self.optimization_history:
            return {}
        
        successful_trials = [h for h in self.optimization_history if h['status'] == 'success']
        failed_trials = [h for h in self.optimization_history if h['status'] == 'failed']
        
        if successful_trials:
            scores = [h['score'] for h in successful_trials]
            return {
                'total_trials': len(self.optimization_history),
                'successful_trials': len(successful_trials),
                'failed_trials': len(failed_trials),
                'best_score': max(scores) if self.optimization_state.get('direction', 'maximize') == 'maximize' else min(scores),
                'worst_score': min(scores) if self.optimization_state.get('direction', 'maximize') == 'maximize' else max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'optimization_time': time.time() - self.optimization_state.get('start_time', time.time())
            }
        else:
            return {
                'total_trials': len(self.optimization_history),
                'successful_trials': 0,
                'failed_trials': len(failed_trials),
                'optimization_time': time.time() - self.optimization_state.get('start_time', time.time())
            }
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """绘制优化历史"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.optimization_history:
                self.logger.warning("没有优化历史数据")
                return
            
            successful_trials = [h for h in self.optimization_history if h['status'] == 'success']
            if not successful_trials:
                self.logger.warning("没有成功的试验")
                return
            
            # 提取数据
            trial_ids = [h['trial_id'] for h in successful_trials]
            scores = [h['score'] for h in successful_trials]
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 得分趋势
            ax1.plot(trial_ids, scores, 'b-', alpha=0.7, label='Score')
            ax1.axhline(y=self.optimization_state['best_score'], color='r', linestyle='--', label='Best Score')
            ax1.set_xlabel('Trial ID')
            ax1.set_ylabel('Score')
            ax1.set_title('Optimization History')
            ax1.legend()
            ax1.grid(True)
            
            # 得分分布
            ax2.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            ax2.axvline(x=self.optimization_state['best_score'], color='r', linestyle='--', label='Best Score')
            ax2.set_xlabel('Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Score Distribution')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"优化历史图表已保存: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib未安装，无法绘制图表")
        except Exception as e:
            self.logger.error(f"绘制优化历史失败: {e}")
    
    def export_optimization_data(self, export_path: str, format: str = 'json'):
        """导出优化数据"""
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                with open(export_path, 'w') as f:
                    json.dump({
                        'optimization_state': self.optimization_state,
                        'optimization_history': self.optimization_history,
                        'summary': self.get_optimization_summary()
                    }, f, indent=2, default=str)
            
            elif format == 'csv':
                import pandas as pd
                
                # 准备数据
                data = []
                for trial in self.optimization_history:
                    row = {
                        'trial_id': trial['trial_id'],
                        'score': trial['score'],
                        'status': trial['status'],
                        'evaluation_time': trial['evaluation_time'],
                        'timestamp': trial['timestamp']
                    }
                    # 添加参数
                    for key, value in trial['params'].items():
                        row[f'param_{key}'] = value
                    data.append(row)
                
                df = pd.DataFrame(data)
                df.to_csv(export_path, index=False)
            
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"优化数据已导出: {export_path}")
            
        except Exception as e:
            self.logger.error(f"导出优化数据失败: {e}")
            raise
