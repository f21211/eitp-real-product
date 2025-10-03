"""
随机搜索优化器
"""

import random
from typing import Dict, Any, List, Optional, Callable

from .hyperparameter_optimizer import HyperparameterOptimizer


class RandomSearchOptimizer(HyperparameterOptimizer):
    """随机搜索优化器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger.info("随机搜索优化器已初始化")
    
    def _run_optimization(self, objective_function: Callable, parameter_space: Dict[str, Any], direction: str):
        """运行随机搜索优化"""
        best_params = None
        best_score = float('-inf') if direction == 'maximize' else float('inf')
        optimization_history = []
        
        # 随机搜索
        for trial in range(self.config['max_trials']):
            # 随机采样参数
            trial_params = {}
            for param_name, param_values in parameter_space.items():
                if isinstance(param_values, list):
                    trial_params[param_name] = random.choice(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    # 连续参数范围
                    trial_params[param_name] = random.uniform(param_values[0], param_values[1])
                else:
                    trial_params[param_name] = param_values
            
            # 评估参数
            trial_result = self._evaluate_trial(trial_params, objective_function, trial)
            optimization_history.append(trial_result)
            
            # 更新最佳参数
            if self._is_better_score(trial_result['score'], best_score):
                best_score = trial_result['score']
                best_params = trial_params.copy()
        
        return best_params, best_score, optimization_history
