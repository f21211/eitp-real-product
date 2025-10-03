"""
网格搜索优化器
"""

from typing import Dict, Any, List, Optional, Callable
import itertools

from .hyperparameter_optimizer import HyperparameterOptimizer


class GridSearchOptimizer(HyperparameterOptimizer):
    """网格搜索优化器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger.info("网格搜索优化器已初始化")
    
    def _run_optimization(self, objective_function: Callable, parameter_space: Dict[str, Any], direction: str):
        """运行网格搜索优化"""
        # 生成参数网格
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        
        # 生成所有参数组合
        param_combinations = list(itertools.product(*param_values))
        
        # 限制搜索空间大小
        max_combinations = min(len(param_combinations), self.config['max_trials'])
        param_combinations = param_combinations[:max_combinations]
        
        best_params = None
        best_score = float('-inf') if direction == 'maximize' else float('inf')
        optimization_history = []
        
        # 遍历所有参数组合
        for trial, param_combination in enumerate(param_combinations):
            trial_params = dict(zip(param_names, param_combination))
            
            # 评估参数
            trial_result = self._evaluate_trial(trial_params, objective_function, trial)
            optimization_history.append(trial_result)
            
            # 更新最佳参数
            if self._is_better_score(trial_result['score'], best_score):
                best_score = trial_result['score']
                best_params = trial_params.copy()
        
        return best_params, best_score, optimization_history
