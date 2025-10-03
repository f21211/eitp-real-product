"""
遗传算法优化器
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable
import random

from .hyperparameter_optimizer import HyperparameterOptimizer


class GeneticOptimizer(HyperparameterOptimizer):
    """遗传算法优化器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger.info("遗传算法优化器已初始化")
    
    def _run_optimization(self, objective_function: Callable, parameter_space: Dict[str, Any], direction: str):
        """运行遗传算法优化"""
        # 简化的遗传算法实现
        best_params = None
        best_score = float('-inf') if direction == 'maximize' else float('inf')
        optimization_history = []
        
        # 初始化种群
        population_size = min(20, self.config['max_trials'])
        population = self._initialize_population(parameter_space, population_size)
        
        for generation in range(self.config['max_trials'] // population_size):
            # 评估种群
            for i, individual in enumerate(population):
                trial_result = self._evaluate_trial(individual, objective_function, generation * population_size + i)
                optimization_history.append(trial_result)
                
                # 更新最佳参数
                if self._is_better_score(trial_result['score'], best_score):
                    best_score = trial_result['score']
                    best_params = individual.copy()
            
            # 选择、交叉、变异
            population = self._evolve_population(population, parameter_space)
        
        return best_params, best_score, optimization_history
    
    def _initialize_population(self, parameter_space: Dict[str, Any], size: int) -> List[Dict[str, Any]]:
        """初始化种群"""
        population = []
        for _ in range(size):
            individual = {}
            for param_name, param_values in parameter_space.items():
                if isinstance(param_values, list):
                    individual[param_name] = random.choice(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    individual[param_name] = random.uniform(param_values[0], param_values[1])
                else:
                    individual[param_name] = param_values
            population.append(individual)
        return population
    
    def _evolve_population(self, population: List[Dict[str, Any]], parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """进化种群"""
        # 简化的进化操作
        new_population = []
        
        # 保留最佳个体
        new_population.append(population[0])
        
        # 生成新个体
        for _ in range(len(population) - 1):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = self._crossover(parent1, parent2, parameter_space)
            child = self._mutate(child, parameter_space)
            new_population.append(child)
        
        return new_population
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], 
                   parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """交叉操作"""
        child = {}
        for param_name in parameter_space.keys():
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        return child
    
    def _mutate(self, individual: Dict[str, Any], parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """变异操作"""
        mutated = individual.copy()
        for param_name, param_values in parameter_space.items():
            if random.random() < 0.1:  # 10%变异概率
                if isinstance(param_values, list):
                    mutated[param_name] = random.choice(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    mutated[param_name] = random.uniform(param_values[0], param_values[1])
        return mutated
