"""
实验分析器
分析A/B测试结果并提供统计洞察
"""

import math
import statistics
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class StatisticalTest(Enum):
    """统计检验类型"""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"
    FISHER_EXACT = "fisher_exact"


@dataclass
class StatisticalResult:
    """统计检验结果"""
    test_type: StatisticalTest
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    is_significant: bool
    power: float
    sample_size: int


@dataclass
class ExperimentInsight:
    """实验洞察"""
    insight_type: str
    description: str
    confidence: float
    recommendation: str
    supporting_data: Dict[str, Any]


class ExperimentAnalyzer:
    """实验分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger("experiment_analyzer")
    
    def analyze_experiment(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析实验数据"""
        try:
            # 提取基本统计信息
            basic_stats = self._calculate_basic_statistics(experiment_data)
            
            # 执行统计检验
            statistical_tests = self._perform_statistical_tests(experiment_data)
            
            # 生成洞察
            insights = self._generate_insights(experiment_data, basic_stats, statistical_tests)
            
            # 计算置信区间
            confidence_intervals = self._calculate_confidence_intervals(experiment_data)
            
            # 评估实验质量
            quality_metrics = self._evaluate_experiment_quality(experiment_data)
            
            return {
                "experiment_id": experiment_data.get("experiment_id", "unknown"),
                "analysis_timestamp": datetime.now().isoformat(),
                "basic_statistics": basic_stats,
                "statistical_tests": statistical_tests,
                "insights": insights,
                "confidence_intervals": confidence_intervals,
                "quality_metrics": quality_metrics,
                "recommendations": self._generate_recommendations(insights, quality_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing experiment: {e}")
            return {"error": str(e)}
    
    def _calculate_basic_statistics(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算基本统计信息"""
        variants = experiment_data.get("variants", {})
        stats = {}
        
        for variant_name, variant_data in variants.items():
            values = variant_data.get("values", [])
            if not values:
                continue
            
            n = len(values)
            mean_val = statistics.mean(values)
            median_val = statistics.median(values)
            std_val = statistics.stdev(values) if n > 1 else 0
            min_val = min(values)
            max_val = max(values)
            
            # 计算分位数
            sorted_values = sorted(values)
            q1 = sorted_values[int(n * 0.25)] if n > 0 else 0
            q3 = sorted_values[int(n * 0.75)] if n > 0 else 0
            
            stats[variant_name] = {
                "sample_size": n,
                "mean": round(mean_val, 4),
                "median": round(median_val, 4),
                "std_dev": round(std_val, 4),
                "min": min_val,
                "max": max_val,
                "q1": q1,
                "q3": q3,
                "iqr": round(q3 - q1, 4),
                "coefficient_of_variation": round(std_val / mean_val, 4) if mean_val != 0 else 0
            }
        
        return stats
    
    def _perform_statistical_tests(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行统计检验"""
        variants = experiment_data.get("variants", {})
        variant_names = list(variants.keys())
        
        if len(variant_names) < 2:
            return {"error": "Need at least 2 variants for comparison"}
        
        tests = {}
        
        # 两两比较
        for i in range(len(variant_names)):
            for j in range(i + 1, len(variant_names)):
                variant1 = variant_names[i]
                variant2 = variant_names[j]
                
                values1 = variants[variant1].get("values", [])
                values2 = variants[variant2].get("values", [])
                
                if not values1 or not values2:
                    continue
                
                comparison_key = f"{variant1}_vs_{variant2}"
                
                # t检验
                t_test_result = self._t_test(values1, values2)
                tests[comparison_key] = {
                    "t_test": t_test_result,
                    "effect_size": self._calculate_effect_size(values1, values2),
                    "power": self._calculate_power(values1, values2, t_test_result.p_value)
                }
        
        return tests
    
    def _t_test(self, group1: List[float], group2: List[float]) -> StatisticalResult:
        """执行t检验"""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return StatisticalResult(
                test_type=StatisticalTest.T_TEST,
                p_value=1.0,
                confidence_interval=(0, 0),
                effect_size=0,
                is_significant=False,
                power=0,
                sample_size=n1 + n2
            )
        
        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
        var1, var2 = statistics.variance(group1), statistics.variance(group2)
        
        # 合并方差
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = math.sqrt(pooled_var)
        
        # t统计量
        se_diff = pooled_std * math.sqrt(1/n1 + 1/n2)
        t_stat = (mean1 - mean2) / se_diff
        
        # 自由度
        df = n1 + n2 - 2
        
        # p值（简化计算）
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        
        # 置信区间
        t_critical = self._t_critical_value(df, 0.95)
        margin_error = t_critical * se_diff
        ci_lower = (mean1 - mean2) - margin_error
        ci_upper = (mean1 - mean2) + margin_error
        
        return StatisticalResult(
            test_type=StatisticalTest.T_TEST,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0,
            is_significant=p_value < 0.05,
            power=0,  # 将在后续计算
            sample_size=n1 + n2
        )
    
    def _t_cdf(self, t: float, df: int) -> float:
        """t分布累积分布函数的近似"""
        # 简化的近似计算
        if df > 30:
            # 大样本时近似为标准正态分布
            return 0.5 * (1 + math.erf(t / math.sqrt(2)))
        else:
            # 小样本的简化近似
            return 0.5 + 0.5 * math.tanh(t / 2)
    
    def _t_critical_value(self, df: int, confidence: float) -> float:
        """t分布临界值"""
        # 简化的临界值表
        if df >= 30:
            return 1.96  # 95%置信度
        elif df >= 20:
            return 2.086
        elif df >= 10:
            return 2.228
        else:
            return 2.262
    
    def _calculate_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """计算效应大小（Cohen's d）"""
        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
        var1, var2 = statistics.variance(group1), statistics.variance(group2)
        n1, n2 = len(group1), len(group2)
        
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = math.sqrt(pooled_var)
        
        return abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    def _calculate_power(self, group1: List[float], group2: List[float], p_value: float) -> float:
        """计算统计功效"""
        # 简化的功效计算
        effect_size = self._calculate_effect_size(group1, group2)
        n = min(len(group1), len(group2))
        
        # 基于效应大小和样本量的简化功效估计
        if effect_size < 0.2:
            return 0.2
        elif effect_size < 0.5:
            return 0.5
        elif effect_size < 0.8:
            return 0.7
        else:
            return 0.9
    
    def _calculate_confidence_intervals(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算置信区间"""
        variants = experiment_data.get("variants", {})
        intervals = {}
        
        for variant_name, variant_data in variants.items():
            values = variant_data.get("values", [])
            if not values:
                continue
            
            n = len(values)
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if n > 1 else 0
            
            # 95%置信区间
            t_critical = self._t_critical_value(n - 1, 0.95)
            margin_error = t_critical * (std_val / math.sqrt(n))
            
            intervals[variant_name] = {
                "mean": round(mean_val, 4),
                "confidence_interval": (
                    round(mean_val - margin_error, 4),
                    round(mean_val + margin_error, 4)
                ),
                "margin_error": round(margin_error, 4)
            }
        
        return intervals
    
    def _evaluate_experiment_quality(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估实验质量"""
        variants = experiment_data.get("variants", {})
        
        # 样本量评估
        sample_sizes = [len(variant_data.get("values", [])) for variant_data in variants.values()]
        min_sample_size = min(sample_sizes) if sample_sizes else 0
        max_sample_size = max(sample_sizes) if sample_sizes else 0
        
        # 平衡性检查
        balance_ratio = min_sample_size / max_sample_size if max_sample_size > 0 else 0
        
        # 数据质量检查
        quality_issues = []
        if min_sample_size < 30:
            quality_issues.append("Small sample size")
        if balance_ratio < 0.8:
            quality_issues.append("Unbalanced groups")
        
        # 异常值检查
        outlier_ratios = {}
        for variant_name, variant_data in variants.items():
            values = variant_data.get("values", [])
            if len(values) > 4:
                q1 = statistics.quantiles(values, n=4)[0]
                q3 = statistics.quantiles(values, n=4)[2]
                iqr = q3 - q1
                outliers = [v for v in values if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr]
                outlier_ratios[variant_name] = len(outliers) / len(values)
        
        return {
            "min_sample_size": min_sample_size,
            "max_sample_size": max_sample_size,
            "balance_ratio": round(balance_ratio, 3),
            "quality_issues": quality_issues,
            "outlier_ratios": outlier_ratios,
            "overall_quality": "Good" if not quality_issues else "Needs attention"
        }
    
    def _generate_insights(self, experiment_data: Dict[str, Any], 
                          basic_stats: Dict[str, Any], 
                          statistical_tests: Dict[str, Any]) -> List[ExperimentInsight]:
        """生成实验洞察"""
        insights = []
        
        # 显著性洞察
        for test_name, test_data in statistical_tests.items():
            if isinstance(test_data, dict) and "t_test" in test_data:
                t_test = test_data["t_test"]
                if t_test.is_significant:
                    insights.append(ExperimentInsight(
                        insight_type="statistical_significance",
                        description=f"Significant difference found in {test_name} (p={t_test.p_value:.4f})",
                        confidence=1 - t_test.p_value,
                        recommendation="Consider implementing the winning variant",
                        supporting_data={"test_name": test_name, "p_value": t_test.p_value}
                    ))
                else:
                    insights.append(ExperimentInsight(
                        insight_type="no_significance",
                        description=f"No significant difference in {test_name} (p={t_test.p_value:.4f})",
                        confidence=t_test.p_value,
                        recommendation="Continue experiment or increase sample size",
                        supporting_data={"test_name": test_name, "p_value": t_test.p_value}
                    ))
        
        # 效应大小洞察
        for test_name, test_data in statistical_tests.items():
            if isinstance(test_data, dict) and "effect_size" in test_data:
                effect_size = test_data["effect_size"]
                if effect_size > 0.8:
                    insights.append(ExperimentInsight(
                        insight_type="large_effect",
                        description=f"Large effect size detected in {test_name} (d={effect_size:.3f})",
                        confidence=0.9,
                        recommendation="Strong evidence for practical significance",
                        supporting_data={"test_name": test_name, "effect_size": effect_size}
                    ))
                elif effect_size > 0.5:
                    insights.append(ExperimentInsight(
                        insight_type="medium_effect",
                        description=f"Medium effect size detected in {test_name} (d={effect_size:.3f})",
                        confidence=0.7,
                        recommendation="Moderate practical significance",
                        supporting_data={"test_name": test_name, "effect_size": effect_size}
                    ))
        
        return insights
    
    def _generate_recommendations(self, insights: List[ExperimentInsight], 
                                quality_metrics: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于洞察的建议
        significant_tests = [i for i in insights if i.insight_type == "statistical_significance"]
        if significant_tests:
            recommendations.append("实验显示统计显著性，建议实施获胜变体")
        else:
            recommendations.append("未发现统计显著性，建议继续实验或增加样本量")
        
        # 基于质量的建议
        if quality_metrics.get("overall_quality") == "Needs attention":
            recommendations.append("实验质量需要关注，建议检查数据收集过程")
        
        if quality_metrics.get("min_sample_size", 0) < 30:
            recommendations.append("样本量较小，建议增加样本量以提高统计功效")
        
        return recommendations
    
    def generate_report(self, analysis_result: Dict[str, Any]) -> str:
        """生成分析报告"""
        report = []
        report.append("# A/B测试分析报告")
        report.append(f"**实验ID**: {analysis_result.get('experiment_id', 'Unknown')}")
        report.append(f"**分析时间**: {analysis_result.get('analysis_timestamp', 'Unknown')}")
        report.append("")
        
        # 基本统计
        report.append("## 基本统计信息")
        basic_stats = analysis_result.get("basic_statistics", {})
        for variant, stats in basic_stats.items():
            report.append(f"### {variant}")
            report.append(f"- 样本量: {stats['sample_size']}")
            report.append(f"- 均值: {stats['mean']}")
            report.append(f"- 标准差: {stats['std_dev']}")
            report.append(f"- 中位数: {stats['median']}")
            report.append("")
        
        # 统计检验
        report.append("## 统计检验结果")
        statistical_tests = analysis_result.get("statistical_tests", {})
        for test_name, test_data in statistical_tests.items():
            if isinstance(test_data, dict) and "t_test" in test_data:
                t_test = test_data["t_test"]
                report.append(f"### {test_name}")
                report.append(f"- p值: {t_test.p_value:.4f}")
                report.append(f"- 显著性: {'是' if t_test.is_significant else '否'}")
                report.append(f"- 效应大小: {t_test.effect_size:.3f}")
                report.append("")
        
        # 洞察和建议
        report.append("## 洞察和建议")
        insights = analysis_result.get("insights", [])
        for insight in insights:
            report.append(f"### {insight.insight_type}")
            report.append(f"- 描述: {insight.description}")
            report.append(f"- 建议: {insight.recommendation}")
            report.append("")
        
        recommendations = analysis_result.get("recommendations", [])
        if recommendations:
            report.append("## 总体建议")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
        
        return "\n".join(report)


# 全局分析器实例
_global_analyzer = None


def get_experiment_analyzer() -> ExperimentAnalyzer:
    """获取全局实验分析器实例"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = ExperimentAnalyzer()
    return _global_analyzer