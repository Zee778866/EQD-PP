"""
任务自适应机制 - 根据数据集特性动态调整优化权重
文件位置: assembled_ensembles/methods/qdo/task_adaptive.py
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.utils import check_random_state
import warnings

class TaskAdaptiveController:
    """
    任务自适应控制器
    根据数据集的元特征动态调整质量、多样性和可解释性的权重分配
    """
    
    def __init__(self, 
                 default_weights: Optional[Dict[str, float]] = None,
                 random_state: Optional[int] = None):
        """
        初始化任务自适应控制器
        
        Parameters:
        -----------
        default_weights : dict, optional
            默认权重配置，包含quality_weight, diversity_weight, explainability_weight
        random_state : int, optional
            随机种子，用于可重复的权重调整
        """
        self.random_state = check_random_state(random_state)
        
        # 设置默认权重
        if default_weights is None:
            self.default_weights = {
                'quality_weight': 0.5,
                'diversity_weight': 0.3,
                'explainability_weight': 0.2
            }
        else:
            self.default_weights = default_weights.copy()
        
        # 验证权重总和为1
        self._validate_weights(self.default_weights)
        
        # 预定义调整规则
        self._setup_adjustment_rules()
    
    def _validate_weights(self, weights: Dict[str, float]):
        """验证权重配置是否有效"""
        required_keys = {'quality_weight', 'diversity_weight', 'explainability_weight'}
        if not required_keys.issubset(weights.keys()):
            raise ValueError(f"权重配置必须包含: {required_keys}")
        
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"权重总和必须为1.0，当前为: {total}")
    
    def _setup_adjustment_rules(self):
        """设置权重调整规则"""
        self.adjustment_rules = [
            # 规则1: 样本量影响
            {
                'condition': lambda mf: mf['n_samples'] < 100,
                'adjustment': {'quality_weight': +0.2, 'diversity_weight': -0.1, 'explainability_weight': -0.1},
                'description': '小样本数据：优先质量，避免过拟合'
            },
            {
                'condition': lambda mf: mf['n_samples'] > 10000,
                'adjustment': {'quality_weight': -0.1, 'diversity_weight': +0.1, 'explainability_weight': +0.0},
                'description': '大样本数据：增加多样性，提升泛化'
            },
            
            # 规则2: 特征维度影响
            {
                'condition': lambda mf: mf['n_features'] > 100,
                'adjustment': {'quality_weight': -0.1, 'diversity_weight': -0.05, 'explainability_weight': +0.15},
                'description': '高维数据：增强可解释性，理解特征重要性'
            },
            {
                'condition': lambda mf: mf['n_features'] < 10,
                'adjustment': {'quality_weight': +0.1, 'diversity_weight': +0.0, 'explainability_weight': -0.1},
                'description': '低维数据：侧重质量，可解释性需求低'
            },
            
            # 规则3: 类别平衡影响
            {
                'condition': lambda mf: mf.get('balance_ratio', 1.0) < 0.3,
                'adjustment': {'quality_weight': -0.05, 'diversity_weight': +0.15, 'explainability_weight': -0.1},
                'description': '不平衡数据：增强多样性，改善少数类表现'
            },
            {
                'condition': lambda mf: mf.get('n_classes', 2) > 10,
                'adjustment': {'quality_weight': +0.0, 'diversity_weight': +0.1, 'explainability_weight': -0.1},
                'description': '多分类问题：需要更多样性的模型'
            },
            
            # 规则4: 数据稀疏性影响
            {
                'condition': lambda mf: mf.get('sparsity_ratio', 0) > 0.8,
                'adjustment': {'quality_weight': -0.1, 'diversity_weight': +0.1, 'explainability_weight': +0.0},
                'description': '稀疏数据：多样性更重要'
            }
        ]
    
    def extract_meta_features(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        从数据集中提取元特征
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            目标标签
        
        Returns:
        --------
        dict : 包含元特征的字典
        """
        if X.size == 0 or len(y) == 0:
            return {
                'n_samples': 0,
                'n_features': 0,
                'n_classes': 0,
                'balance_ratio': 1.0,
                'sparsity_ratio': 0.0
            }
        
        n_samples, n_features = X.shape
        
        # 基础元特征
        meta_features = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': len(np.unique(y)),
        }
        
        # 计算类别平衡度
        try:
            class_counts = np.bincount(y)
            if len(class_counts) > 0:
                balance_ratio = np.min(class_counts) / np.max(class_counts)
                meta_features['balance_ratio'] = balance_ratio
            else:
                meta_features['balance_ratio'] = 1.0
        except:
            meta_features['balance_ratio'] = 1.0
        
        # 计算数据稀疏性（近似）
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sparsity_ratio = np.mean(np.isclose(X, 0)) if X.size > 0 else 0.0
                meta_features['sparsity_ratio'] = sparsity_ratio
        except:
            meta_features['sparsity_ratio'] = 0.0
        
        # 计算特征相关性（近似）
        try:
            if n_features > 1 and n_samples > 1:
                # 使用绝对相关系数的均值作为特征相关性的代理
                corr_matrix = np.corrcoef(X.T)
                np.fill_diagonal(corr_matrix, 0)  # 忽略自相关
                avg_correlation = np.mean(np.abs(corr_matrix))
                meta_features['avg_feature_correlation'] = avg_correlation
            else:
                meta_features['avg_feature_correlation'] = 0.0
        except:
            meta_features['avg_feature_correlation'] = 0.0
        
        return meta_features
    
    def adjust_weights(self, meta_features: Dict[str, Any]) -> Dict[str, float]:
        """
        根据元特征调整优化权重
        
        Parameters:
        -----------
        meta_features : dict
            数据集的元特征
        
        Returns:
        --------
        dict : 调整后的权重配置
        """
        # 从默认权重开始
        adjusted_weights = self.default_weights.copy()
        
        # 应用调整规则
        total_adjustment = {k: 0.0 for k in adjusted_weights.keys()}
        applied_rules = []
        
        for rule in self.adjustment_rules:
            if rule['condition'](meta_features):
                for key, adjustment in rule['adjustment'].items():
                    total_adjustment[key] += adjustment
                applied_rules.append(rule['description'])
        
        # 应用总调整量
        for key in adjusted_weights.keys():
            adjusted_weights[key] += total_adjustment[key]
        
        # 确保权重在有效范围内 [0, 1]
        for key in adjusted_weights.keys():
            adjusted_weights[key] = max(0.0, min(1.0, adjusted_weights[key]))
        
        # 重新归一化，确保总和为1
        total = sum(adjusted_weights.values())
        if total > 0:
            for key in adjusted_weights.keys():
                adjusted_weights[key] /= total
        else:
            # 如果出现异常，回退到默认权重
            adjusted_weights = self.default_weights.copy()
        
        # 记录调整信息（用于调试）
        self.last_adjustment_info = {
            'meta_features': meta_features,
            'applied_rules': applied_rules,
            'final_weights': adjusted_weights.copy()
        }
        
        return adjusted_weights
    
    def get_adjustment_info(self) -> Dict[str, Any]:
        """获取最后一次权重调整的详细信息"""
        return getattr(self, 'last_adjustment_info', {})


# 预定义的任务自适应策略
class PredefinedStrategies:
    """预定义的任务自适应策略"""
    
    @staticmethod
    def conservative_strategy() -> TaskAdaptiveController:
        """保守策略：小幅调整，保持稳定性"""
        return TaskAdaptiveController({
            'quality_weight': 0.6,
            'diversity_weight': 0.25,
            'explainability_weight': 0.15
        })
    
    @staticmethod
    def aggressive_strategy() -> TaskAdaptiveController:
        """激进策略：大幅调整，追求性能"""
        controller = TaskAdaptiveController({
            'quality_weight': 0.4,
            'diversity_weight': 0.3,
            'explainability_weight': 0.3
        })
        # 添加更激进的规则
        controller.adjustment_rules.extend([
            {
                'condition': lambda mf: mf['n_samples'] < 500,
                'adjustment': {'quality_weight': +0.3, 'diversity_weight': -0.2, 'explainability_weight': -0.1},
                'description': '小样本激进调整：强烈侧重质量'
            }
        ])
        return controller
    
    @staticmethod
    def explainability_focused_strategy() -> TaskAdaptiveController:
        """可解释性优先策略"""
        return TaskAdaptiveController({
            'quality_weight': 0.3,
            'diversity_weight': 0.3,
            'explainability_weight': 0.4
        })