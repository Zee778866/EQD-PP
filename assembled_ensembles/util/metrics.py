# Potentially useful metrics for evaluation wrapped in an easier to use object

import pandas as pd
import numpy as np

from sklearn.utils.validation import _check_y, check_array

from typing import Union, Callable, List, Optional
from abc import ABCMeta, abstractmethod


# -- Metric Utils
def make_metric(metric_func: Callable, metric_name: str, maximize: bool,
                classification: bool, always_transform_conf_to_pred: bool,
                optimum_value: int, pos_label: int = 1, requires_confidences: bool = False,
                only_positive_class: bool = False):
    """ Make a metric that has additional information

    Parameters
    ----------
    metric_func: Callable
        The metric function to call.
        We expect it to be metric_func(y_true, y_pred) with y_pred potentially being
        probabilities instead of classes.
    metric_name: str
        Name of the metric
    maximize: bool
        Whether to maximize the metric or not
    classification: bool
        If it is a classification metric or not
    always_transform_conf_to_pred: bool
        Set to Ture if the metric can not handle confidences and only accepts predictions (only for classification)
    optimum_value: int
        The maximal value the metric can reach (used to compute the loss).
    pos_label: int, default=1
        Index of the label used as positive label (relevant only for binary classification metrics)
    requires_confidences: bool, default=False
        If the metric requires confidences.
    only_positive_class: bool, default=False
        Only relevant if requires_confidences is True. If only_positive_class is true, only the positive class
        values are passed. This is only needed for binary classification. Ignored if always_transform_conf_to_pred is
        True.
    """

    return AbstractMetric(metric_func, metric_name, maximize, classification,
                          always_transform_conf_to_pred, optimum_value, pos_label,
                          requires_confidences, only_positive_class)


class AbstractMetric:
    """Abstract Metric used in some codes

    We transform confidences to prediction if needed for a metric and the model does it not by itself.
    Thereby, we assume y_true to be integers because we transform y_pred into integers as well.

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, metric, name, maximize, classification, transform_conf_to_pred, optimum_value, pos_label,
                 requires_confidences, only_positive_class):
        self.metric = metric
        self.maximize = maximize
        self.name = name
        self.classification = classification
        self.transform_conf_to_pred = transform_conf_to_pred
        self.optimum_value = optimum_value
        self.pos_label = pos_label
        self.threshold = 0.5
        self.requires_confidences = requires_confidences
        self.only_positive_class = only_positive_class

    def __call__(self, y_true: Union[pd.DataFrame, np.ndarray], y_pred: Union[pd.DataFrame, np.ndarray],
                 to_loss: bool = False, checks=True):
        """

        Parameters
        ----------
        y_true: array-like
            ground truth, assumed to be integers!
        y_pred: array-like
            Either confidences/probabilities matrix (n_samples, n_classes) or prediction vector (n_samples, )
            If not classification, only prediction vector is allowed for now.
            If confidences, we expect the order of n_classes to be identical to the order of np.unique(y_true).
        to_loss: bool
            Whether to return the loss or not
        """

        # -- Input validation
        if checks:
            y_true = _check_y(y_true)

        if not self.classification:
            if checks:
                y_pred = _check_y(y_pred, y_numeric=True)
        else:
            if y_pred.ndim == 1:
                if checks:
                    y_pred = _check_y(y_pred)

                    if self.requires_confidences:
                        raise ValueError("Confidences are needed for this metric but predictions are passed.")
            elif y_pred.ndim == 2:
                if checks:
                    y_pred = check_array(y_pred)

                # - Special case if metric can not handle confidences
                if self.transform_conf_to_pred:
                    y_pred = np.argmax(y_pred, axis=1)
                elif self.only_positive_class:
                    y_pred = y_pred[:, self.pos_label]

            else:
                raise ValueError("y_pred has to many dimensions! Found ndim: {}".format(y_pred.ndim))

        # --- Call metric
        metric_value = self.metric(y_true, y_pred)

        # --- Return
        if to_loss:
            return self.to_loss(metric_value)

        return metric_value

    def to_loss(self, metric_value):
        # General Purpose Loss: the absolute difference to the optimum
        #   -> smaller is always better
        return abs(self.optimum_value - metric_value)

    def inverse_loss(self, loss_value):

        if (self.optimum_value == 0) and (not self.maximize):
            return loss_value

        # FIXME: this ignores negative metric_values or optima
        return self.optimum_value - loss_value


# ==================== 具体指标实现 ====================

class AccuracyMetric(AbstractMetric):
    """准确率指标实现"""
    
    def __init__(self):
        from sklearn.metrics import accuracy_score
        super().__init__(
            metric=accuracy_score,
            name="accuracy",
            maximize=True,
            classification=True,
            transform_conf_to_pred=True,
            optimum_value=1.0,
            pos_label=1,
            requires_confidences=False,
            only_positive_class=False
        )
    
    def __repr__(self):
        return "AccuracyMetric()"


class ROCAUCMetric(AbstractMetric):
    """ROC AUC指标实现"""
    
    def __init__(self):
        from sklearn.metrics import roc_auc_score
        super().__init__(
            metric=roc_auc_score,
            name="roc_auc",
            maximize=True,
            classification=True,
            transform_conf_to_pred=False,
            optimum_value=1.0,
            pos_label=1,
            requires_confidences=True,
            only_positive_class=True
        )
    
    def __call__(self, y_true, y_pred, to_loss=False, checks=True):
        """重写调用方法以处理多分类情况"""
        if checks:
            y_true = _check_y(y_true)
            y_pred = _check_y(y_pred)
        
        # 处理多分类情况
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 2:
            score = self.metric(y_true, y_pred, multi_class='ovr')
        else:
            score = self.metric(y_true, y_pred[:, 1] if y_pred.shape[1] == 2 else y_pred)
        
        return 1 - score if to_loss else score
    
    def __repr__(self):
        return "ROCAUCMetric()"


class MSEMetric(AbstractMetric):
    """均方误差指标实现"""
    
    def __init__(self):
        from sklearn.metrics import mean_squared_error
        super().__init__(
            metric=mean_squared_error,
            name="mse",
            maximize=False,
            classification=False,
            transform_conf_to_pred=False,
            optimum_value=0.0,
            pos_label=1,
            requires_confidences=False,
            only_positive_class=False
        )
    
    def __repr__(self):
        return "MSEMetric()"


class MAEMetric(AbstractMetric):
    """平均绝对误差指标实现"""
    
    def __init__(self):
        from sklearn.metrics import mean_absolute_error
        super().__init__(
            metric=mean_absolute_error,
            name="mae",
            maximize=False,
            classification=False,
            transform_conf_to_pred=False,
            optimum_value=0.0,
            pos_label=1,
            requires_confidences=False,
            only_positive_class=False
        )
    
    def __repr__(self):
        return "MAEMetric()"


# ==================== 新增功能：缺失函数添加 ====================

def get_metric(metric_name: str, **kwargs) -> AbstractMetric:
    """
    根据名称获取预定义指标实例
    
    Parameters
    ----------
    metric_name : str
        指标名称，支持: 'accuracy', 'roc_auc', 'mse', 'mae'
    **kwargs
        传递给具体指标构造函数的额外参数
        
    Returns
    -------
    AbstractMetric
        对应的指标实例
        
    Raises
    ------
    ValueError
        当指标名称不被支持时
    """
    metric_map = {
        'accuracy': AccuracyMetric,
        'roc_auc': ROCAUCMetric,
        'mse': MSEMetric,
        'mae': MAEMetric
    }
    
    if metric_name not in metric_map:
        raise ValueError(
            f"未知的指标名称: '{metric_name}'. "
            f"支持的指标: {list(metric_map.keys())}"
        )
    
    return metric_map[metric_name](**kwargs)


def validate_metric_compatibility(metric: AbstractMetric, y_true: np.ndarray, 
                                 y_pred: np.ndarray, **kwargs) -> bool:
    """
    验证指标与数据的兼容性
    
    Parameters
    ----------
    metric : AbstractMetric
        要验证的指标实例
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray  
        预测值
    **kwargs
        传递给指标调用的额外参数
        
    Returns
    -------
    bool
        True表示兼容，False表示不兼容
    """
    try:
        # 尝试计算指标值来验证兼容性
        test_score = metric(y_true, y_pred, to_loss=False, checks=True, **kwargs)
        
        # 检查得分是否在合理范围内
        if hasattr(metric, 'optimum_value'):
            if metric.maximize:
                valid = test_score <= metric.optimum_value
            else:
                valid = test_score >= metric.optimum_value
        else:
            # 如果没有optimum_value，只检查是否为数值
            valid = np.isfinite(test_score)
            
        return valid
        
    except Exception as e:
        # 记录调试信息（可选）
        import warnings
        warnings.warn(f"指标兼容性验证失败: {e}")
        return False


def list_available_metrics() -> List[str]:
    """
    获取所有可用的指标名称列表
    
    Returns
    -------
    List[str]
        可用的指标名称列表
    """
    return ['accuracy', 'roc_auc', 'mse', 'mae']


def create_custom_metric(metric_func: Callable, metric_name: str, **kwargs) -> AbstractMetric:
    """
    创建自定义指标的便捷函数
    
    Parameters
    ----------
    metric_func : Callable
        指标计算函数，格式应为 func(y_true, y_pred)
    metric_name : str
        指标名称
    **kwargs
        传递给make_metric的额外参数
        
    Returns
    -------
    AbstractMetric
        自定义指标实例
        
    Examples
    --------
    >>> def my_metric(y_true, y_pred):
    ...     return np.mean(y_true == y_pred)
    >>> custom_metric = create_custom_metric(my_metric, "my_accuracy", maximize=True)
    """
    # 默认参数（可根据需要调整）
    default_kwargs = {
        'maximize': True,
        'classification': True,
        'always_transform_conf_to_pred': False,
        'optimum_value': 1.0,
        'pos_label': 1,
        'requires_confidences': False,
        'only_positive_class': False
    }
    
    # 更新默认参数
    default_kwargs.update(kwargs)
    
    return make_metric(metric_func, metric_name, **default_kwargs)


def metric_from_config(config: dict) -> AbstractMetric:
    """
    从配置字典创建指标实例
    
    Parameters
    ----------
    config : dict
        包含指标配置的字典，必须包含 'name' 键
        
    Returns
    -------
    AbstractMetric
        配置对应的指标实例
        
    Raises
    ------
    ValueError
        当配置缺少必要信息时
    KeyError
        当指标名称不被支持时
    """
    if 'name' not in config:
        raise ValueError("配置字典必须包含 'name' 键")
    
    # 支持预定义指标和自定义指标
    if config['name'] in list_available_metrics():
        return get_metric(config['name'], **{k: v for k, v in config.items() if k != 'name'})
    else:
        # 假设这是自定义指标，需要提供 metric_func
        if 'metric_func' not in config:
            raise ValueError("自定义指标必须提供 'metric_func'")
        
        return create_custom_metric(
            config['metric_func'],
            config['name'],
            **{k: v for k, v in config.items() if k not in ['name', 'metric_func']}
        )


# ==================== 工具函数 ====================

def _safe_metric_call(metric: AbstractMetric, y_true: np.ndarray, 
                     y_pred: np.ndarray, default_value: float = 0.5, **kwargs) -> float:
    """
    安全的指标调用，避免异常导致程序中断
    
    Parameters
    ----------
    metric : AbstractMetric
        要调用的指标
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray
        预测值
    default_value : float, optional
        发生错误时返回的默认值，默认为0.5
    **kwargs
        传递给指标调用的额外参数
        
    Returns
    -------
    float
        指标计算结果或默认值
    """
    try:
        return metric(y_true, y_pred, **kwargs)
    except Exception as e:
        import warnings
        warnings.warn(f"指标计算失败，使用默认值 {default_value}: {e}")
        return default_value


def batch_metrics_evaluation(metrics: List[AbstractMetric], y_true: np.ndarray,
                           y_pred: np.ndarray, **kwargs) -> dict:
    """
    批量计算多个指标
    
    Parameters
    ----------
    metrics : List[AbstractMetric]
        要计算的指标列表
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray
        预测值
    **kwargs
        传递给每个指标调用的额外参数
        
    Returns
    -------
    dict
        包含每个指标计算结果的字典
    """
    results = {}
    
    for metric in metrics:
        try:
            result = metric(y_true, y_pred, **kwargs)
            results[metric.name] = result
        except Exception as e:
            import warnings
            warnings.warn(f"指标 {metric.name} 计算失败: {e}")
            results[metric.name] = None
    
    return results


# ==================== 导出定义 ====================

__all__ = [
    # 原有导出
    'AbstractMetric', 'AccuracyMetric', 'ROCAUCMetric', 'MSEMetric', 'MAEMetric', 'make_metric',
    
    # 新增导出
    'get_metric', 'validate_metric_compatibility', 'list_available_metrics',
    'create_custom_metric', 'metric_from_config', '_safe_metric_call',
    'batch_metrics_evaluation'
]


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """模块测试代码"""
    print("=== metrics.py 模块测试 ===")
    
    # 测试数据
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred_proba = np.random.rand(6, 2)
    y_pred_proba = y_pred_proba / np.sum(y_pred_proba, axis=1, keepdims=True)
    y_pred_label = np.argmax(y_pred_proba, axis=1)
    
    # 测试 get_metric 函数
    print("1. 测试 get_metric 函数:")
    try:
        accuracy_metric = get_metric('accuracy')
        roc_auc_metric = get_metric('roc_auc')
        mse_metric = get_metric('mse')
        mae_metric = get_metric('mae')
        print("   ✅ 所有预定义指标获取成功")
    except Exception as e:
        print(f"   ❌ 获取指标失败: {e}")
    
    # 测试指标计算
    print("2. 测试指标计算:")
    try:
        acc_score = accuracy_metric(y_true, y_pred_label)
        auc_score = roc_auc_metric(y_true, y_pred_proba)
        print(f"   ✅ 准确率: {acc_score:.4f}")
        print(f"   ✅ ROC AUC: {auc_score:.4f}")
    except Exception as e:
        print(f"   ❌ 指标计算失败: {e}")
    
    # 测试兼容性验证
    print("3. 测试兼容性验证:")
    try:
        compatible = validate_metric_compatibility(accuracy_metric, y_true, y_pred_label)
        print(f"   ✅ 兼容性验证: {compatible}")
    except Exception as e:
        print(f"   ❌ 兼容性验证失败: {e}")
    
    # 测试列表函数
    print("4. 测试可用指标列表:")
    try:
        available_metrics = list_available_metrics()
        print(f"   ✅ 可用指标: {available_metrics}")
    except Exception as e:
        print(f"   ❌ 获取指标列表失败: {e}")
    
    print("=== 测试完成 ===")