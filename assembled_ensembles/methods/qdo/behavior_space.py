#!/usr/bin/env python3
"""
BehaviorSpace 模块 - 最终修复版
修复内容：
1. 修复 numpy 数组比较歧义错误
2. 增强错误处理和验证
3. 优化代码健壮性
"""

from typing import Callable, List, Tuple, Optional, Any, Dict, Union
import numpy as np

# 允许的参数名称
ALLOWED_ARGUMENTS = ["y_true", "y_pred_ensemble", "Y_pred_base_models", "weights",
                     "input_metadata", "y_pred"]

# 允许的预测格式
ALLOWED_PREDICTION_FORMATS = ["raw", "proba", "none"]


class BehaviorFunction:
    """
    Behavior function class for defining individual behavior functions in QDO.
    """
    
    def __init__(self, 
                 function: Callable[..., float], 
                 required_arguments: List[str],
                 range_tuple: Tuple[float, float], 
                 required_prediction_format: str, 
                 name: Optional[str] = None):
        
        # 验证 function 的可调用性
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function
        
        # 验证参数名称
        invalid_args = [arg for arg in required_arguments if arg not in ALLOWED_ARGUMENTS]
        if invalid_args:
            raise ValueError(
                f"Invalid argument names: {invalid_args}. "
                f"Allowed arguments are: {ALLOWED_ARGUMENTS}"
            )
        self.required_arguments = required_arguments
        
        # 验证范围元组
        if (not isinstance(range_tuple, tuple) or 
            len(range_tuple) != 2 or 
            not all(isinstance(x, (int, float)) for x in range_tuple) or
            range_tuple[0] >= range_tuple[1]):
            raise ValueError(
                "range_tuple must be a tuple of two values (lower, upper) where lower < upper"
            )
        self.range_tuple = range_tuple
        
        # 验证预测格式
        if required_prediction_format not in ALLOWED_PREDICTION_FORMATS:
            raise ValueError(
                f"Invalid prediction format: {required_prediction_format}. "
                f"Allowed formats are: {ALLOWED_PREDICTION_FORMATS}"
            )
        self.required_prediction_format = required_prediction_format
        
        # 设置名称和元数据要求
        self.name = name or f"BehaviorFunction_{id(self)}"
        self.requires_base_model_metadata = "input_metadata" in required_arguments
    
    def __call__(self, **kwargs) -> float:
        """
        Execute the behavior function with the provided arguments.
        """
        try:
            # 提取所需的参数
            func_args = {arg: kwargs[arg] for arg in self.required_arguments}
            return self.function(**func_args)
        except KeyError as e:
            raise ValueError(f"Missing required argument: {e}")
        except Exception as e:
            raise RuntimeError(f"Behavior function '{self.name}' execution failed: {e}")
    
    def __repr__(self) -> str:
        return f"BehaviorFunction(name={self.name}, arguments={self.required_arguments})"


class BehaviorSpace:
    """
    A class for managing behavior functions in Quality-Diversity Optimization (QDO).
    """
    
    def __init__(self, behavior_functions: List[BehaviorFunction]):
        if not behavior_functions:
            raise ValueError("BehaviorSpace must contain at least one BehaviorFunction")
        
        # 验证所有元素都是 BehaviorFunction 实例
        for i, bf in enumerate(behavior_functions):
            if not isinstance(bf, BehaviorFunction):
                raise TypeError(
                    f"All elements must be BehaviorFunction instances. "
                    f"Got {type(bf)} at index {i}"
                )
        
        self.behavior_functions = behavior_functions
    
    @property
    def ranges(self) -> List[Tuple[float, float]]:
        """Get the ranges for all behavior functions."""
        return [bf.range_tuple for bf in self.behavior_functions]
    
    @property
    def n_dims(self) -> int:
        """Get the number of dimensions in the behavior space."""
        return len(self.behavior_functions)
    
    @property
    def required_prediction_types(self) -> set:
        """Get the set of required prediction formats."""
        return set(bf.required_prediction_format for bf in self.behavior_functions)
    
    @property
    def requires_base_model_metadata(self) -> bool:
        """Check if any behavior function requires base model metadata."""
        return any(bf.requires_base_model_metadata for bf in self.behavior_functions)
    
    def _is_none_tuple(self, tup: tuple) -> bool:
        """
        安全检查元组是否全为 None
        
        Parameters
        ----------
        tup : tuple
            要检查的元组
            
        Returns
        -------
        bool
            如果元组中所有元素都是 None，返回 True
        """
        return all(item is None for item in tup)
    
    def __call__(self, 
                 weights: np.ndarray, 
                 y_true: np.ndarray,
                 raw_preds: Optional[Tuple[np.ndarray, List[np.ndarray]]] = (None, None),
                 proba_preds: Optional[Tuple[np.ndarray, List[np.ndarray]]] = (None, None),
                 input_metadata: Optional[Any] = None) -> List[float]:
        """
        Calculate behavior features for the given inputs.
        """
        # 验证输入参数类型
        if not isinstance(weights, np.ndarray):
            raise TypeError("weights must be a numpy array")
        if not isinstance(y_true, np.ndarray):
            raise TypeError("y_true must be a numpy array")
        
        # 解包预测元组
        raw_y_pred_ensemble, raw_Y_pred_base_models = raw_preds
        proba_y_pred_ensemble, proba_Y_pred_base_models = proba_preds
        
        # 验证至少有一种预测格式可用
        requires_raw = any(bf.required_prediction_format == "raw" for bf in self.behavior_functions)
        requires_proba = any(bf.required_prediction_format == "proba" for bf in self.behavior_functions)
        
        # 🔧 修复：使用安全的 None 检查方法
        if requires_raw and self._is_none_tuple(raw_preds):
            raise ValueError("Raw predictions are required but not provided")
        
        if requires_proba and self._is_none_tuple(proba_preds):
            raise ValueError("Probability predictions are required but not provided")
        
        # 准备参数字典
        potential_args = {
            "y_true": y_true,
            "weights": weights,
            "input_metadata": input_metadata,
            "y_pred_ensemble": None,
            "Y_pred_base_models": None,
            "y_pred": None
        }
        
        # 计算行为特征
        b_space_instance = []
        
        for bf in self.behavior_functions:
            try:
                # 根据预测格式选择参数
                if bf.required_prediction_format == "raw":
                    potential_args["y_pred_ensemble"] = raw_y_pred_ensemble
                    potential_args["Y_pred_base_models"] = raw_Y_pred_base_models
                    potential_args["y_pred"] = raw_y_pred_ensemble
                elif bf.required_prediction_format == "proba":
                    potential_args["y_pred_ensemble"] = proba_y_pred_ensemble
                    potential_args["Y_pred_base_models"] = proba_Y_pred_base_models
                    potential_args["y_pred"] = proba_y_pred_ensemble
                # 对于"none"格式，不需要设置预测参数
                
                # 提取行为函数所需的参数
                func_args = {arg: potential_args[arg] for arg in bf.required_arguments}
                value = bf(**func_args)
                b_space_instance.append(float(value))
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to compute behavior function '{bf.name}': {e}"
                )
        
        return b_space_instance
    
    def __repr__(self) -> str:
        return f"BehaviorSpace(n_dims={self.n_dims}, functions={[bf.name for bf in self.behavior_functions]})"


# ==================== 实用函数 ====================

def validate_behavior_space(behavior_space: BehaviorSpace) -> Tuple[bool, str]:
    """
    Validate a BehaviorSpace instance and return detailed results.
    """
    try:
        # 检查行为函数列表
        if not behavior_space.behavior_functions:
            return False, "BehaviorSpace must contain at least one BehaviorFunction"
        
        # 检查每个行为函数
        for i, bf in enumerate(behavior_space.behavior_functions):
            if not isinstance(bf, BehaviorFunction):
                return False, f"Element at index {i} is not a BehaviorFunction instance"
            
            # 检查参数名称
            for arg in bf.required_arguments:
                if arg not in ALLOWED_ARGUMENTS:
                    return False, f"Invalid argument name '{arg}' in function '{bf.name}'"
            
            # 检查预测格式
            if bf.required_prediction_format not in ALLOWED_PREDICTION_FORMATS:
                return False, f"Invalid prediction format '{bf.required_prediction_format}' in function '{bf.name}'"
            
            # 检查范围元组
            if (not isinstance(bf.range_tuple, tuple) or 
                len(bf.range_tuple) != 2 or 
                not all(isinstance(x, (int, float)) for x in bf.range_tuple) or
                bf.range_tuple[0] >= bf.range_tuple[1]):
                return False, f"Invalid range tuple {bf.range_tuple} in function '{bf.name}'"
        
        return True, "BehaviorSpace is valid"
    
    except Exception as e:
        return False, f"Validation failed with error: {e}"


# ==================== 修复版示例用法 ====================

if __name__ == "__main__":
    """修复版示例用法和测试"""
    
    # 示例行为函数
    def example_diversity_metric(y_true, Y_pred_base_models):
        """示例多样性度量函数"""
        # 简化实现：计算预测之间的平均差异
        differences = []
        for i in range(len(Y_pred_base_models)):
            for j in range(i + 1, len(Y_pred_base_models)):
                # 🔧 修复：使用 numpy 的安全比较
                diff = np.mean(np.abs(Y_pred_base_models[i] - Y_pred_base_models[j]))
                differences.append(diff)
        return np.mean(differences) if differences else 0.0
    
    def example_accuracy_metric(y_true, y_pred_ensemble):
        """示例准确率度量函数"""
        # 🔧 修复：使用 numpy 的安全比较
        return np.mean(y_true == y_pred_ensemble)
    
    # 创建行为函数实例
    try:
        diversity_func = BehaviorFunction(
            function=example_diversity_metric,
            required_arguments=["y_true", "Y_pred_base_models"],
            range_tuple=(0.0, 1.0),
            required_prediction_format="raw",
            name="DiversityMetric"
        )
        
        accuracy_func = BehaviorFunction(
            function=example_accuracy_metric,
            required_arguments=["y_true", "y_pred_ensemble"],
            range_tuple=(0.0, 1.0),
            required_prediction_format="raw",
            name="AccuracyMetric"
        )
        
        # 创建行为空间
        behavior_space = BehaviorSpace([diversity_func, accuracy_func])
        
        print(f"✅ Created BehaviorSpace with {behavior_space.n_dims} dimensions")
        print(f"   Ranges: {behavior_space.ranges}")
        print(f"   Required prediction types: {behavior_space.required_prediction_types}")
        
        # 测试验证函数
        is_valid, message = validate_behavior_space(behavior_space)
        print(f"   Validation: {'✅' if is_valid else '❌'} {message}")
        
        # 测试调用
        weights = np.array([0.5, 0.5])
        y_true = np.array([0, 1, 0, 1])
        y_pred_ensemble = np.array([0, 1, 0, 1])
        Y_pred_base_models = [
            np.array([0, 1, 0, 0]),
            np.array([0, 1, 1, 1])
        ]
        
        result = behavior_space(
            weights=weights,
            y_true=y_true,
            raw_preds=(y_pred_ensemble, Y_pred_base_models)
        )
        print(f"✅ BehaviorSpace execution successful: {result}")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()