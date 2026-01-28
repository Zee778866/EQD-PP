"""
QDO (Quality-Diversity Optimization) 方法包
包含质量-多样性优化的集成选择算法
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "Assembled Ensembles"
__description__ = "Quality-Diversity Optimized Ensemble Selection"

# 导出主要类和方法
from .behavior_space import BehaviorSpace, BehaviorFunction
from .qdo_es import QDOESEnsembleSelection
from .emitters import DiscreteWeightSpaceEmitter
from .task_adaptive import TaskAdaptiveEnsemble

# 尝试导入自定义存档
try:
    from .custom_archives.custom_sliding_boundaries_archive import SlidingBoundariesArchive
    from .custom_archives.quality_archive import QualityArchive
except ImportError:
    # 回退实现将在使用时动态创建
    pass

# 尝试导入行为函数
try:
    from .behavior_functions import *
except ImportError:
    # 行为函数可能不存在，使用默认实现
    pass

# 包级别配置
PACKAGE_CONFIG = {
    "supported_archive_types": ["sliding", "quality"],
    "supported_emitter_types": ["DiscreteWeightSpaceEmitter"],
    "default_behavior_dims": 3
}

def get_package_info():
    """获取包信息"""
    return {
        "version": __version__,
        "description": __description__,
        "modules": [
            "behavior_space", "qdo_es", "emitters", "task_adaptive"
        ]
    }

# 包初始化代码
print(f"✅ QDO 包加载成功 v{__version__}")

# 允许从包中导入所有内容
__all__ = [
    "BehaviorSpace",
    "BehaviorFunction", 
    "QDOESEnsembleSelection",
    "DiscreteWeightSpaceEmitter",
    "TaskAdaptiveEnsemble",
    "get_package_info"
]
