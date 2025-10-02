"""
Rust实现的Brent优化算法包装器
"""
import os
import sys
from typing import Callable, Tuple, Optional

# 尝试导入Rust编译的模块
from .brent_rust import brent as brent_rust

def brent(f: Callable[[float], float], 
            bound: Tuple[float, float], 
            tol: float = 1e-6, 
            maxiter: int = 100) -> float:
    """
    Rust实现的Brent优化算法
    
    Parameters:
    -----------
    f : callable
        目标函数 f(x)
    bound : tuple
        搜索区间 (a, b)
    tol : float, optional
        容差，默认1e-6
    maxiter : int, optional
        最大迭代次数，默认100
        
    Returns:
    --------
    float
        找到的最小值点
    """
    return brent_rust(f, bound, tol, maxiter)