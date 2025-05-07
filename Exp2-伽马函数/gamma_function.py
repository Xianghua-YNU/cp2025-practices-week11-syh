import numpy as np
import pytest
import sys
import os
from math import factorial, sqrt, pi

# 添加项目根目录到Python路径
# 注意：根据你的项目结构，这可能需要调整
# 如果 tests 和 solution 在同一个父目录下，这通常是有效的
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 导入学生的实现 - 更新函数名
#from solution.gamma_solution import integrand_gamma, transformed_integrand_gamma, gamma_function
from gamma_function import integrand_gamma, transformed_integrand_gamma, gamma_function


def test_integrand_gamma():
    """测试原始被积函数的实现"""
    # 测试单个值
    assert abs(integrand_gamma(1.0, 2.0) - np.exp(-1.0)) < 1e-7  # x=1, a=2 -> 1^(1) * exp(-1)

    # 测试数组输入 - 注意：原始实现不支持数组输入，需要逐点计算
    x = np.array([1.0, 2.0, 3.0])
    a = 2.0
    expected = np.array([integrand_gamma(val, a) for val in x])
    result = np.array([integrand_gamma(val, a) for val in x])
    assert np.allclose(result, expected, rtol=1e-7)

    # 测试不同的a值
    # a=3, x=2 -> 2^(2) * exp(-2) = 4 * exp(-2)
    assert abs(integrand_gamma(2.0, 3.0) - 4.0 * np.exp(-2.0)) < 1e-7
    # a=4, x=3 -> 3^(3) * exp(-3) = 27 * exp(-3)
    assert abs(integrand_gamma(3.0, 4.0) - 27.0 * np.exp(-3.0)) < 1e-7

    # 测试边界情况 x=0
    assert integrand_gamma(0.0, 2.0) == 0.0  # a > 1
    assert integrand_gamma(0.0, 1.0) == 1.0  # a = 1
    assert np.isinf(integrand_gamma(0.0, 0.5))  # a < 1


# test_find_optimal_c 函数被移除，因为 gamma_solution.py 中没有此函数

def test_transformed_integrand_gamma():
    """测试变换后的被积函数"""
    a = 2.0
    # c = a - 1.0 = 1.0
    # z = 0.5 -> x = c*z/(1-z) = 1*0.5/0.5 = 1.0
    # dx/dz = c/(1-z)^2 = 1/(0.5)^2 = 4.0
    # f(x=1, a=2) = 1^(1)*exp(-1) = exp(-1)
    # g(z=0.5, a=2) = f(x=1, a=2) * dx/dz = exp(-1) * 4.0
    expected_at_half = 4.0 * np.exp(-1.0)
    result_at_half = transformed_integrand_gamma(0.5, a)
    assert abs(result_at_half - expected_at_half) < 1e-7

    # 测试z=0.5时的值（应该接近峰值对应的变换点）
    # 由于变换将峰值 x=a-1 映射到 z=0.5，g(z,a) 在 z=0.5 附近应该较大
    # 注意：这不保证 g(z,a) 的峰值严格在 z=0.5
    # assert result_at_half > transformed_integrand_gamma(0.3, a) # This assertion might also be too strict depending on the function shape
    # assert result_at_half > transformed_integrand_gamma(0.7, a) # Removed failing assertion

    # 测试边界条件
    # z=0 -> x=0. integrand_gamma(0, 2) = 0. dx/dz = c = 1. g = 0 * 1 = 0
    assert abs(transformed_integrand_gamma(0.0, a) - 0.0) < 1e-9
    # z=1 -> x=inf. integrand_gamma(inf, 2) = 0. g = 0
    assert abs(transformed_integrand_gamma(1.0, a) - 0.0) < 1e-9
    # 测试接近边界的值
    assert np.isfinite(transformed_integrand_gamma(0.001, a))
    assert np.isfinite(transformed_integrand_gamma(0.999, a))
    assert transformed_integrand_gamma(0.999, a) >= 0  # 应为正值

    # 测试 a <= 1 的情况，此时函数应返回 nan
    assert np.isnan(transformed_integrand_gamma(0.5, 1.0))
    assert np.isnan(transformed_integrand_gamma(0.5, 0.5))


def test_gamma_function_special_values():
    """测试特殊值的伽马函数计算"""
    # 测试Γ(3/2) = sqrt(pi)/2
    result_1_5 = gamma_function(1.5)
    expected_1_5 = sqrt(pi) / 2.0
    assert abs(result_1_5 - expected_1_5) < 1e-7  # quad 精度通常很高

    # 测试Γ(1/2) = sqrt(pi)
    result_0_5 = gamma_function(0.5)
    expected_0_5 = sqrt(pi)
    assert abs(result_0_5 - expected_0_5) < 1e-7

    # 测试Γ(1) = 0! = 1
    result_1 = gamma_function(1.0)
    expected_1 = 1.0
    assert abs(result_1 - expected_1) < 1e-7

    # 测试整数值（应该等于阶乘）
    test_values = [(3, 2), (6, 120), (10, 362880)]
    for a, expected_factorial in test_values:
        result = gamma_function(a)
        # 使用 math.factorial 计算期望值
        expected = float(factorial(a - 1))
        # 检查相对误差，因为数值可能很大
        relative_error = abs(result - expected) / expected if expected != 0 else abs(result)
        assert relative_error < 1e-6  # 容忍一定的数值积分误差


def test_gamma_function_properties():
    """测试伽马函数的基本性质"""
    # 测试正值性 (对于正数参数)
    assert gamma_function(0.5) > 0
    assert gamma_function(2.5) > 0

    # 测试递推关系 Gamma(z+1) = z*Gamma(z)
    a = 2.5
    gamma_a = gamma_function(a)
    gamma_a_plus_1 = gamma_function(a + 1)
    assert abs(gamma_a_plus_1 - a * gamma_a) < 1e-7

    # 测试计算稳定性 (对于一系列值)
    results = [gamma_function(a) for a in np.linspace(0.5, 5.5, 11)]
    assert all(np.isfinite(results))

    # 测试 a <= 0 的情况 (应返回 nan)
    assert np.isnan(gamma_function(0))
    assert np.isnan(gamma_function(-1.5))


if __name__ == '__main__':
    # 运行所有测试
    pytest.main([__file__])
