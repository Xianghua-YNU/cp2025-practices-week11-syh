# -*- coding: utf-8 -*-
"""
学生代码模板：计算伽马函数 Gamma(a)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import factorial, sqrt, pi, exp, log

# --- Task 1: 绘制被积函数 ---

def integrand_gamma(x, a):
    """
    计算伽马函数的原始被积函数: f(x, a) = x^(a-1) * exp(-x)

    Args:
        x (float or np.array): 自变量值。
        a (float): 伽马函数的参数。

    Returns:
        float or np.array: 被积函数在 x 处的值。

    Hints:
        - 需要处理 x=0 的情况 (根据 a 的值可能为 0, 1, 或 inf)。
        - 对于 x > 0, 考虑使用 exp((a-1)*log(x) - x) 来提高数值稳定性。
    """
    if x < 0:
        return 0.0  # 或者抛出错误，因为积分区间是 [0, inf)

    if x == 0:
        if a > 1:
            return 0.0
        elif a == 1:
            return 1.0
        else:
            return np.inf
    elif x > 0:
        try:
            log_f = (a - 1) * log(x) - x
            return exp(log_f)
        except ValueError:
            return np.nan  # 处理可能的计算错误
    else:  # 理论上不会进入这里
        return np.nan


def plot_integrands():
    """绘制 a=2, 3, 4 时的被积函数图像"""
    x_vals = np.linspace(0.01, 10, 400)  # 从略大于0开始
    plt.figure(figsize=(10, 6))

    print("绘制被积函数图像...")
    for a_val in [2, 3, 4]:
        print(f"  计算 a = {a_val}...")
        y_vals = np.array([integrand_gamma(x, a_val) for x in x_vals])
        valid_indices = np.isfinite(y_vals)
        plt.plot(x_vals[valid_indices], y_vals[valid_indices], label=f'$a = {a_val}$')

        peak_x = a_val - 1
        if peak_x > 0:
            peak_y = integrand_gamma(peak_x, a_val)
            plt.plot(peak_x, peak_y, 'o', ms=5)

    plt.xlabel("$x$")
    plt.ylabel("$f(x, a) = x^{a-1} e^{-x}$")
    plt.title("Integrand of the Gamma Function")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlim(left=0)


# --- Task 2 & 3: 解析推导 (在注释或报告中完成) ---
# Task 2: 峰值位置推导
# 对 f(x, a) = x^(a - 1) * exp(-x) 求导，根据求导公式 (uv)' = u'v + uv'，
# 令 u = x^(a - 1)，v = exp(-x)，则 u' = (a - 1) * x^(a - 2)，v' = -exp(-x)。
# f'(x, a) = (a - 1) * x^(a - 2) * exp(-x) - x^(a - 1) * exp(-x) = x^(a - 2) * exp(-x) * (a - 1 - x)。
# 令 f'(x, a) = 0，因为 x^(a - 2) * exp(-x) 恒大于 0（x > 0），所以 a - 1 - x = 0，解得 x = a - 1。
# 结果: x = a - 1

# Task 3: 变量代换 z = x/(c+x)
# 1. 当 z = 1/2 时，1/2 = x/(c + x)，交叉相乘得 c + x = 2x，解得 x = c。
#    结果: x = c
# 2. 为使峰值 x = a - 1 映射到 z = 1/2，将 x = a - 1 代入 z = x/(c + x) 中，1/2 = (a - 1)/(c + a - 1)，
#    交叉相乘得 c + a - 1 = 2(a - 1)，解得 c = a - 1。
#    结果: c = a - 1

# --- Task 4: 实现伽马函数计算 ---

def transformed_integrand_gamma(z, a):
    """
    计算变换后的被积函数 g(z, a) = f(x(z), a) * dx/dz
    其中 x = cz / (1-z), dx/dz = c / (1-z)^2, 且 c = a-1

    Args:
        z (float or np.array): 变换后的自变量 (积分区间 [0, 1])。
        a (float): 伽马函数的参数。

    Returns:
        float or np.array: 变换后的被积函数值。

    Hints:
        - 这个变换主要对 a > 1 有意义 (c > 0)。需要考虑如何处理 a <= 1 的情况。
        - 计算 f(x(z), a) 时可以调用上面实现的 integrand_gamma 函数。
        - 处理 z=0 和 z=1 的边界情况。
    """
    c = a - 1.0
    if c <= 0:
        return 0.0

    if z < 0 or z > 1:
        return 0.0
    if z == 1:
        return 0.0
    if z == 0:
        return integrand_gamma(0, a) * c

    x = c * z / (1 - z)
    dxdz = c / ((1 - z) ** 2)
    val_f = integrand_gamma(x, a)

    if not np.isfinite(val_f) or not np.isfinite(dxdz):
        return 0.0

    return val_f * dxdz


def gamma_function(a):
    """
    计算 Gamma(a) 使用数值积分。

    Args:
        a (float): 伽马函数的参数。

    Returns:
        float: Gamma(a) 的计算值。

    Hints:
        - 检查 a <= 0 的情况。
        - 考虑对 a > 1 使用变换后的积分 (transformed_integrand_gamma, 区间 [0, 1])。
        - 考虑对 a <= 1 使用原始积分 (integrand_gamma, 区间 [0, inf])，因为变换推导不适用。
        - 使用导入的数值积分函数 (例如 `quad`)。
    """
    if a <= 0:
        print(f"警告: Gamma(a) 对 a={a} <= 0 无定义 (或为复数)。")
        return np.nan

    try:
        if a > 1.0:
            result, error = quad(transformed_integrand_gamma, 0, 1, args=(a,))
        else:
            result, error = quad(integrand_gamma, 0, np.inf, args=(a,))

        return result

    except Exception as e:
        print(f"计算 Gamma({a}) 时发生错误: {e}")
        return np.nan


# --- 主程序 ---
def test_gamma():
    """测试伽马函数的计算结果"""
    # 测试Γ(3/2)
    a_test = 1.5
    result = gamma_function(a_test)
    expected = np.sqrt(np.pi) / 2
    relative_error = abs(result - expected) / expected if expected != 0 else 0
    print(f"Γ({a_test}) = {result:.8f} (精确值: {expected:.8f}, 相对误差: {relative_error:.2e})")

    # 测试整数值
    test_values = [3, 6, 10]
    print("\n测试整数值：")
    print("-" * 60)
    print("a\t计算值 Γ(a)\t精确值 (a-1)!\t相对误差")
    print("-" * 60)
    for a in test_values:
        result = gamma_function(a)
        factorial_val = float(factorial(a - 1))
        relative_error = abs(result - factorial_val) / factorial_val if factorial_val != 0 else 0
        print(f"{a}\t{result:<12.6e}\t{factorial_val:<12.0f}\t{relative_error:.2e}")
    print("-" * 60)


def main():
    # 绘制原始被积函数
    plot_integrands()

    # 运行测试
    test_gamma()


if __name__ == "__main__":
    main()
