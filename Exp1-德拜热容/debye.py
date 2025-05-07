import numpy as np
import matplotlib.pyplot as plt

# 物理常数
kB = 1.380649e-23  # 玻尔兹曼常数，单位：J/K

# 样本参数
V = 1000e-6  # 体积，1000立方厘米转换为立方米
rho = 6.022e28  # 原子数密度，单位：m^-3
theta_D = 428  # 德拜温度，单位：K


def integrand(x):
    """被积函数：x^4 * e^x / (e^x - 1)^2

    参数：
    x : float 或 numpy.ndarray
        积分变量

    返回：
    float 或 numpy.ndarray：被积函数的值
    """
    exp_x = np.exp(x)
    return (x ** 4 * exp_x) / ((exp_x - 1) ** 2)


def gauss_quadrature(f, a, b, n):
    """实现高斯-勒让德积分

    参数：
    f : callable
        被积函数
    a, b : float
        积分区间的端点
    n : int
        高斯点的数量

    返回：
    float：积分结果
    """
    x, w = np.polynomial.legendre.leggauss(n)
    y = 0.5 * (b - a) * x + 0.5 * (b + a)
    return 0.5 * (b - a) * np.sum(w * f(y))


def cv(T):
    """计算给定温度T下的热容

    参数：
    T : float
        温度，单位：K

    返回：
    float：热容值，单位：J/K
    """
    integral = gauss_quadrature(integrand, 0, theta_D / T, 50)
    return 9 * V * rho * kB * (T / theta_D) ** 3 * integral


def plot_cv():
    """绘制热容随温度的变化曲线"""
    temperatures = np.linspace(5, 500, 500)
    heat_capacities = [cv(T) for T in temperatures]
    plt.plot(temperatures, heat_capacities)
    plt.xlabel('温度 (K)')
    plt.ylabel('热容 (J/K)')
    plt.title('热容随温度的变化曲线')
    plt.grid(True)
    plt.show()


def test_cv():
    """测试热容计算函数"""
    # 测试一些特征温度点的热容值
    test_temperatures = [5, 100, 300, 500]
    print("\n测试不同温度下的热容值：")
    print("-" * 40)
    print("温度 (K)\t热容 (J/K)")
    print("-" * 40)
    for T in test_temperatures:
        result = cv(T)
        print(f"{T:8.1f}\t{result:10.3e}")


def main():
    # 运行测试
    test_cv()

    # 绘制热容曲线
    plot_cv()


if __name__ == '__main__':
    main()
