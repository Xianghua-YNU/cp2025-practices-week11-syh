import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# 设置字体为黑体
plt.rcParams['font.family'] = 'SimHei'

# 物理常数
G = 6.67430e-11  # 万有引力常数 (单位: m^3 kg^-1 s^-2)
particle_mass = 1  # 测试质点的质量 (kg)


def calculate_sigma(length, mass):
    """
    计算薄片的面密度

    参数:
        length: 薄片边长 (m)
        mass: 薄片总质量 (kg)

    返回:
        面密度 (kg/m^2)
    """
    return mass / (length ** 2)


def integrand(x, y, z):
    """
    被积函数，计算引力积分核

    参数:
        x, y: 薄片上点的坐标 (m)
        z: 测试点高度 (m)

    返回:
        积分核函数值
    """
    return z / ((x ** 2 + y ** 2 + z ** 2) ** (3 / 2))


def gauss_legendre_integral(length, z, n_points=100):
    """
    使用高斯-勒让德求积法计算二重积分

    参数:
        length: 薄片边长 (m)
        z: 测试点高度 (m)
        n_points: 积分点数 (默认100)

    返回:
        积分结果值

    提示:
        1. 使用np.polynomial.legendre.leggauss获取高斯点和权重
        2. 将积分区间从[-1,1]映射到[-L/2,L/2]
        3. 实现双重循环计算二重积分
    """
    x_points, x_weights = np.polynomial.legendre.leggauss(n_points)
    y_points, y_weights = np.polynomial.legendre.leggauss(n_points)

    a, b = -length / 2, length / 2
    integral = 0
    for i in range(n_points):
        for j in range(n_points):
            x = 0.5 * (b - a) * x_points[i] + 0.5 * (b + a)
            y = 0.5 * (b - a) * y_points[j] + 0.5 * (b + a)
            integral += x_weights[i] * y_weights[j] * integrand(x, y, z)

    return integral * (0.5 * (b - a)) ** 2


def calculate_force(length, mass, z, method='gauss'):
    """
    计算给定高度处的引力

    参数:
        length: 薄片边长 (m)
        mass: 薄片质量 (kg)
        z: 测试点高度 (m)
        method: 积分方法 ('gauss'或'scipy')

    返回:
        引力值 (N)
    """
    sigma = calculate_sigma(length, mass)
    if method == 'gauss':
        integral = gauss_legendre_integral(length, z)
    elif method == 'scipy':
        integral, _ = dblquad(integrand, -length / 2, length / 2,
                              lambda x: -length / 2, lambda x: length / 2, args=(z,))
    else:
        raise ValueError("method 必须是 'gauss' 或 'scipy'")
    return G * sigma * particle_mass * integral


def plot_force_vs_height(length, mass, z_min=0.1, z_max=10, n_points=100):
    """
    绘制引力随高度变化的曲线

    参数:
        length: 薄片边长 (m)
        mass: 薄片质量 (kg)
        z_min: 最小高度 (m)
        z_max: 最大高度 (m)
        n_points: 采样点数
    """
    z_values = np.linspace(z_min, z_max, n_points)
    forces_gauss = [calculate_force(length, mass, z, method='gauss') for z in z_values]
    forces_scipy = [calculate_force(length, mass, z, method='scipy') for z in z_values]

    plt.plot(z_values, forces_gauss, label='Gauss-Legendre')
    plt.plot(z_values, forces_scipy, label='SciPy dblquad', linestyle='--')
    plt.axhline(y=G * mass * particle_mass / (z_max ** 2), color='r', linestyle='-.', label='理论极限')
    plt.title('引力随高度变化曲线')
    plt.xlabel('高度 z (m)')
    plt.ylabel('引力 F_z (N)')
    plt.legend()
    plt.grid(True)
    plt.show()


# 示例使用
if __name__ == '__main__':
    # 参数设置 (边长10m，质量1e4kg)
    length = 10
    mass = 1e4

    # 计算并绘制引力曲线
    plot_force_vs_height(length, mass)

    # 打印几个关键点的引力值
    for z in [0.1, 1, 5, 10]:
        F = calculate_force(length, mass, z)
        print(f"高度 z = {z:.1f}m 处的引力 F_z = {F:.3e} N")
