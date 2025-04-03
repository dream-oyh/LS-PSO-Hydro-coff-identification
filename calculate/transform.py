import numpy as np

from calculate import calculate_J1, calculate_J2


# 进行坐标变换
def transform_velocities(linear_vel, angular_vel, euler_angles):
    """
    将速度从体坐标系转换到惯性坐标系
    """
    n_timesteps = len(euler_angles)
    linear_vel_inertial = np.zeros_like(linear_vel)
    angular_vel_inertial = np.zeros_like(angular_vel)

    for t in range(n_timesteps):
        phi, theta, psi = euler_angles[t]  # roll, pitch, yaw

        # 计算变换矩阵
        J1 = calculate_J1(phi, theta, psi)
        J2 = calculate_J2(phi, theta)

        # 进行坐标变换 (使用逆矩阵)
        linear_vel_inertial[t] = np.linalg.inv(J1) @ linear_vel[t]
        angular_vel_inertial[t] = np.linalg.inv(J2) @ angular_vel[t]

    return linear_vel_inertial, angular_vel_inertial
