import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from calculate import calculate_forces_and_torques
from calculate.transform import transform_velocities




def load_data(pose_dir, thruster_dir):

    pose_data = pd.read_csv(pose_dir).to_numpy()
    thruster_data = pd.read_csv(thruster_dir).to_numpy()

    r_f = np.array(
        [
            [0.707, 0.707, 0],
            [0.707, -0.707, 0],
            [-0.707, 0.707, 0],
            [-0.707, -0.707, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    r_p = np.array(
        [
            [0.1355, -0.1, -0.0725],
            [0.1355, 0.1, -0.0725],
            [-0.1475, -0.1, -0.0725],
            [-0.1475, 0.1, -0.0725],
            [0.0025, -0.1105, -0.005],
            [0.0025, 0.1105, -0.005],
        ]
    )

    m = 11.2

    x_size = 0.448
    y_size = 0.2384
    z_size = 0.28066

    J_x = 0.2 * m * y_size**2 + 0.2 * m * z_size**2
    J_y = 0.2 * m * x_size**2 + 0.2 * m * z_size**2
    J_z = 0.2 * m * x_size**2 + 0.2 * m * y_size**2

    linear_vel = pose_data[:, 8:11]
    angular_vel = pose_data[:, 11:14]
    timestamps = pose_data[:, 0]
    thruster = thruster_data[:-1, 1:7]

    # 计算力和力矩时间序列
    tau = calculate_forces_and_torques(r_p, r_f, thruster)  # √

    # 坐标系转换
    q = pose_data[:, 4:8]
    # 将四元数转换为欧拉角（弧度制） (按照 xyz 顺序，也就是 roll, pitch, yaw)
    # 四元数按照 xyzw 的顺序
    euler_angles = R.from_quat(q).as_euler("xyz", degrees=False)

    # 执行坐标变换
    linear_vel_inertial, angular_vel_inertial = transform_velocities(
        linear_vel, angular_vel, euler_angles
    )

    linear_acc_inertial = (
        np.diff(linear_vel_inertial, axis=0) / np.diff(timestamps)[:, np.newaxis]
    )
    angular_acc_inertial = (
        np.diff(angular_vel_inertial, axis=0) / np.diff(timestamps)[:, np.newaxis]
    )

    # 对齐时间数据
    linear_vel_inertial = linear_vel_inertial[:-1, :]
    angular_vel_inertial = angular_vel_inertial[:-1, :]
    timestamps = timestamps[:-1]

    return (
        m,
        J_x,
        J_y,
        J_z,
        tau,
        linear_vel_inertial,
        angular_vel_inertial,
        linear_acc_inertial,
        angular_acc_inertial,
        timestamps,
    )
