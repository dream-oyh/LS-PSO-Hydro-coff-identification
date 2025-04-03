import numpy as np

def calculate_forces_and_torques(r_positions, r_directions, thrusts):
    n_timesteps = thrusts.shape[0]
    forces = np.zeros((n_timesteps, 3))
    torques = np.zeros((n_timesteps, 3))

    for t in range(n_timesteps):
        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        for i in range(6):  # 遍历 6 个电机
            # 计算推力向量 (方向 * 大小)
            force_vector = r_directions[i] * thrusts[t, i]
            # 累加合力
            total_force += force_vector
            # 计算力矩 (位置叉乘力)
            torque = np.cross(r_positions[i], force_vector)
            total_torque += torque

        forces[t] = total_force
        torques[t] = total_torque


    return np.concatenate([forces, torques],axis=1)


def calculate_J1(phi, theta, psi):

    J1 = np.array(
        [
            [
                np.cos(psi) * np.cos(theta),
                -np.sin(psi) * np.cos(phi) + np.cos(psi) * np.sin(theta) * np.sin(phi),
                np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(phi) * np.sin(theta),
            ],
            [
                np.sin(psi) * np.cos(theta),
                np.cos(psi) * np.cos(phi) + np.sin(phi) * np.sin(theta) * np.sin(psi),
                -np.cos(psi) * np.sin(phi) + np.sin(theta) * np.sin(psi) * np.cos(phi),
            ],
            [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)],
        ]
    )
    return J1


def calculate_J2(phi, theta):

    J2 = np.array(
        [
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
        ]
    )
    return J2


if __name__ == "__main__":
    thrusts = np.array(
        [5.7758111, -0.82137335, -4.03845802, -2.00370368, 5.8010407, 3.78824998]
    )
    print(calculate_forces_and_torques(thrusts))
