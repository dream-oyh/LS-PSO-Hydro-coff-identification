import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

pose_data = pd.read_csv("pose_data.csv").to_numpy()
thruster_data = pd.read_csv("thruster_data.csv").to_numpy()

r_f = np.array(
    [
        [0.707, 0.707],
        [0.707, -0.707],
        [-0.707, 0.707],
        [-0.707, -0.707],
    ]
)
r_p = np.array(
    [
        [0.1355, -0.1],
        [0.1355, 0.1],
        [-0.1475, -0.1],
        [-0.1475, 0.1],
    ]
)

model_name = "bluerov"
mass = 11.2
rho = 10**3
L = 0.448

linear_vel = pose_data[:, 8:11]
angular_vel = pose_data[:, 11:14]
timestamps = pose_data[:, 0]
thruster = thruster_data[:-1, 1:5]

# 坐标系转换
q = pose_data[:, 4:8]
r = R.from_quat(q).as_matrix()
euler = R.from_quat(q).as_euler("xyz", degrees=False)
angular_matrix = []
linear_matrix = []
for i in range(len(euler)):
    euler_0 = euler[i, :]
    phi = euler_0[0]
    theta = euler_0[1]
    psai = euler_0[2]

    J_2 = [
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
    ]
    angular_matrix.append(J_2)

    J_1 = [
        [
            np.cos(psai) * np.cos(theta),
            -np.sin(psai) * np.cos(phi) + np.cos(psai) * np.sin(theta) * np.sin(phi),
            np.sin(psai) * np.sin(phi) + np.cos(psai) * np.cos(phi) * np.sin(theta),
        ],
        [
            np.sin(psai) * np.cos(theta),
            np.cos(psai) * np.cos(phi) + np.sin(phi) * np.sin(theta) * np.sin(psai),
            -np.cos(psai) * np.sin(phi) + np.sin(theta) * np.sin(psai) * np.cos(phi),
        ],
        [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)],
    ]
    linear_matrix.append(J_1)

angular_rotation = np.stack(angular_matrix, axis=0)
linear_rotation = np.stack(linear_matrix, axis=0)

linear_vel = np.matmul(
    np.linalg.inv(linear_rotation), linear_vel[..., np.newaxis]
).squeeze(-1)
angular_vel = np.matmul(
    np.linalg.inv(angular_rotation), angular_vel[..., np.newaxis]
).squeeze(-1)

linear_acc = np.diff(linear_vel, axis=0) / np.diff(timestamps)[:, np.newaxis]
angular_acc = np.diff(angular_vel, axis=0) / np.diff(timestamps)[:, np.newaxis]

# 对齐时间数据
linear_vel = linear_vel[:-1, :]
angular_vel = angular_vel[:-1, :]
timestamps = timestamps[:-1]

thruster_0 = np.stack([thruster[:, 0] * 0.707, thruster[:, 0] * 0.707], axis=-1)
thruster_1 = np.stack([thruster[:, 1] * 0.707, -thruster[:, 1] * 0.707], axis=-1)
thruster_2 = np.stack([-thruster[:, 2] * 0.707, thruster[:, 2] * 0.707], axis=-1)
thruster_3 = np.stack([-thruster[:, 3] * 0.707, -thruster[:, 3] * 0.707], axis=-1)

moment_z = (
    np.cross(r_p[0, :], thruster_0)
    + np.cross(r_p[1, :], thruster_1)
    + np.cross(r_p[2, :], thruster_2)
    + np.cross(r_p[3, :], thruster_3)
)

F_p = np.dot(thruster, r_f)


u = linear_vel[:, 0]
v = linear_vel[:, 1]
r = angular_vel[:, 2]
u_acc = linear_acc[:, 0]
v_acc = linear_acc[:, 1]
r_acc = angular_acc[:, 2]

mass = 11.2
rho = 10**3
L = 0.448
J_z = 0.2 * mass * 0.448**2 + 0.2 * mass * 0.2384**2

hydro_coff_true = np.array([-1.7182, 0, -0.4006, -11.7391, 0, -20, 0, -5, 0])


def fitness_function(
    u: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    r_pred: np.ndarray,
):
    return np.sum(
        np.abs((u - u_pred) / (u))
        + np.abs((v - v_pred) / (v))
        + np.abs((r - r_pred) / (r))
    )


def forward(hydro_coff):  # 单个粒子中依据动力学方程预测速度

    X_udot = hydro_coff[0]
    Y_vdot = hydro_coff[1]
    N_rdot = hydro_coff[2]
    X_u = hydro_coff[3]
    X_uabsu = hydro_coff[4]
    Y_v = hydro_coff[5]
    Y_vabsv = hydro_coff[6]
    N_r = hydro_coff[7]
    N_rabsr = hydro_coff[8]

    u_0 = u[0]
    v_0 = v[0]
    r_0 = r[0]
    u_pred = []
    v_pred = []
    r_pred = []
    r_acc = []

    A = np.array(
        [
            (mass - Y_vdot) / (mass - X_udot),
            X_u / (mass - X_udot),
            X_uabsu / (mass - X_udot),
            1 / (mass - X_udot),
        ]
    )

    B = np.array(
        [
            (X_udot - mass) / (mass - Y_vdot),
            Y_v / (mass - Y_vdot),
            Y_vabsv / (mass - Y_vdot),
            1 / (mass - Y_vdot),
        ]
    )

    C = np.array(
        [
            (Y_vdot - X_udot) / (J_z - N_rdot),
            N_r / (J_z - N_rdot),
            N_rabsr / (J_z - N_rdot),
            1 / (J_z - N_rdot),
        ]
    )

    for i in range(len(timestamps)):
        u_acc_pred = (
            A[0] * v_0 * r_0 + A[1] * u_0 + A[2] * u_0 * np.abs(u_0) + A[3] * F_p[i, 0]
        )
        v_acc_pred = (
            B[0] * u_0 * r_0 + B[1] * v_0 + B[2] * v_0 * np.abs(v_0) + B[3] * F_p[i, 1]
        )
        r_acc_pred = (
            C[0] * u_0 * v_0
            + C[1] * r_0
            + C[2] * r_0 * np.abs(r_0)
            + C[3] * moment_z[i]
        )

        u_next = u_0 + 0.05 * u_acc_pred
        v_next = v_0 + 0.05 * v_acc_pred
        r_next = r_0 + 0.05 * r_acc_pred

        u_0 = u_next
        v_0 = v_next
        r_0 = r_next

        u_pred.append(u_next)
        v_pred.append(v_next)
        r_pred.append(r_next)
        r_acc.append(r_acc_pred)

    u_pred = np.array(u_pred)
    v_pred = np.array(v_pred)
    r_pred = np.array(r_pred)
    r_acc = np.array(r_acc)

    return u_pred, v_pred, r_pred


N = 200

hydro_coff = np.array(
    [
        -3.17,
        -0.36,
        -0.572,
        -10.495,
        -6.633,
        -18.7589,
        -7.7811,
        -4.49,
        -1.90,
    ]
)
k = 2
vel_scale = 0.2
vel_weight = 0.3
par_pos = []  # 粒子位置
par_vel = []  # 粒子速度
par_pbest = []
par_best_fitness = []
# 初始化

X_max = hydro_coff + k * np.abs(hydro_coff)
X_min = hydro_coff - k * np.abs(hydro_coff)
V_max = vel_scale * np.abs(hydro_coff)
V_min = -vel_scale * np.abs(hydro_coff)

for i in range(N):
    x_0 = np.random.rand(len(hydro_coff)) * (2 * k * np.abs(hydro_coff)) + (
        hydro_coff - k * np.abs(hydro_coff)
    )  # 按照水动力系数允许的估计范围随机取粒子初始位置
    x_0 = np.random.rand(len(hydro_coff)) * (X_max - X_min) + X_min
    v_0 = np.random.rand(len(hydro_coff)) * (V_max - V_min) + V_min
    fitness_0 = 100000
    par_pos.append(x_0)  # 记录每个粒子的位置
    par_vel.append(v_0)  # 记录每个粒子的速度
    par_pbest.append(x_0)
    par_best_fitness.append(fitness_0)

gbest_fitness = par_best_fitness[np.argmin(np.abs(np.array(par_best_fitness)))]
gbest_fitness_list = [gbest_fitness]


c_1 = 2
c_2 = 3
epoch = 10000


def fitness_function(
    u: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    r_pred: np.ndarray,
):
    return np.sum(
        np.abs((u - u_pred) / (u + 1e-6))
        + np.abs((v - v_pred) / (v + 1e-6))
        + np.abs((r - r_pred) / (r + 1e-6))
    )


for i in range(epoch):
    start = time.time()
    for j in range(N):

        par_pos_j = par_pos[j]
        par_vel_j = par_vel[j]

        u_pred, v_pred, r_pred = forward(par_pos_j)

        current_fitness = fitness_function(u, v, r, u_pred, v_pred, r_pred)

        if np.abs(current_fitness) < np.abs(
            par_best_fitness[j]
        ):  # 如果到了目前最好的适应度位置
            par_best_fitness[j] = current_fitness
            par_pbest[j] = par_pos_j  # 更新目前个体最好的位置
        minind = np.argmin(np.abs(np.array(par_best_fitness)))
        gbest_fitness = par_best_fitness[minind]  # 更新全局最优适合度
        gbest_pos = par_pbest[minind]  # 更新全局最优位置

        if gbest_fitness != gbest_fitness_list[-1]:
            # plt.figure(2)
            # plt.subplot(3, 1, 1)
            # plt.title(f"the {i} epoch, the {minind} partical, the fit: {gbest_fitness}")
            # plt.plot(timestamps, u, "b--", label="truth")
            # plt.plot(timestamps, u_pred, "r-", label="prediction")
            # plt.ylabel("u")
            # plt.subplot(3, 1, 2)
            # plt.plot(timestamps, v, "b--", label="truth")
            # plt.plot(timestamps, v_pred, "r-", label="prediction")
            # plt.ylabel("v")
            # plt.subplot(3, 1, 3)
            # plt.plot(timestamps, r, "b--", label="truth")
            # plt.plot(timestamps, r_pred, "r-", label="prediction")
            # plt.ylabel("r")
            # plt.xlabel("t")

            # plt.legend()
            # plt.show()
            print(gbest_pos)

        gbest_fitness_list.append(gbest_fitness)
        delta_v = c_1 * np.random.rand(len(hydro_coff)) * (
            par_pbest[j] - par_pos_j
        ) + c_2 * np.random.rand(len(hydro_coff)) * (gbest_pos - par_pos_j)
        delta_v = np.clip(
            delta_v, -vel_scale * np.abs(par_pos_j), vel_scale * np.abs(par_pos_j)
        )

        par_vel_j = vel_weight * par_vel_j + delta_v
        par_pos_j += par_vel_j
        par_pos_j = np.clip(
            par_pos_j,
            hydro_coff - k * np.abs(hydro_coff),
            hydro_coff + k * np.abs(hydro_coff),
        )

        par_pos[j] = par_pos_j
        par_vel[j] = par_vel_j

    print(
        f"epoch {i+1}: the global best fitness is {gbest_fitness}, time:{time.time()-start}"
    )


gbest_fitness_list = np.array(gbest_fitness_list)

print(f"the best global hydro coff: {gbest_pos}")

plt.figure(1)
plt.plot(gbest_fitness_list, label="global best fitness")

# u_pred, v_pred, r_pred = pred_vel(gbest_pos)
# plt.figure(2)
# plt.subplot(3, 1, 1)
# plt.plot(timestamps, u, "b--", label="truth")
# plt.plot(timestamps, u_pred, "r-", label="prediction")
# plt.ylabel("u")
# plt.subplot(3, 1, 2)
# plt.plot(timestamps, v, "b--", label="truth")
# plt.plot(timestamps, v_pred, "r-", label="prediction")
# plt.ylabel("v")
# plt.subplot(3, 1, 3)
# plt.plot(timestamps, r, "b--", label="truth")
# plt.plot(timestamps, r_pred, "r-", label="prediction")
# plt.ylabel("r")
# plt.xlabel("t")
# plt.legend()
# plt.show()
