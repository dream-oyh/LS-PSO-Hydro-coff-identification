import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

pose_df = pd.read_csv("pose_data.csv").to_numpy()
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

linear_vel = pose_df[:, 8:11]
angular_vel = pose_df[:, 11:14]
timestamps = pose_df[:, 0]
thruster = thruster_data[:-1, 1:5]

# 坐标系转换
q = pose_df[:, 4:8]
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
N = 200

hydro_coff = np.array([-1.7182, 0, -0.4006, -11.7391, 0, -20, 0, -5, 0])  # true

hydro_coff = np.array(
[-1.98595247e+00, -1.08000000e+00, -6.74183771e-01, -7.99131866e+00,
 -1.98990000e+01 ,-2.00878976e+01,  7.75762734e-49, -5.02667415e+00,
 -1.66855312e-05]
)


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

        u_pred.append(u_acc_pred)
        v_pred.append(v_acc_pred)
        r_pred.append(r_acc_pred)
        # r_acc.append(r_acc_pred)

    u_pred = np.array(u_pred)
    v_pred = np.array(v_pred)
    r_pred = np.array(r_pred)
    r_acc = np.array(r_acc)

    return u_pred, v_pred, r_pred


u_pred, v_pred, r_pred = forward(hydro_coff)
current_fitness = fitness_function(u, v, r, u_pred, v_pred, r_pred)


plt.figure(2)
plt.subplot(3, 1, 1)
plt.title("pso fitness %.3f" % (current_fitness))
plt.plot(timestamps, u_acc, "b--", label="truth")
plt.plot(timestamps, u_pred, "r-", label="prediction")
plt.ylabel("u")
plt.subplot(3, 1, 2)
plt.plot(timestamps, v_acc, "b--", label="truth")
plt.plot(timestamps, v_pred, "r-", label="prediction")
plt.ylabel("v")
plt.subplot(3, 1, 3)
plt.plot(timestamps, r_acc, "b--", label="truth")
plt.plot(timestamps, r_pred, "r-", label="prediction")
plt.ylabel("r")
plt.xlabel("t")
plt.legend()
plt.show()
