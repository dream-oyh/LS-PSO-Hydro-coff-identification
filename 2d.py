import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from calculate import calculate_forces_and_torques

pose_dir = "3d_data/pose_data_3d.csv"
thruster_dir = "3d_data/thrust_data_3d.csv"

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


model_name = "bluerov"
mass = 11.2
rho = 10**3
L = 0.448

linear_vel = pose_data[:, 8:11]
angular_vel = pose_data[:, 11:14]
timestamps = pose_data[:, 0]
thruster = thruster_data[:-1, 1:7]
tau = calculate_forces_and_torques(r_p, r_f, thruster)
F_p = tau[:,:2]
moment_z = tau[:,5]
# 坐标系转换
q = pose_data[:, 4:8]
r = R.from_quat(q).as_matrix()
euler = R.from_quat(q).as_euler("xyz")
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
linear_vel_true = linear_vel[:-1, :]
angular_vel_true = angular_vel[:-1, :]
linear_vel = linear_vel_true + np.random.normal(0, 0.01)
angular_vel = angular_vel_true + np.random.normal(0, 0.01)

timestamps = timestamps[:-1]

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
        # r_acc.append(r_acc_pred)

    u_pred = np.array(u_pred)
    v_pred = np.array(v_pred)
    r_pred = np.array(r_pred)
    r_acc = np.array(r_acc)

    return u_pred, v_pred, r_pred


# y_a = A * K_a
K_a = np.stack([v * r, u, u * np.abs(u), F_p[:, 0]], axis=-1)
y_a = u_acc
A = np.linalg.lstsq(K_a, y_a, rcond=None)[0]
rmse_a = np.sqrt(np.mean((K_a @ A - y_a) ** 2))

X_udot1 = mass - 1 / A[3]
Y_vdot1 = mass - A[0] * (mass - X_udot1)
X_u = A[1] * (mass - X_udot1)
X_uabsu = A[2] * (mass - X_udot1)

# B
K_b = np.stack([u * r, v, v * np.abs(v), F_p[:, 1]], axis=-1)
y_b = v_acc
B = np.linalg.lstsq(K_b, y_b, rcond=None)[0]
rmse_b = np.sqrt(np.mean((K_b @ B - y_b) ** 2))

Y_vdot2 = mass - 1 / B[3]
X_udot2 = mass + B[0] * (mass - Y_vdot2)
Y_v = B[1] * (mass - Y_vdot2)
Y_vabsv = B[2] * (mass - Y_vdot2)

# C
K_c = np.stack([v * u, r, r * np.abs(r), moment_z[:]], axis=-1)
y_c = r_acc
C = np.linalg.lstsq(K_c, y_c, rcond=None)[0]
rmse_c = np.sqrt(np.mean((K_c @ C - y_c) ** 2))

N_rdot = J_z - 1 / C[3]
Y_vdot3 = (X_udot1 + X_udot2) / 2 + C[0] * (J_z - N_rdot)
X_udot3 = (Y_vdot1 + Y_vdot2) / 2 - C[0] * (J_z - N_rdot)
N_r = C[1] * (J_z - N_rdot)
N_rabsr = C[2] * (J_z - N_rdot)


print(f"X_udot:{(X_udot1+X_udot2+X_udot3)/3}")
print(f"Y_vdot:{(Y_vdot1+Y_vdot2+Y_vdot3)/3}")
print(f"N_rdot:{N_rdot}")
print(f"X_u:{X_u}")
print(f"X_uabsu:{X_uabsu}")
print(f"Y_v:{Y_v}")
print(f"Y_vabsv:{Y_vabsv}")
print(f"N_r:{N_r}")
print(f"N_rabsr:{N_rabsr}")

# print("Relative ERROR:")
# print(
#     f"X_udot:{((X_udot1 + X_udot2 + X_udot3) / 3 - hydro_coff_true[0]) / hydro_coff_true[0] * 100}%"
# )
# print(
#     f"Y_vdot:{(Y_vdot1 + Y_vdot2 + Y_vdot3) / 3 * 100}%"
# )
# print(f"N_rdot:{(N_rdot - hydro_coff_true[2]) / hydro_coff_true[2] * 100}%")
# print(f"X_u:{(X_u - hydro_coff_true[3]) / hydro_coff_true[3] * 100}%")
# print(f"X_uabsu:{X_uabsu*100}%")
# print(f"Y_v:{(Y_v - hydro_coff_true[5]) / hydro_coff_true[5] * 100}%")
# print(f"Y_vabsv:{Y_vabsv*100}%")
# print(f"N_r:{(N_r - hydro_coff_true[7]) / hydro_coff_true[7] * 100}%")
# print(f"N_rabsr:{N_rabsr*100}%")


hydro_coff = np.array(
    [
        X_udot1,
        Y_vdot1,
        N_rdot,
        X_u,
        X_uabsu,
        Y_v,
        Y_vabsv,
        N_r,
        N_rabsr,
    ]
)

u_pred, v_pred, r_pred = forward(hydro_coff_true)
fitness = fitness_function(u, v, r, u_pred, v_pred, r_pred)
plt.figure()
plt.title(f"the estimation error of LS is {fitness}")
plt.subplot(3, 1, 1)
plt.plot(timestamps, linear_vel_true[:, 0], "b--", label="truth")
plt.plot(timestamps, u_pred, "r-", label="prediction")
plt.ylabel("u")
plt.subplot(3, 1, 2)
plt.plot(timestamps, linear_vel_true[:, 1], "b--", label="truth")
plt.plot(timestamps, v_pred, "r-", label="prediction")
plt.ylabel("v")
plt.subplot(3, 1, 3)
plt.plot(timestamps, angular_vel_true[:, 2], "b--", label="truth")
plt.plot(timestamps, r_pred, "r-", label="prediction")
plt.ylabel("r")
plt.xlabel("t")
plt.legend()
plt.show()
