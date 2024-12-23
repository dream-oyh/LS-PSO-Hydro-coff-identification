import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

linear_acc = np.diff(linear_vel, axis=0) / np.diff(timestamps)[:, np.newaxis]
angular_acc = np.diff(angular_vel, axis=0) / np.diff(timestamps)[:, np.newaxis]

# 对齐时间数据
linear_vel = linear_vel[:-1, :]
angular_vel = angular_vel[:-1, :]
timestamps = timestamps[:-1]

thruster_0 = np.stack([thruster[:, 0] * 0.707, thruster[:, 0] * 0.707], axis=-1)
thruster_1 = np.stack([thruster[:, 1] * 0.707, -thruster[:, 1] * 0.707], axis=-1)
thruster_2 = np.stack([-thruster[:, 2] * 0.707, thruster[:, 2] * 0.707], axis=-1)
thruster_3 = np.stack([-thruster[:, 2] * 0.707, -thruster[:, 2] * 0.707], axis=-1)

F_p = np.dot(thruster, r_f)
moment_z = (
    np.cross(r_p[0, :], thruster_0, 0, 1)
    - np.cross(r_p[1, :], thruster_1, 0, 1)
    - np.cross(r_p[2, :], thruster_2, 0, 1)
    + np.cross(r_p[3, :], thruster_3, 0, 1)
)


u = linear_vel[:, 0]
v = linear_vel[:, 1]
r = angular_vel[:, 2]
u_acc = linear_acc[:, 0]
v_acc = linear_acc[:, 1]
r_acc = angular_acc[:, 2]


# 前向运动辨识
K_a = np.stack([u**2, v**2, r**2, v * r, mass * v * r + F_p[:, 0]], axis=-1)
y_a = u_acc
A = np.linalg.lstsq(K_a, y_a, rcond=None)[0]
rmse_a = np.sqrt(np.mean((K_a @ A - y_a) ** 2))
X_udot = (mass - 1 / A[4]) * 2 / (rho * L**3)
X_uu = A[0] * (mass - 0.5 * rho * L**3 * X_udot) / (0.5 * rho * L**2)
X_vv = A[1] * (mass - 0.5 * rho * L**3 * X_udot) / (0.5 * rho * L**2)
X_rr = A[2] * (mass - 0.5 * rho * L**3 * X_udot) / (0.5 * rho * L**4)
X_vr = A[3] * (mass - 0.5 * rho * L**3 * X_udot) / (0.5 * rho * L**3)

print(f"X_udot:{X_udot}")
print(f"X_uu:{X_uu}")
print(f"X_vv:{X_vv}")
print(f"X_rr:{X_rr}")
print(f"X_vr:{X_vr}")
print(f"rmse_a:{rmse_a}")


# 侧向运动辨识
K_b = np.stack(
    [
        r_acc,
        u * v,
        u * r,
        v * np.abs(v),
        v * np.abs(r),
        r * np.abs(r),
        mass * u * r - F_p[:, 1],
    ],
    axis=-1,
)
y_b = v_acc
B = np.linalg.lstsq(K_b, y_b, rcond=None)[0]
rmse_b = np.sqrt(np.mean((K_b @ B - y_b) ** 2))
Y_vdot = (mass + 1 / B[6]) * 2 / (rho * L**3)
Y_rdot = B[0] * (mass - 0.5 * rho * L**3 * Y_vdot) / (0.5 * rho * L**4)
Y_v = B[1] * (mass - 0.5 * rho * L**3 * Y_vdot) / (0.5 * rho * L**2)
Y_r = B[2] * (mass - 0.5 * rho * L**3 * Y_vdot) / (0.5 * rho * L**3)
Y_vabsv = B[3] * (mass - 0.5 * rho * L**3 * Y_vdot) / (0.5 * rho * L**2)
Y_vabsr = B[4] * (mass - 0.5 * rho * L**3 * Y_vdot) / (0.5 * rho * L**3)
Y_rabsr = B[5] * (mass - 0.5 * rho * L**3 * Y_vdot) / (0.5 * rho * L**4)

print(f"Y_vdot:{Y_vdot}")
print(f"Y_rdot:{Y_rdot}")
print(f"Y_v:{Y_v}")
print(f"Y_r:{Y_r}")
print(f"Y_vabsv:{Y_vabsv}")
print(f"Y_vabsr:{Y_vabsr}")
print(f"Y_rabsr:{Y_rabsr}")
print(f"rmse_b:{rmse_b}")

# 回转运动辨识
K_c = np.stack(
    [v_acc, u * v, u * r, v * np.abs(v), v * np.abs(r), r * np.abs(r), moment_z],
    axis=-1,
)
y_c = r_acc
C = np.linalg.lstsq(K_c, y_c, rcond=None)[0]
rmse_c = np.sqrt(np.mean((K_c @ C - y_c) ** 2))
J_z = 0.2 * mass * 0.448**2 + 0.2 * mass * 0.2384**2
N_rdot = (J_z + 1 / C[6]) * 2 / (rho * L**5)
N_vdot = C[0] * (J_z - 0.5 * rho * L**5 * N_rdot) / (0.5 * rho * L**4)
N_v = C[1] * (J_z - 0.5 * rho * L**5 * N_rdot) / (0.5 * rho * L**3)
N_r = C[2] * (J_z - 0.5 * rho * L**5 * N_rdot) / (0.5 * rho * L**4)
N_vabsv = C[3] * (J_z - 0.5 * rho * L**5 * N_rdot) / (0.5 * rho * L**3)
N_vabsr = C[4] * (J_z - 0.5 * rho * L**5 * N_rdot) / (0.5 * rho * L**4)
N_rabsr = C[5] * (J_z - 0.5 * rho * L**5 * N_rdot) / (0.5 * rho * L**5)
print(f"N_rdot:{N_rdot}")
print(f"N_vdot:{N_vdot}")
print(f"N_v:{N_v}")
print(f"N_r:{N_r}")
print(f"N_vabsv:{N_vabsv}")
print(f"N_vabsr:{N_vabsr}")
print(f"N_rabsr:{N_rabsr}")
print(f"rmse_c:{rmse_c}")

u_0 = u[0]
v_0 = v[0]
r_0 = r[0]
r_acc_pred = r_acc[0]
v_acc_pred = v_acc[0]
u_pred = []
v_pred = []
r_pred = []

print(A)
print(B)
print(C)


# A = np.array([0.71418372, 0.48115741, 0.00819683, 0.91763344, -0.0217232])
# B = np.array(
#     [
#         -0.00100322,
#         0.03779426,
#         -0.47575094,
#         -0.08316492,
#         0.04723739,
#         -0.09548781,
#         -0.00208863,
#     ]
# )
# C = np.array(
#     [
#         -0.09062868,
#         0.09727289,
#         0.74768144,
#         1.92329872,
#         -1.42247587,
#         0.31065917,
#         0.42721038,
#     ]
# )

for i in range(len(timestamps)):

    u_acc_pred = (
        A[0] * u_0**2
        + A[1] * v_0**2
        + A[2] * r_0**2
        + A[3] * v_0 * r_0
        + A[4] * (mass * v_0 * r_0 + F_p[i, 0])
    )

    v_acc_pred = (
        B[0] * r_acc_pred
        + B[1] * u_0 * v_0
        + B[2] * u_0 * r_0
        + B[3] * v_0 * np.abs(v_0)
        + B[4] * v_0 * np.abs(r_0)
        + B[5] * r_0 * np.abs(r_0)
        + B[6] * (mass * u_0 * r_0 - F_p[i, 1])
    )
    r_acc_pred = (
        C[0] * v_acc_pred
        + C[1] * u_0 * v_0
        + C[2] * u_0 * r_0
        + C[3] * v_0 * np.abs(v_0)
        + C[4] * v_0 * np.abs(r_0)
        + C[5] * r_0 * np.abs(r_0)
        + C[6] * moment_z[i]
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

u_pred = np.array(u_pred)
v_pred = np.array(v_pred)
r_pred = np.array(r_pred)


plt.figure()
plt.subplot(3, 1, 1)
plt.plot(timestamps, u, "b--", label="truth")
plt.plot(timestamps, u_pred, "r-", label="prediction")
plt.ylabel("u")
plt.subplot(3, 1, 2)
plt.plot(timestamps, v, "b--", label="truth")
plt.plot(timestamps, v_pred, "r-", label="prediction")
plt.ylabel("v")
plt.subplot(3, 1, 3)
plt.plot(timestamps, r, "b--", label="truth")
plt.plot(timestamps, r_pred, "r-", label="prediction")
plt.ylabel("r")
plt.xlabel("t")
plt.legend()
plt.show()
