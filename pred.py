import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取数据

thruster = pd.read_csv("thruster_data.csv").to_numpy()[:-1, 1:5]
linear_vel = pd.read_csv("pose_data.csv").to_numpy()[:, 8:11]
angular_vel = pd.read_csv("pose_data.csv").to_numpy()[:, 11:14]
timestamps = pd.read_csv("pose_data.csv").to_numpy()[:, 0]


linear_acc = np.diff(linear_vel, axis=0) / np.diff(timestamps)[:, np.newaxis]
angular_acc = np.diff(angular_vel, axis=0) / np.diff(timestamps)[:, np.newaxis]

linear_vel = linear_vel[:-1, :]
angular_vel = angular_vel[:-1, :]
timestamps = timestamps[:-1]

thruster_0 = np.stack([thruster[:, 0] * 0.707, thruster[:, 0] * 0.707], axis=-1)
thruster_1 = np.stack([thruster[:, 1] * 0.707, -thruster[:, 1] * 0.707], axis=-1)
thruster_2 = np.stack([-thruster[:, 2] * 0.707, thruster[:, 2] * 0.707], axis=-1)
thruster_3 = np.stack([-thruster[:, 2] * 0.707, -thruster[:, 2] * 0.707], axis=-1)

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

mass = 11.2
rho = 10**3
L = 0.448
J_z = 0.2 * mass * 0.448**2 + 0.2 * mass * 0.2384**2
N = 30


# fitness: 11488
hydro_coff = np.array(
[ 3.15645871e+00, -1.60622770e-06, -1.58088748e-02, -6.30187922e-02,
  3.40000000e+00, -9.93784818e-02, -4.56827156e-07,  3.22492258e-03,
  1.26750000e+01, -7.93244287e-01,  2.87511067e-02, -3.31421475e-04,
 -3.62095472e-06, -1.17190709e-21, -1.31262652e-04, -6.52246289e-51,
  8.09911684e-02, -1.87500000e-01, -9.25000000e-01]
)

hydro_coff_LS = np.array(
    [
        1.95,
        -0.36,
        -0.088,
        -0.16,
        1.36,
        -4.61,
        -0.017,
        0.18,
        5.07,
        -0.69,
        0.65,
        -1.23,
        -0.071,
        -0.0055,
        -0.016,
        0.10,
        0.075,
        -0.075,
        -0.37,
    ]
)

k = 0.1


def fitness_function(
    u: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    r_pred: np.ndarray,
):
    return np.sum(
        np.abs((u - u_pred) / u) + np.abs((v - v_pred) / v)
    )


def pred_vel(hydro_coff):  # 单个粒子中依据动力学方程预测速度

    X_udot = hydro_coff[0]
    X_uu = hydro_coff[1]
    X_vv = hydro_coff[2]
    X_rr = hydro_coff[3]
    X_vr = hydro_coff[4]
    Y_vdot = hydro_coff[5]
    Y_rdot = hydro_coff[6]
    Y_v = hydro_coff[7]
    Y_r = hydro_coff[8]
    Y_vabsv = hydro_coff[9]
    Y_vabsr = hydro_coff[10]
    Y_rabsr = hydro_coff[11]
    N_rdot = hydro_coff[12]
    N_vdot = hydro_coff[13]
    N_v = hydro_coff[14]
    N_r = hydro_coff[15]
    N_vabsv = hydro_coff[16]
    N_vabsr = hydro_coff[17]
    N_rabsr = hydro_coff[18]

    u_0 = u[0]
    v_0 = v[0]
    r_0 = r[0]
    r_acc_pred = r_acc[0]
    v_acc_pred = v_acc[0]
    u_pred = []
    v_pred = []
    r_pred = []

    A = np.array(
        [
            0.5 * rho * L**2 * X_uu / (mass - 0.5 * rho * L**3 * X_udot),
            0.5 * rho * L**2 * X_vv / (mass - 0.5 * rho * L**3 * X_udot),
            0.5 * rho * L**4 * X_rr / (mass - 0.5 * rho * L**3 * X_udot),
            0.5 * rho * L**3 * X_vr / (mass - 0.5 * rho * L**3 * X_udot),
            1 / (mass - 0.5 * rho * L**3 * X_udot),
        ]
    )

    B = np.array(
        [
            0.5 * rho * L**4 * Y_rdot / (mass - 0.5 * rho * L**3 * Y_vdot),
            0.5 * rho * L**2 * Y_v / (mass - 0.5 * rho * L**3 * Y_vdot),
            0.5 * rho * L**3 * Y_r / (mass - 0.5 * rho * L**3 * Y_vdot),
            0.5 * rho * L**2 * Y_vabsv / (mass - 0.5 * rho * L**3 * Y_vdot),
            0.5 * rho * L**3 * Y_vabsr / (mass - 0.5 * rho * L**3 * Y_vdot),
            0.5 * rho * L**4 * Y_rabsr / (mass - 0.5 * rho * L**3 * Y_vdot),
            -1 / (mass - 0.5 * rho * L**3 * Y_vdot),
        ]
    )

    C = np.array(
        [
            0.5 * rho * L**4 * N_vdot / (J_z - 0.5 * rho * L**5 * N_rdot),
            0.5 * rho * L**3 * N_v / (J_z - 0.5 * rho * L**5 * N_rdot),
            0.5 * rho * L**4 * N_r / (J_z - 0.5 * rho * L**5 * N_rdot),
            0.5 * rho * L**3 * N_vabsv / (J_z - 0.5 * rho * L**5 * N_rdot),
            0.5 * rho * L**4 * N_vabsr / (J_z - 0.5 * rho * L**5 * N_rdot),
            0.5 * rho * L**5 * N_rabsr / (J_z - 0.5 * rho * L**5 * N_rdot),
            1 / (J_z - 0.5 * rho * L**5 * N_rdot),
        ]
    )
    u_pred = []
    v_pred = []
    r_pred = []
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

        u_pred.append(u_acc_pred)
        v_pred.append(v_acc_pred)
        r_pred.append(r_acc_pred)

    u_pred = np.array(u_pred)
    v_pred = np.array(v_pred)
    r_pred = np.array(r_pred)

    return u_pred, v_pred, r_pred


u_pred, v_pred, r_pred = pred_vel(hydro_coff)
current_fitness = fitness_function(u_acc, v_acc, r_acc, u_pred, v_pred, r_pred)

u_pred_LS, v_pred_LS, r_pred_LS = pred_vel(hydro_coff_LS)
current_fitness_LS = fitness_function(u_acc, v_acc, r_acc, u_pred_LS, v_pred_LS, r_pred_LS)
plt.figure(2)
plt.subplot(3, 1, 1)
plt.title("pso fitness %.3f, LS fitness %.3f" % (current_fitness, current_fitness_LS))
plt.plot(timestamps, u_acc, "b--", label="acc_truth")
plt.plot(timestamps, u_pred, "r-", label="prediction_pso")
plt.plot(timestamps, u_pred_LS, "g-", label="prediction_LS")
plt.ylabel("u")
plt.subplot(3, 1, 2)
plt.plot(timestamps, v_acc, "b--", label="acc_truth")
plt.plot(timestamps, v_pred, "r-", label="prediction_pso")
plt.plot(timestamps, v_pred_LS, "g-", label="prediction_LS")
plt.ylabel("v")
plt.subplot(3, 1, 3)
plt.plot(timestamps, r_acc, "b--", label="acc_truth")
plt.plot(timestamps, r_pred, "r-", label="prediction_pso")
plt.plot(timestamps, r_pred_LS, "g-", label="prediction_LS")
plt.ylabel("r")
plt.xlabel("t")
plt.legend()
plt.show()
