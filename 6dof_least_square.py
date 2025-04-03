import numpy as np

from load_data import load_data

pose_dir = "2d_data/pose_data.csv"
thruster_dir = "2d_data/thrust_data.csv"

(
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
) = load_data(pose_dir, thruster_dir)


X, Y, Z, K, M, N = tau[:, 0], tau[:, 1], tau[:, 2], tau[:, 3], tau[:, 4], tau[:, 5]
u, v, w = (
    linear_vel_inertial[:, 0],
    linear_vel_inertial[:, 1],
    linear_vel_inertial[:, 2],
)
p, q, r = (
    angular_vel_inertial[:, 0],
    angular_vel_inertial[:, 1],
    angular_vel_inertial[:, 2],
)
u_dot, v_dot, w_dot = (
    linear_acc_inertial[:, 0],
    linear_acc_inertial[:, 1],
    linear_acc_inertial[:, 2],
)
p_dot, q_dot, r_dot = (
    angular_acc_inertial[:, 0],
    angular_acc_inertial[:, 1],
    angular_acc_inertial[:, 2],
)


F = np.zeros((6 * len(timestamps), 1))
A = np.zeros((6 * len(timestamps), 12))

for i in range(len(timestamps)):
    F[i * 6] = X[i] - m * u_dot[i] - (w[i] * q[i] - v[i] * r[i]) * m
    F[i * 6 + 1] = Y[i] - m * v_dot[i] - (u[i] * r[i] - w[i] * p[i]) * m
    F[i * 6 + 2] = Z[i] - m * w_dot[i] - (v[i] * p[i] - u[i] * q[i]) * m
    F[i * 6 + 3] = K[i] - J_x * p_dot[i] - (J_z - J_y) * q[i] * r[i]
    F[i * 6 + 4] = M[i] - J_y * q_dot[i] - (J_x - J_z) * p[i] * r[i]
    F[i * 6 + 5] = N[i] - J_z * r_dot[i] - (J_y - J_x) * p[i] * q[i]

    A[i * 6] = [-u_dot[i], v[i] * r[i], -w[i] * q[i], 0, 0, 0, -u[i], 0, 0, 0, 0, 0]
    A[i * 6 + 1] = [-u[i] * r[i], -v_dot[i], w[i] * p[i], 0, 0, 0, 0, -v[i], 0, 0, 0, 0]
    A[i * 6 + 2] = [u[i] * q[i], -v[i] * p[i], -w_dot[i], 0, 0, 0, 0, 0, -w[i], 0, 0, 0]
    A[i * 6 + 3] = [
        0,
        v[i] * w[i],
        -v[i] * w[i],
        -p_dot[i],
        q[i] * r[i],
        -q[i] * r[i],
        0,
        0,
        0,
        -p[i],
        0,
        0,
    ]
    A[i * 6 + 4] = [
        -u[i] * w[i],
        0,
        u[i] * w[i],
        -r[i] * p[i],
        -q_dot[i],
        r[i] * p[i],
        0,
        0,
        0,
        0,
        -q[i],
        0,
    ]
    A[i * 6 + 5] = [
        u[i] * v[i],
        -u[i] * v[i],
        0,
        p[i] * q[i],
        -p[i] * q[i],
        -r_dot[i],
        0,
        0,
        0,
        0,
        0,
        -r[i],
    ]

theta = np.linalg.lstsq(A, F, rcond=None)[0]

print(theta)
