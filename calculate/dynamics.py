import numpy as np


def S(vector):
    assert vector.shape == (3,)
    return np.cross(np.eye(3), vector)


def calculate_coriolis_matrix(mass, vel):
    v1 = vel[:3]
    v2 = vel[3:]
    M11 = mass[0:3, 0:3]
    M12 = mass[0:3, 3:]
    M21 = mass[3:, 0:3]
    M22 = mass[3:, 3:]
    C = np.zeros((6, 6))

    C[0:3, 3:] = -S(np.dot(M11, v1) + np.dot(M12, v2))
    C[3:, 0:3] = C[0:3, 3:]
    C[3:, 3:] = -S(np.dot(M21, v1) + np.dot(M22, v2))
    return C


def calculate_mass_matrix(mass, J_x, J_y, J_z, hydro_coff):
    X_udot, Y_vdot, Z_wdot, K_pdot, M_qdot, N_rdot = hydro_coff[:6]

    M_rb = np.diag([mass, mass, mass, J_x, J_y, J_z])
    M_a = -np.diag([X_udot, Y_vdot, Z_wdot, K_pdot, M_qdot, N_rdot])
    return M_rb, M_a


def calculate_damping_matrix(hydro_coff):
    X_u, Y_v, Z_w, K_p, M_q, N_r = hydro_coff[6:]
    D = -np.diag([X_u, Y_v, Z_w, K_p, M_q, N_r])
    return D


def forward_dynamics(
    M_rb,
    M_a,
    D,
    torques,
    linear_vel,
    angular_vel,
    timestamps,
):
    n_timesteps = len(timestamps)
    dt = 0.05
    v_pred = np.zeros((n_timesteps, 6))
    v_pred[0] = np.concatenate([linear_vel[0], angular_vel[0]])
    M = M_rb + M_a
    M_inv = np.linalg.inv(M)

    for t in range(n_timesteps - 1):

        v_current = v_pred[t]
        C_rb = calculate_coriolis_matrix(M_rb, v_current)
        C_a = calculate_coriolis_matrix(M_a, v_current)
        C = C_a + C_rb
        Cv = C @ v_current
        Dv = D @ v_current
        v_dot = M_inv @ (torques[t, :] - Cv + Dv)

        v_dot_real = np.array([ 0.15789908, -0.00454942, 0.01100861,0.01531216, 0.0167554 , -0.50773806])
        tau_real = M @ v_dot_real - D @ v_current + C @ v_current
        print(tau_real)

        v_pred[t + 1] = v_current + v_dot * dt

    return v_pred


if __name__ == "__main__":
    added_mass = np.diag([1.7182, 0, 5.468, 0, 1.2481, 0.4006])
    vel = np.array([0.447, 0.050, 0.3049, -0.0029, 0.0014, 0.414])
    print(calculate_coriolis_matrix(added_mass, vel))
