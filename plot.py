# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy.spatial.transform import Rotation as R
# pose_df = pd.read_csv("pose_data.csv").to_numpy()
# thruster_data = pd.read_csv("thruster_data.csv").to_numpy()

# r_f = np.array(
#     [
#         [0.707, 0.707],
#         [0.707, -0.707],
#         [-0.707, 0.707],
#         [-0.707, -0.707],
#     ]
# )

# r_p = np.array(
#     [
#         [0.1355, -0.1],
#         [0.1355, 0.1],
#         [-0.1475, -0.1],
#         [-0.1475, 0.1],
#     ]
# )

# model_name = "bluerov"
# mass = 11.2
# rho = 10**3
# L = 0.448


# linear_vel = pose_df[:, 8:11]
# angular_vel = pose_df[:, 11:14]
# timestamps = pose_df[:, 0]
# thruster = thruster_data[:-1, 1:5]

# q = pose_df[:, 4:8]
# r = R.from_quat(q).as_matrix()
# linear_vel = np.matmul(np.linalg.inv(r), linear_vel[..., np.newaxis]).squeeze(-1)
# angular_vel = np.matmul(np.linalg.inv(r), angular_vel[..., np.newaxis]).squeeze(-1)

# plt.figure()
# plt.subplot(6,1,1)
# plt.plot(linear_vel[:,0])
# plt.subplot(6,1,2)
# plt.plot(linear_vel[:,1])
# plt.subplot(6,1,3)
# plt.plot(linear_vel[:,2])
# plt.subplot(6,1,4)
# plt.plot(angular_vel[:,0])
# plt.subplot(6,1,5)
# plt.plot(angular_vel[:,1])
# plt.subplot(6,1,6)
# plt.plot(angular_vel[:,2])
# plt.show()

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 
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

    # J_1 = [
    #     [
    #         np.cos(psai) * np.cos(theta),
    #         -np.sin(psai) * np.cos(phi) + np.cos(psai) * np.sin(theta) * np.sin(phi),
    #         np.sin(psai) * np.sin(phi) + np.cos(psai) * np.cos(phi) * np.sin(theta),
    #     ],
    #     [
    #         np.sin(psai) * np.cos(theta),
    #         np.cos(psai) * np.cos(phi) + np.sin(phi) * np.sin(theta) * np.sin(psai),
    #         -np.cos(psai) * np.sin(phi) + np.sin(theta) * np.sin(psai) * np.cos(phi),
    #     ],
    #     [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)],
    # ]
    # linear_matrix.append(J_1)

angular_rotation = np.stack(angular_matrix, axis=0)
# linear_rotation = np.stack(linear_matrix, axis=0)

linear_vel = np.matmul(
    np.linalg.inv(r), linear_vel[..., np.newaxis]
).squeeze(-1)
angular_vel_trans = np.matmul(
    np.linalg.inv(angular_rotation), angular_vel[..., np.newaxis]
).squeeze(-1)

# plt.figure()
# plt.subplot(2,1,1)
# plt.ylabel("Before transformation")
# plt.plot(angular_vel[:,2])
# plt.subplot(2,1,2)
# plt.ylabel("After transformation")
# plt.plot(angular_vel_trans[:,2])
# plt.show()

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

moment_z1 = (
    np.cross(r_p[0, :], thruster_0)
    + np.cross(r_p[1, :], thruster_1)
    + np.cross(r_p[2, :], thruster_2)
    + np.cross(r_p[3, :], thruster_3)
)

moment_coff = np.array([0.166523,-0.166523,-0.175,0.175])
moment_z2 = thruster[:,0]*moment_coff[0] +thruster[:,1]*moment_coff[1] +thruster[:,2]*moment_coff[2]+thruster[:,3]*moment_coff[3]
plt.figure()
plt.subplot(2,1,1)
plt.plot(moment_z1)
plt.subplot(2,1,2)
plt.plot(moment_z2)
plt.show()