import os
import sys

sys.path.append(os.getcwd())

import numpy as np


class UUV(object):
    def __init__(
        self,
        name: str,
        mass: float,
        r_p: np.ndarray,
        r_f: np.ndarray,
        # r_t: np.ndarray,
    ):
        self.name = name
        self.gravity = 9.81
        self.mass = mass
        self.r_p = r_p
        self.r_f = r_f
        # self.r_t = r_t


class DataLoader(object):
    def __init__(
        self,
        linear_vel: np.ndarray,
        angular_vel: np.ndarray,
        linear_acc: np.ndarray,
        angular_acc: np.ndarray,
        thruster: np.ndarray,
        timestamps: np.ndarray,
        uuv: UUV,
        frame: str = "flu",
    ):
        self.timestamps = timestamps
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel
        self.linear_acc = linear_acc
        self.angular_acc = angular_acc
        self.thruster = thruster
        self.frame = frame
        self.uuv = uuv

        thruster_0 = np.stack(
            [self.thruster[:, 0] * 0.707, self.thruster[:, 0] * 0.707], axis=-1
        )
        thruster_1 = np.stack(
            [self.thruster[:, 1] * 0.707, -self.thruster[:, 1] * 0.707], axis=-1
        )
        thruster_2 = np.stack(
            [-self.thruster[:, 2] * 0.707, self.thruster[:, 2] * 0.707], axis=-1
        )
        thruster_3 = np.stack(
            [-self.thruster[:, 2] * 0.707, -self.thruster[:, 2] * 0.707], axis=-1
        )

        self.F_p = np.dot(self.thruster, self.uuv.r_f)
        self.moment_z = (
            np.cross(self.uuv.r_p[0, :], thruster_0, 0, 1)
            - np.cross(self.uuv.r_p[1, :], thruster_1, 0, 1)
            - np.cross(self.uuv.r_p[2, :], thruster_2, 0, 1)
            + np.cross(self.uuv.r_p[3, :], thruster_3, 0, 1)
        )
