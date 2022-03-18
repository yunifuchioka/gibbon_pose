from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


class Landmarks(Enum):
    right_eye = 0
    left_eye = 1
    nose = 2
    head = 3
    neck = 4
    left_shoulder = 5
    left_elbow = 6
    left_wrist = 7
    right_shoulder = 8
    right_elbow = 9
    right_wrist = 10
    hip = 11
    left_knee = 12
    left_ankle = 13
    right_knee = 14
    right_ankle = 15
    tail = 16


# helper function to plot poses
def plot_pose_on_img(pose, ax=plt):
    face_coords = np.vstack((pose["left_eye"], pose["nose"], pose["right_eye"]))
    spine_coords = np.vstack((pose["head"], pose["neck"], pose["hip"], pose["tail"]))
    left_arm_coords = np.vstack(
        (
            pose["neck"],
            pose["left_shoulder"],
            pose["left_elbow"],
            pose["left_wrist"],
        )
    )
    right_arm_coords = np.vstack(
        (
            pose["neck"],
            pose["right_shoulder"],
            pose["right_elbow"],
            pose["right_wrist"],
        )
    )
    left_leg_coords = np.vstack(
        (
            pose["hip"],
            pose["left_knee"],
            pose["left_ankle"],
        )
    )
    right_leg_coords = np.vstack(
        (
            pose["hip"],
            pose["right_knee"],
            pose["right_ankle"],
        )
    )
    plt.plot(face_coords[:, 0], face_coords[:, 1], "-o")
    plt.plot(spine_coords[:, 0], spine_coords[:, 1], "-o")
    plt.plot(left_arm_coords[:, 0], left_arm_coords[:, 1], "-o")
    plt.plot(right_arm_coords[:, 0], right_arm_coords[:, 1], "-o")
    plt.plot(left_leg_coords[:, 0], left_leg_coords[:, 1], "-o")
    plt.plot(right_leg_coords[:, 0], right_leg_coords[:, 1], "-o")


def plot_img(img, pose=None, ax=plt):
    plt.imshow(img)
    if pose is not None:
        plot_pose_on_img(pose)
