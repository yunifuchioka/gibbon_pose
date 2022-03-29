from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


class Keypoints(Enum):
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


# helper function to convert from keypoint array to dictionary
def keypoint_array2dict(keypoint_array):
    keypoint_array_reshape = keypoint_array.reshape((17, 2))
    keypoint_dict = {}
    for keypoint in Keypoints:
        keypoint_dict[keypoint.name] = keypoint_array_reshape[keypoint.value, :]
    return keypoint_dict


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
    ax.plot(face_coords[:, 0], face_coords[:, 1], "-o")
    ax.plot(spine_coords[:, 0], spine_coords[:, 1], "-o")
    ax.plot(left_arm_coords[:, 0], left_arm_coords[:, 1], "-o")
    ax.plot(right_arm_coords[:, 0], right_arm_coords[:, 1], "-o")
    ax.plot(left_leg_coords[:, 0], left_leg_coords[:, 1], "-o")
    ax.plot(right_leg_coords[:, 0], right_leg_coords[:, 1], "-o")


def plot_img(img, pose=None, ax=plt):
    ax.imshow(img)
    if pose is not None:
        plot_pose_on_img(pose, ax)


def plot_img_pred(img, predictions, ax=plt, thresh=0.2):
    ax.imshow(img)

    coord_above_thresh = predictions[predictions[:, 2] > thresh, :2]

    ax.plot(
        coord_above_thresh[:, 0], coord_above_thresh[:, 1], "o", ms=10, mec="r", mfc="r"
    )
