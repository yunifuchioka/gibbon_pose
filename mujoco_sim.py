import mujoco_py
from mujoco_py.generated import const
import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import glfw

training_data_csv = '/home/niloofar/Documents/coursework/cpsc533r/project/mujoco_sim/training/data.csv'
labels_csv = '/home/niloofar/Documents/coursework/cpsc533r/project/mujoco_sim/training/labels.csv'

image_output_path = '/home/niloofar/Documents/coursework/cpsc533r/project/mujoco_sim/output_png/'
xml_output_path = '/home/niloofar/Documents/coursework/cpsc533r/project/mujoco_sim/output_xml/'      
model_path = '/home/niloofar/Documents/coursework/cpsc533r/project/mujoco_sim/models/gibbon3d.xml'

model = mujoco_py.load_model_from_path(model_path)

# 15 joints from mujoco xml model
mujoco_joint_names = [
                'right_hip_x', 'right_hip_z', 'right_hip_y', 
                'right_knee',
                'left_hip_x', 'left_hip_z', 'left_hip_y', 
                'left_knee', 
                'right_shoulder_x', 'right_shoulder_y', 
                'right_elbow_z', 'right_elbow_y',
                'left_shoulder_x', 'left_shoulder_y', 
                'left_elbow_z', 'left_elbow_y']

# 12 joints to match with DLC pose estimation results
target_joint_names = ['head', 
                    'right_shoulder', 'right_elbow', 'right_wrist',
                    'left_shoulder', 'left_elbow', 'left_wrist',
                    'hip',
                    'right_knee', 'right_ankle',
                    'left_knee', 'left_ankle']

def computePositionData3D_naive(data):
    positions = []
    positions.append(data.get_body_xpos("head"))
    positions.append(data.get_body_xpos("right_upper_arm"))
    positions.append(data.get_body_xpos("right_lower_arm"))
    positions.append(data.get_body_xpos("right_hand"))
    positions.append(data.get_body_xpos("left_upper_arm"))
    positions.append(data.get_body_xpos("left_lower_arm"))
    positions.append(data.get_body_xpos("left_hand"))
    positions.append(data.get_body_xpos("pelvis"))
    positions.append(data.get_body_xpos("right_shin"))
    positions.append(data.get_body_xpos("right_foot"))
    positions.append(data.get_body_xpos("left_shin"))
    positions.append(data.get_body_xpos("left_foot"))
    return positions

def computePositionData3D(data):
    positions_3d = [] # np.zeros((12, 3))

    joint_qpos = []
    joint_axis = []

    #print("\n3D compute:")
    for j in range(0, len(mujoco_joint_names)):
        pos = data.get_joint_qpos(mujoco_joint_names[j])
        axis = data.get_joint_xaxis(mujoco_joint_names[j])
        joint_qpos.append(pos)
        joint_axis.append(axis)
        #print(mujoco_joint_names[j], ": ", pos, axis)
        

    #head
    head_pos = data.get_body_xpos("gibbon3d") + data.get_body_xpos("head")
    positions_3d.append(head_pos) # head

    # right upper arm
    right_upper_arm_pos = data.get_body_xpos("gibbon3d") + data.get_body_xpos("right_upper_arm")
    positions_3d.append(right_upper_arm_pos) # right_shoulder

    # right lower arm
    angle = data.get_joint_qpos("right_shoulder_x")
    axis = data.get_joint_xaxis("right_shoulder_x")
    shoulder_rot_x = get_rotation_matrix(axis, angle)

    angle = data.get_joint_qpos("right_shoulder_y")
    axis = data.get_joint_xaxis("right_shoulder_y")
    shoulder_rot_y = get_rotation_matrix(axis, angle)

    right_elbow_rel_pos = data.get_body_xpos("right_lower_arm")
    right_elbow_pos = right_upper_arm_pos + shoulder_rot_y.dot(shoulder_rot_x.dot(right_elbow_rel_pos - right_upper_arm_pos))

    positions_3d.append(right_elbow_pos) # right_elbow

    # right hand
    angle = data.get_joint_qpos("right_elbow_z")
    axis = data.get_joint_xaxis("right_elbow_z")
    elbow_rot_z = get_rotation_matrix(axis, angle)

    angle = data.get_joint_qpos("right_elbow_y")
    axis = data.get_joint_xaxis("right_elbow_y")
    elbow_rot_y = get_rotation_matrix(axis, angle)

    right_wrist_rel_pos = data.get_body_xpos("right_hand")
    right_wrist_pos = right_elbow_pos + elbow_rot_z.dot(elbow_rot_y.dot(right_wrist_rel_pos - right_elbow_pos))

    positions_3d.append(right_wrist_pos) # right_wrist

    # left upper arm
    left_upper_arm_pos = data.get_body_xpos("gibbon3d") + data.get_body_xpos("left_upper_arm")
    positions_3d.append(left_upper_arm_pos) # left_shoulder

    # left lower arm
    angle = data.get_joint_qpos("left_shoulder_x")
    axis = data.get_joint_xaxis("left_shoulder_x")
    shoulder_rot_x = get_rotation_matrix(axis, angle)

    angle = data.get_joint_qpos("left_shoulder_y")
    axis = data.get_joint_xaxis("left_shoulder_y")
    shoulder_rot_y = get_rotation_matrix(axis, angle)

    left_elbow_rel_pos = data.get_body_xpos("left_lower_arm")
    left_elbow_pos = left_upper_arm_pos + shoulder_rot_y.dot(shoulder_rot_x.dot(left_elbow_rel_pos - left_upper_arm_pos))

    positions_3d.append(left_elbow_pos) # left_elbow

    # left hand
    angle = data.get_joint_qpos("left_elbow_z")
    axis = data.get_joint_xaxis("left_elbow_z")
    elbow_rot_z = get_rotation_matrix(axis, angle)

    angle = data.get_joint_qpos("left_elbow_y")
    axis = data.get_joint_xaxis("left_elbow_y")
    elbow_rot_y = get_rotation_matrix(axis, angle)

    left_wrist_rel_pos = data.get_body_xpos("left_hand")
    left_wrist_pos = left_elbow_pos + elbow_rot_z.dot(elbow_rot_y.dot(left_wrist_rel_pos - left_elbow_pos))

    positions_3d.append(left_wrist_pos) # left_wrist

    # lower body
    waist_pos = data.get_body_xpos("gibbon3d") + data.get_body_xpos("waist")
    pelvis_pos = waist_pos + data.get_body_xpos("pelvis")
    right_thigh_pos = pelvis_pos + data.get_body_xpos("right_thigh")
    left_thigh_pos = pelvis_pos + data.get_body_xpos("left_thigh")
    #hip_pos = 0.5 * (right_thigh_pos + left_thigh_pos)
    hip_pos = waist_pos

    positions_3d.append(hip_pos)  # hip

    #right knee
    angle = data.get_joint_qpos("right_hip_x")
    axis = data.get_joint_xaxis("right_hip_x")
    hip_rot_x = get_rotation_matrix(axis, angle)

    angle = data.get_joint_qpos("right_hip_y")
    axis = data.get_joint_xaxis("right_hip_y")
    hip_rot_y = get_rotation_matrix(axis, angle)

    angle = data.get_joint_qpos("right_hip_z")
    axis = data.get_joint_xaxis("right_hip_z")
    hip_rot_z = get_rotation_matrix(axis, angle)

    right_knee_rel_pos = data.get_body_xpos("right_shin")
    right_knee_pos = right_thigh_pos + hip_rot_x.dot(hip_rot_y.dot(hip_rot_z.dot(right_knee_rel_pos - right_thigh_pos)))
    right_shin_pos = right_knee_pos

    positions_3d.append(right_knee_pos) # right_knee

    # right ankle
    angle = data.get_joint_qpos("right_knee")
    axis = data.get_joint_xaxis("right_knee")
    knee_rot = get_rotation_matrix(axis, angle)

    right_ankle_rel_pos = data.get_body_xpos("right_foot")
    right_ankle_pos = right_shin_pos + knee_rot.dot(right_ankle_rel_pos - right_shin_pos)

    positions_3d.append(right_ankle_pos) # right_ankle

    #left hip
    angle = data.get_joint_qpos("left_hip_x")
    axis = data.get_joint_xaxis("left_hip_x")
    hip_rot_x = get_rotation_matrix(axis, angle)

    angle = data.get_joint_qpos("left_hip_y")
    axis = data.get_joint_xaxis("left_hip_y")
    hip_rot_y = get_rotation_matrix(axis, angle)

    angle = data.get_joint_qpos("left_hip_z")
    axis = data.get_joint_xaxis("left_hip_z")
    hip_rot_z = get_rotation_matrix(axis, angle)

    left_knee_rel_pos = data.get_body_xpos("left_shin")
    left_knee_pos = left_thigh_pos + hip_rot_x.dot(hip_rot_y.dot(hip_rot_z.dot(left_knee_rel_pos - left_thigh_pos)))
    left_shin_pos = left_knee_pos

    positions_3d.append(left_knee_pos) # left_knee

    # left ankle
    angle = data.get_joint_qpos("left_knee")
    axis = data.get_joint_xaxis("left_knee")
    knee_rot = get_rotation_matrix(axis, angle)

    left_ankle_rel_pos = data.get_body_xpos("left_foot")
    left_ankle_pos = left_shin_pos + knee_rot.dot(left_ankle_rel_pos - left_shin_pos)

    positions_3d.append(left_ankle_pos) # left_ankle

    return positions_3d

# source: https://www.programcreek.com/python/?CodeExample=get+rotation+matrix
def get_rotation_matrix(axis, theta):
    """
    Find the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.
    Credit: http://stackoverflow.com/users/190597/unutbu

    Args:
        axis (list): rotation axis of the form [x, y, z]
        theta (float): rotational angle in radians

    Returns:
        array. Rotation matrix.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]]) 

def get_camera_pos(camera):
    init_axis = np.array([0., 1., 0.])

    azimuth_axis = np.array([0., 0., 1.])
    azimuth_angle = camera.azimuth
    azimuth_mat = get_rotation_matrix(azimuth_axis, azimuth_angle * np.pi / 180)

    new_axis = azimuth_mat.dot(init_axis)
    elev_axis = np.cross(new_axis, [0., 0., 1.])

    elev_angle = camera.elevation
    elev_mat = get_rotation_matrix(elev_axis, elev_angle * np.pi / 180)

    cam_init = np.array(camera.distance * init_axis)
    cam_pos = azimuth_mat.dot(cam_init)
    cam_pos = elev_mat.dot(cam_pos)
    
    return cam_pos

def get_camera_normal(camera):
    pos = get_camera_pos(camera)
    normal = camera.lookat - pos
    normal = normal / np.linalg.norm(normal)

    return normal

def project_joints_to_camera_plane(pos3d, cam_pos, cam_normal):
    projected_pos3d = []
    projected_pos2d = []

    for p in pos3d:
        v = p - cam_pos
        dist = v.dot(cam_normal)
        projected_p = p - dist * cam_normal
        projected_pos3d.append(projected_p)

    cam_y = [0., 0., 1.]
    cam_z = -cam_normal
    cam_x = np.cross(cam_y, cam_z)

    for pos in projected_pos3d:
        p = pos - cam_pos
        pos2d = [p.dot(cam_x), p.dot(cam_y)]
        projected_pos2d.append(pos2d)

    return projected_pos2d

# construct the 2d image with gibbon pose
def construct_image(projected_pos2d, write_path):
    img_size = 512
    img = np.zeros( (img_size,img_size,3), dtype=np.uint8)
    #img[img_size / 2, img_size / 2] = [0,255,0]

    # map from 3D screen of 5. by 5. to image of 256 by 256
    uniform_scale = img_size / 2

    projected_pos2d = np.array(projected_pos2d)
    scaled_points = np.floor(projected_pos2d * uniform_scale).astype(int)
    scaled_points[:,0] = img_size / 2 - scaled_points[:,0] 
    scaled_points[:,1] = img_size / 2 - scaled_points[:,1] 

    # head to hip
    x1, y1 = [scaled_points[0][0], scaled_points[7][0]], [scaled_points[0][1], scaled_points[7][1]]
    plt.plot(x1, y1, marker = 'o')

    # right to left shoulder
    x1, y1 = [scaled_points[1][0], scaled_points[4][0]], [scaled_points[1][1], scaled_points[4][1]]
    plt.plot(x1, y1, marker = 'o')

    # right shoulder to right elbow
    x1, y1 = [scaled_points[1][0], scaled_points[2][0]], [scaled_points[1][1], scaled_points[2][1]]
    plt.plot(x1, y1, marker = 'o')

    # right elbow to right wrist
    x1, y1 = [scaled_points[2][0], scaled_points[3][0]], [scaled_points[2][1], scaled_points[3][1]]
    plt.plot(x1, y1, marker = 'o')

    # right elbow to right wrist
    x1, y1 = [scaled_points[4][0], scaled_points[5][0]], [scaled_points[4][1], scaled_points[5][1]]
    plt.plot(x1, y1, marker = 'o')

    # left shoulder to left elbow
    x1, y1 = [scaled_points[5][0], scaled_points[6][0]], [scaled_points[5][1], scaled_points[6][1]]
    plt.plot(x1, y1, marker = 'o')

    # hip to right knee
    x1, y1 = [scaled_points[7][0], scaled_points[8][0]], [scaled_points[7][1], scaled_points[8][1]]
    plt.plot(x1, y1, marker = 'o')

    # hip to left knee
    x1, y1 = [scaled_points[7][0], scaled_points[10][0]], [scaled_points[7][1], scaled_points[10][1]]
    plt.plot(x1, y1, marker = 'o')
    
    # right knee to right ankle
    x1, y1 = [scaled_points[8][0], scaled_points[9][0]], [scaled_points[8][1], scaled_points[9][1]]
    plt.plot(x1, y1, marker = 'o')

    # left knee to left ankle
    x1, y1 = [scaled_points[10][0], scaled_points[11][0]], [scaled_points[10][1], scaled_points[11][1]]
    plt.plot(x1, y1, marker = 'o')

    # plot
    # plt_size = 400
    # plt.xlim([0, plt_size])
    # plt.ylim([0, plt_size])
    plt.imshow(img)
    #plt.show()
    plt.savefig(write_path)
    plt.clf()
    return img

def runSim():
    sim = mujoco_py.MjSim(model)
    data = sim.data
    state = sim.get_state()
    
    num_joints = len(state.qpos)
    viewer = mujoco_py.MjViewer(sim)
    t = 0
    count = 0

    cam_distance_range = [model.stat.extent * 2, model.stat.extent * 5]
    cam_elevation_range = [-50, 50]
    cam_azimuth_range = [-180, 180]

    while True:
        t += 1
        sim.forward()


        #if t % 500 == 0: # slow down for visualization
        viewer.cam.trackbodyid = 1

        # set camera random properties
        viewer.cam.distance = random.uniform(cam_distance_range[0], cam_distance_range[1])
        viewer.cam.elevation = random.uniform(cam_elevation_range[0], cam_elevation_range[1])
        viewer.cam.azimuth = random.uniform(cam_azimuth_range[0], cam_azimuth_range[1])
        
        gibbon_pos = data.get_body_xpos("gibbon3d")
        viewer.cam.lookat[:] = gibbon_pos
        
        data = sim.data
        state = sim.get_state()

        for j in range(0,len(mujoco_joint_names)):
            joint_range = model.jnt_range[j]
            pos = data.get_joint_qpos(mujoco_joint_names[j])
            randpos = random.uniform(joint_range[0], joint_range[1])
            data.set_joint_qpos(mujoco_joint_names[j], randpos)
            #data.qpos[j] = randpos
        
        # #test cam settings
        # viewer.cam.azimuth = 90
        # viewer.cam.elevation = 0

        joint_positions = computePositionData3D_naive(data)
        cam_pos = get_camera_pos(viewer.cam)
        cam_norm = get_camera_normal(viewer.cam)
        pos_2d = project_joints_to_camera_plane(joint_positions, cam_pos, cam_norm)

        count += 1

        construct_image(pos_2d, image_output_path + f'{count:05d}.png')

        with open(xml_output_path + f'gibbon3d_{count+1:05d}.xml', 'w') as fd:
            sim.save(fd)
            print("Saved frame", count)

        viewer.render()

if __name__ == '__main__':
    runSim()
