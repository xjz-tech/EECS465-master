import numpy as np
from pybullet_tools.utils import connect, disconnect, set_joint_positions, wait_if_gui, set_point, load_model,\
                                 joint_from_name, link_from_name, get_joint_info, HideOutput, get_com_pose, wait_for_duration
from pybullet_tools.transformations import quaternion_matrix
from pybullet_tools.pr2_utils import DRAKE_PR2_URDF
import time
import sys
### YOUR IMPORTS HERE ###

#########################

from utils import draw_sphere_marker

def get_ee_transform(robot, joint_indices, joint_vals=None):
    # returns end-effector transform in the world frame with input joint configuration or with current configuration if not specified
    if joint_vals is not None:
        set_joint_positions(robot, joint_indices, joint_vals)
    ee_link = 'l_gripper_tool_frame'
    pos, orn = get_com_pose(robot, link_from_name(robot, ee_link))
    res = quaternion_matrix(orn)
    res[:3, 3] = pos
    return res

def get_joint_axis(robot, joint_idx):
    # returns joint axis in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn) # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn) # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    R_W_J = H_W_J[:3, :3]
    joint_axis_local = np.array(j_info.jointAxis)
    joint_axis_world = np.dot(R_W_J, joint_axis_local)
    return joint_axis_world

def get_joint_position(robot, joint_idx):
    # returns joint position in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn) # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn) # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    j_world_posi = H_W_J[:3, 3]
    return j_world_posi

def set_joint_positions_np(robot, joints, q_arr):
    # set active DOF values from a numpy array
    q = [q_arr[0, i] for i in range(q_arr.shape[1])]
    set_joint_positions(robot, joints, q)

def get_translation_jacobian(robot, joint_indices):
    J = np.zeros((3, len(joint_indices)))
    ### YOUR CODE HERE ###

    ee_pos = get_ee_transform(robot, joint_indices)[:3, 3]  # Get the end-effector position in the world frame
    for i, joint_index in enumerate(joint_indices):
        joint_axis = get_joint_axis(robot, joint_index)  # Get the rotation axis direction of the joint (z_i)
        joint_pos = get_joint_position(robot, joint_index)  # Get the position of the joint (p_i)
        # print(joint_pos)
        # # Translational part of the Jacobian (linear velocity)
        # here we only care about position
        J[:3, i] = np.cross(joint_axis, (ee_pos - joint_pos))  # Compute z_i × (p_ee - p_i)

        # # Rotational part of the Jacobian (angular velocity)
        # J[3:, i] = joint_axis  # For revolute joints, the rotation axis contributes directly

    ### YOUR CODE HERE ###
    return J

def get_jacobian_pinv(J):
    J_pinv = []
    ### YOUR CODE HERE ###
    lambd = 0.01
    JJ_T = np.dot(J, J.T)
    # Add the damping factor lambda^2 to the diagonal
    damped_term = lambd ** 2 * np.eye(JJ_T.shape[0])
    # Invert the damped term
    inv_damped = np.linalg.inv(JJ_T + damped_term)
    # Compute the pseudo-inverse using the formula
    J_pinv = np.dot(J.T, inv_damped)

    ### YOUR CODE HERE ###
    return J_pinv

def tuck_arm(robot):
    joint_names = ['torso_lift_joint','l_shoulder_lift_joint','l_elbow_flex_joint',\
        'l_wrist_flex_joint','r_shoulder_lift_joint','r_elbow_flex_joint','r_wrist_flex_joint']
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    set_joint_positions(robot, joint_idx, (0.24,1.29023451,-2.32099996,-0.69800004,1.27843491,-2.32100002,-0.69799996))

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("Specify which target to run:")
        print("  'python3 ik_template.py [target index]' will run the simulation for a specific target index (0-4)")
        exit()
    test_idx = 0
    try:
        test_idx = int(args[0])
    except:
        print("ERROR: Test index has not been specified")
        exit()

    # initialize PyBullet
    connect(use_gui=True, shadows=False)
    # load robot
    with HideOutput():
        robot = load_model(DRAKE_PR2_URDF, fixed_base=True)
        set_point(robot, (-0.75, -0.07551, 0.02))
    tuck_arm(robot)
    # define active DoFs
    joint_names =['l_shoulder_pan_joint','l_shoulder_lift_joint','l_upper_arm_roll_joint', \
        'l_elbow_flex_joint','l_forearm_roll_joint','l_wrist_flex_joint','l_wrist_roll_joint']
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    # intial config
    q_arr = np.zeros((1, len(joint_idx)))
    set_joint_positions_np(robot, joint_idx, q_arr)
    # list of example targets
    targets = [[-0.15070158,  0.47726995, 1.56714123],
               [-0.36535318,  0.11249,    1.08326675],
               [-0.56491217,  0.011443,   1.2922572 ],
               [-1.07012697,  0.81909669, 0.47344636],
               [-1.11050811,  0.97000718,  1.31087581]]
    # define joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robot, joint_idx[i]).jointLowerLimit, get_joint_info(robot, joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}
    q = np.zeros((1, len(joint_names))) # start at this configuration
    target = targets[test_idx]
    # draw a blue sphere at the target
    draw_sphere_marker(target, 0.05, (0, 0, 1, 1))
    
    ### YOUR CODE HERE ###
    x_current = get_ee_transform(robot, joint_idx)[:3, 3]  # Current end-effector position
    error = np.linalg.norm(target - x_current)  # Compute initial error
    threshold = 0.01  # Convergence threshold
    alpha = 0.1  # Step size scaling factor

    configurations = []  # List to store configurations for each target

    while error > threshold:
        # Compute Jacobian and its pseudo-inverse
        J = get_translation_jacobian(robot, joint_idx)
        J_inv = get_jacobian_pinv(J)

        # Compute the error vector (desired velocity direction)
        x_dot = target - x_current

        # Update joint velocities using the pseudo-inverse
        q_dot = np.dot(J_inv, x_dot)

        # Limit joint velocity magnitudes to avoid large jumps
        if np.linalg.norm(q_dot) > alpha:
            q_dot = alpha * (q_dot / np.linalg.norm(q_dot))

        # Update joint configuration
        q += q_dot[np.newaxis, :]

        # Ensure joint values stay within their limits
        for i, joint_name in enumerate(joint_names):
            lower_limit, upper_limit = joint_limits[joint_name]
            q[0, i] = max(lower_limit, min(q[0, i], upper_limit))  # Clamp the value

        set_joint_positions_np(robot, joint_idx, q)

        # Recompute current position and error
        x_current = get_ee_transform(robot, joint_idx)[:3, 3]
        error = np.linalg.norm(target - x_current)


    # Save the configuration if the target is successfully reached
    configurations.append(q.copy())

    # Output the configurations for all targets
    print("Configurations for targets:", configurations)

    ### YOUR CODE HERE ###

    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()