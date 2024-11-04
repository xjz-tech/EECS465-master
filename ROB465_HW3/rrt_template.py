import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
### YOUR IMPORTS HERE ###
from utils import draw_line
class Node:
    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent
#########################


joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2table.json')

    # define active DoFs
    joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')
    joint_idx = [joint_from_name(robots['pr2'], jn) for jn in joint_names]

    # parse active DoF joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robots['pr2'], joint_idx[i]).jointLowerLimit, get_joint_info(robots['pr2'], joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}

    collision_fn = get_collision_fn_PR2(robots['pr2'], joint_idx, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, 1.19, -1.548, 1.557, -1.32, -0.1928)))

    start_config = tuple(get_joint_positions(robots['pr2'], joint_idx))
    goal_config = (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928)
    path = []
    ### YOUR CODE HERE ###
    print("\n=== Starting Path Planning ===")
    path = None
    max_attempts = 10
    current_attempt = 0

    while path is None and current_attempt < max_attempts:
        current_attempt += 1
        print(f"\nAttempt {current_attempt}/{max_attempts}...")

        # RRT Connect Planner
        tree_one = [Node(start_config)]
        tree_two = [Node(goal_config)]
        max_iterations = 10000
        goal_sample_rate = 0.1  # Probability of sampling goal
        step_size = 0.05

        for iteration in range(max_iterations):
            # Random sampling with a chance to directly sample the goal
            if random.random() < goal_sample_rate:
                random_config = goal_config
            else:
                random_config = tuple(
                    random.uniform(joint_limits[joint][0], joint_limits[joint][1]) for joint in joint_names)

            # Expand tree_one towards the random_config
            nearest_node = min(tree_one, key=lambda node: np.linalg.norm(np.subtract(node.config, random_config)))
            direction = np.subtract(random_config, nearest_node.config)
            norm_direction = direction / np.linalg.norm(direction)
            new_config = tuple(nearest_node.config + norm_direction * step_size)

            # Check for collisions before adding the node
            if not collision_fn(new_config):
                new_node = Node(new_config, nearest_node)
                tree_one.append(new_node)

                # Check if tree_one reaches close enough to random_config
                if np.linalg.norm(np.subtract(new_config, random_config)) < step_size:
                    break

            # Expand tree_two towards the last node of tree_one
            nearest_node = min(tree_two, key=lambda node: np.linalg.norm(np.subtract(node.config, tree_one[-1].config)))
            direction = np.subtract(tree_one[-1].config, nearest_node.config)
            norm_direction = direction / np.linalg.norm(direction)
            new_config = tuple(nearest_node.config + norm_direction * step_size)

            # Check for collisions before adding the node
            if not collision_fn(new_config):
                new_node = Node(new_config, nearest_node)
                tree_two.append(new_node)

                # If trees connect, build the path
                if np.linalg.norm(np.subtract(new_config, tree_one[-1].config)) < step_size:
                    path = []
                    node = tree_one[-1]
                    while node:
                        path.append(node.config)
                        node = node.parent
                    path = path[::-1]  # Reverse path from start to goal

                    node = tree_two[-1]
                    while node:
                        path.append(node.config)
                        node = node.parent
                    break

        if path is None:
            print(f"Attempt {current_attempt} failed to find a path")

    if path:
        print(f"\n✓ Path found successfully on attempt {current_attempt}!")
        print(f"Initial path length: {len(path)} configurations")

        # Path smoothing
        print("\nOptimizing path...")
        smoothed = path.copy()
        for _ in range(30):  # Perform path smoothing iterations
            i, j = sorted(random.sample(range(len(smoothed)), 2))
            if j <= i + 1:
                continue
            points = np.linspace(smoothed[i], smoothed[j], num=5)
            if all(not collision_fn(tuple(point)) for point in points):
                smoothed[i:j + 1] = [tuple(point) for point in points]

        # Remove redundant points
        result = [smoothed[0]]
        for k in range(1, len(smoothed) - 1):
            vector1 = np.array(smoothed[k]) - np.array(result[-1])
            vector2 = np.array(smoothed[k + 1]) - np.array(smoothed[k])
            if not np.allclose(vector1, vector2):  # Keep points where direction changes
                result.append(smoothed[k])
        result.append(smoothed[-1])
        smoothed_path = result

        improvement = ((len(path) - len(smoothed_path)) / len(path)) * 100
        print(f"Path optimized: {len(path)} → {len(smoothed_path)} configurations")
        print(f"Improvement: {improvement:.1f}% reduction in path length")

        # Ensure path starts from the start configuration
        if smoothed_path[0] != start_config:
            smoothed_path = smoothed_path[::-1]
            print("Path orientation corrected")

        # Visualization
        print("\nVisualizing paths:")
        print("- Red: Original path")
        print("- Blue: Smoothed path")

        # Draw original path (red)
        set_joint_positions(robots['pr2'], joint_idx, start_config)
        for i in range(len(path) - 1):
            set_joint_positions(robots['pr2'], joint_idx, path[i])
            start_pose = get_link_pose(robots['pr2'], link_from_name(robots['pr2'], 'l_gripper_tool_frame'))[0]
            set_joint_positions(robots['pr2'], joint_idx, path[i + 1])
            end_pose = get_link_pose(robots['pr2'], link_from_name(robots['pr2'], 'l_gripper_tool_frame'))[0]
            draw_line(start_pose, end_pose, width=2, color=(1, 0, 0))

        # Draw smoothed path (blue)
        set_joint_positions(robots['pr2'], joint_idx, start_config)
        for i in range(len(smoothed_path) - 1):
            set_joint_positions(robots['pr2'], joint_idx, smoothed_path[i])
            start_pose = get_link_pose(robots['pr2'], link_from_name(robots['pr2'], 'l_gripper_tool_frame'))[0]
            set_joint_positions(robots['pr2'], joint_idx, smoothed_path[i + 1])
            end_pose = get_link_pose(robots['pr2'], link_from_name(robots['pr2'], 'l_gripper_tool_frame'))[0]
            draw_line(start_pose, end_pose, width=2, color=(0, 0, 1))

        # Use smoothed path for execution
        path = smoothed_path
        print("\nReady for path execution!")

    else:
        print("\n✗ Failed to find a valid path after all attempts")
        print("Possible issues:")
        print("- Goal configuration might be in collision")
        print("- Path might be blocked by obstacles")
        print("- More planning attempts might be needed")

    ######################
    # Execute planned path
    execute_trajectory(robots['pr2'], joint_idx, path, sleep=0.1)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()