import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
from queue import PriorityQueue

class Node:
    def __init__(self, x_in, y_in, theta_in, id_in, parentid_in):
        self.x = x_in
        self.y = y_in
        self.theta = theta_in
        self.id = id_in
        self.parentid = parentid_in

    def printme(self):
        print("\tNode id", self.id, ":", "x =", self.x, "y =", self.y, "theta =", self.theta, "parentid:", self.parentid)


def heuristic(current, goal):
    return np.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2) + \
        min(abs(current[2] - goal[2]), 2 * np.pi - abs(current[2] - goal[2]))


def action_cost(n, m):
    return np.sqrt((n[0] - m[0]) ** 2 + (n[1] - m[1]) ** 2) + \
        min(abs(n[2] - m[2]), 2 * np.pi - abs(n[2] - m[2]))


def astar_search(start, goal, neighbors_fn, collision_fn):
    open_set = PriorityQueue()
    closed_set = set()
    unique_id = 0
    open_set.put((0, unique_id, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    collision_free_configs = []
    colliding_configs = []

    while not open_set.empty():
        _, _, current = open_set.get()

        if np.allclose(current, goal, atol=1e-2):
            path = []
            total_cost = g_score[current]
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, collision_free_configs, colliding_configs, total_cost

        closed_set.add(current)

        for neighbor in neighbors_fn(current):
            if neighbor in closed_set:
                continue

            if collision_fn(neighbor):
                colliding_configs.append(neighbor[:2])
                continue
            else:
                collision_free_configs.append(neighbor[:2])

            tentative_g_score = g_score[current] + action_cost(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                unique_id += 1
                open_set.put((f_score[neighbor], unique_id, neighbor))

    return None, collision_free_configs, colliding_configs, float('inf')


def normalize_angle(theta):
    theta = np.fmod(theta + np.pi, 2 * np.pi)
    if theta < 0:
        theta += 2 * np.pi
    return theta - np.pi


def get_neighbors_4(current):
    x, y, theta = current
    delta = 0.1
    theta_values = [0, np.pi / 2, -np.pi / 2, np.pi]

    neighbors = []

    neighbors.append((x + delta, y, normalize_angle(theta)))
    neighbors.append((x - delta, y, normalize_angle(theta)))
    neighbors.append((x, y + delta, normalize_angle(theta)))
    neighbors.append((x, y - delta, normalize_angle(theta)))

    for new_theta in theta_values:
        if new_theta != theta:
            neighbors.append((x, y, normalize_angle(new_theta)))

    return neighbors


def get_neighbors_8(current):
    x, y, theta = current
    delta = 0.1
    delta_diagonal = round(delta * np.sqrt(2), 1)
    theta_values = [0, np.pi / 2, -np.pi / 2, np.pi]

    neighbors = [
        (round(x + delta, 1), round(y, 1), normalize_angle(theta)),
        (round(x - delta, 1), round(y, 1), normalize_angle(theta)),
        (round(x, 1), round(y + delta, 1), normalize_angle(theta)),
        (round(x, 1), round(y - delta, 1), normalize_angle(theta)),
        (round(x + delta_diagonal, 1), round(y + delta_diagonal, 1), normalize_angle(theta)),
        (round(x - delta_diagonal, 1), round(y - delta_diagonal, 1), normalize_angle(theta)),
        (round(x + delta_diagonal, 1), round(y - delta_diagonal, 1), normalize_angle(theta)),
        (round(x - delta_diagonal, 1), round(y + delta_diagonal, 1), normalize_angle(theta))
    ]

    for new_theta in theta_values:
        if new_theta != theta:
            neighbors.append((round(x, 1), round(y, 1), normalize_angle(new_theta)))

    return neighbors


def draw_path(path, color=(0, 0, 0, 1)):
    for i in range(len(path) - 1):
        draw_line(path[i][:2] + (0.05,), path[i + 1][:2] + (0.05,), width=2, color=color[:3])


def draw_points(points, color, radius=0.05):
    for point in points:
        draw_sphere_marker(point + (0.05,), radius, color)
#########################

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2)))

    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))

    # Example of draw 
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi/2)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###
    # start_time = time.time()
    # print("Using A* with 4 neighbors...")
    # path_4, collision_free_4, colliding_4, cost_4 = astar_search(start_config, goal_config, get_neighbors_4,
    #                                                              collision_fn)
    # print("Planner run time with 4 neighbors: ", time.time() - start_time)
    # print("Path cost with 4 neighbors: ", cost_4)
    #
    # # If a path was found using the 4-connected neighbors, draw and execute it
    # if path_4 is not None:
    #     # draw_path(path_4, color=(0, 0, 0, 1))  # Draw the path with black lines
    #     # draw_points(collision_free_4, color=(0, 0, 1, 1), radius=0.05)  # Draw collision-free points in blue
    #     # draw_points(colliding_4, color=(1, 0, 0, 1), radius=0.05)  # Draw colliding points in red
    #     execute_trajectory(robots['pr2'], base_joints, path_4,
    #                        sleep=0.2)  # Execute the robot's trajectory along the path
    # else:
    #     print("No path found with 4 neighbors.")  # If no path was found, print a message

    # Perform A* path planning using 8-connected neighbors
    start_time = time.time()  # Start the timer to record planning time
    print("Using A* with 8 neighbors...")
    path_8, collision_free_8, colliding_8, cost_8 = astar_search(start_config, goal_config, get_neighbors_8,
                                                                 collision_fn)
    print("Planner run time with 8 neighbors: ", time.time() - start_time)  # Print the time it took to find the path
    print("Path cost with 8 neighbors: ", cost_8)  # Print the cost of the path

    # If a path was found using the 8-connected neighbors, draw and execute it
    if path_8 is not None:
        draw_path(path_8, color=(0, 0, 0, 1))  # Draw the path with black lines
        # draw_points(collision_free_8, color=(0, 0, 1, 1), radius=0.05)  # Draw collision-free points in blue
        # draw_points(colliding_8, color=(1, 0, 0, 1), radius=0.05)  # Draw colliding points in red
        # execute_trajectory(robots['pr2'], base_joints, path_8,
        #                    sleep=0.2)  # Execute the robot's trajectory along the path
    else:
        print("No path found with 8 neighbors.")  # If no path was found, print a message

    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()