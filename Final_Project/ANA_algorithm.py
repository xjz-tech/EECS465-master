import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, \
    joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
from queue import PriorityQueue


def heuristic(current, goal):
    return np.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2) + \
        min(abs(current[2] - goal[2]), 2 * np.pi - abs(current[2] - goal[2]))


def action_cost(n, m):
    return np.sqrt((n[0] - m[0]) ** 2 + (n[1] - m[1]) ** 2) + \
        min(abs(n[2] - m[2]), 2 * np.pi - abs(n[2] - m[2]))


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

def ana_star_search(start, goal, neighbors_fn, collision_fn):
    G = float('inf')  # 当前最佳解决方案的成本
    E = float('inf')
    unique_id = 0
    open_set = PriorityQueue()
    g_score = {}
    pred = {}
    g_score[start] = 0
    closed_set = set()

    h_start = heuristic(start, goal)
    if h_start == 0:
        h_start = 1e-6  # 避免除以零

    e_start = (G - g_score[start]) / h_start
    open_set.put((-e_start, unique_id, start))  # 最大化e(s)，使用负值

    collision_free_configs = []
    colliding_configs = []

    best_path = None

    while not open_set.empty():
        # IMPROVESOLUTION
        while not open_set.empty():
            _, _, s = open_set.get()

            if s in closed_set:
                continue  # 跳过已扩展的节点

            if g_score[s] + heuristic(s, goal) >= G:    # not the calculation method of G && here also need to add s to closed set?
                continue  # 跳过无法导致更好解决方案的节点

            closed_set.add(s)  # 标记节点已扩展

            e_s = (G - g_score[s]) / heuristic(s, goal)
            if e_s < E:
                E = e_s


            if np.allclose(s, goal, atol=1e-2):
                G = g_score[s]
                # 重建路径
                path = []
                current = s
                while current in pred:
                    path.append(current)
                    current = pred[current]
                path.append(start)
                path.reverse()
                best_path = path
                return best_path, collision_free_configs, colliding_configs, G  # 立即返回

            for neighbor in neighbors_fn(s):
                if collision_fn(neighbor):
                    colliding_configs.append(neighbor[:2])
                    continue
                else:
                    collision_free_configs.append(neighbor[:2])

                tentative_g_score = g_score[s] + action_cost(s, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    if neighbor in closed_set:
                        continue  # 跳过已扩展的邻居

                    pred[neighbor] = s
                    g_score[neighbor] = tentative_g_score

                    if g_score[neighbor] + heuristic(neighbor, goal) < G:
                        h_neighbor = heuristic(neighbor, goal)
                        if h_neighbor == 0:
                            h_neighbor = 1e-6  # 避免除以零
                        e_neighbor = (G - g_score[neighbor]) / h_neighbor
                        unique_id += 1
                        open_set.put((-e_neighbor, unique_id, neighbor))  # 最大化e(s)

        # 在退出内部循环后

        # 报告当前的E-次优解
        if best_path is not None:
            print("找到成本为：", G, "的解决方案")
            # 您可以在此处可视化或处理best_path

        # 更新OPEN中的键e(s)，并剪枝满足g(s) + h(s) ≥ G的节点
        new_open_set = PriorityQueue()
        unique_id = 0  # 重置unique_id

        while not open_set.empty():
            _, _, s = open_set.get()
            if g_score[s] + heuristic(s, goal) < G:
                h_s = heuristic(s, goal)
                if h_s == 0:
                    h_s = 1e-6  # 避免除以零
                e_s = (G - g_score[s]) / h_s
                unique_id += 1
                new_open_set.put((-e_s, unique_id, s))

        open_set = new_open_set

        if open_set.empty():
            # 没有更多节点可扩展
            break

    # 返回找到的最佳路径
    if best_path is not None:
        total_cost = G
        return best_path, collision_free_configs, colliding_configs, total_cost
    else:
        return None, collision_free_configs, colliding_configs, float('inf')



def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi / 2)

    # Perform ANA* path planning using 8-connected neighbors
    start_time = time.time()  # Start the timer to record planning time
    print("Using ANA* with 8 neighbors...")
    path_ana, collision_free_ana, colliding_ana, cost_ana = ana_star_search(start_config, goal_config, get_neighbors_8,
                                                                            collision_fn)
    print("Planner run time with ANA*: ", time.time() - start_time)  # Print the time it took to find the path
    print("Path cost with ANA*: ", cost_ana)  # Print the cost of the path

    # If a path was found using ANA*, draw and execute it
    if path_ana is not None:
        draw_path(path_ana, color=(0, 0, 0, 1))  # Draw the path with black lines
        execute_trajectory(robots['pr2'], base_joints, path_ana,
                           sleep=0.2)  # Execute the robot's trajectory along the path
    else:
        print("No path found with ANA*.")  # If no path was found, print a message

    ######################
    print("Total planner run time: ", time.time() - start_time)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()


if __name__ == '__main__':
    main()
