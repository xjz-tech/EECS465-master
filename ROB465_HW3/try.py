from queue import PriorityQueue
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, joint_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time


class Node:
    def __init__(self, x_in, y_in, theta_in, id_in, parentid_in):
        self.x = x_in
        self.y = y_in
        self.theta = theta_in
        self.id = id_in
        self.parentid = parentid_in

    def printme(self):
        print("\tNode id", self.id, ":", "x =", self.x, "y =", self.y, "theta =", self.theta, "parentid:",
              self.parentid)


def heuristic(current, goal):
    """计算当前节点到目标节点的启发式估计值"""
    return np.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2) + \
        min(abs(current[2] - goal[2]), 2 * np.pi - abs(current[2] - goal[2]))


def action_cost(n, m):
    """计算从节点 n 到节点 m 的动作代价"""
    return np.sqrt((n[0] - m[0]) ** 2 + (n[1] - m[1]) ** 2) + \
        min(abs(n[2] - m[2]), 2 * np.pi - abs(n[2] - m[2]))


def astar_search(start, goal, neighbors_fn, collision_fn):
    """
    A* 核心函数，使用 PriorityQueue 实现，包含 closed set 和碰撞检测记录
    - start: 起始配置 (x, y, theta)
    - goal: 目标配置 (x, y, theta)
    - neighbors_fn: 获取邻居节点的函数（4 邻居或 8 邻居）
    - collision_fn: 碰撞检测函数
    """
    open_set = PriorityQueue()  # open set，用于存放待处理节点
    closed_set = set()  # closed set，用于存放已处理节点
    unique_id = 0  # 用于解决优先级相同的问题
    open_set.put((0, unique_id, start))  # 插入 (优先级, 唯一ID, 当前节点)

    came_from = {}  # 记录路径的字典
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    # 用于保存碰撞和无碰撞的配置
    collision_free_configs = []  # 无碰撞点
    colliding_configs = []       # 碰撞点

    while not open_set.empty():
        _, _, current = open_set.get()  # 获取优先级最小的节点

        # 如果当前节点接近目标节点，则回溯路径
        if np.allclose(current, goal, atol=1e-2):
            path = []
            total_cost = g_score[current]
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()  # 路径是从终点到起点的，因此需要反转
            return path, collision_free_configs, colliding_configs, total_cost

        # 将当前节点加入 closed set
        closed_set.add(current)

        # 扩展邻居节点
        for neighbor in neighbors_fn(current):
            # 如果邻居节点已经在 closed set 中，跳过
            if neighbor in closed_set:
                continue

            # 碰撞检测
            if collision_fn(neighbor):
                colliding_configs.append(neighbor[:2])  # 记录碰撞的 (x, y)
                continue  # 如果碰撞，则跳过这个邻居
            else:
                collision_free_configs.append(neighbor[:2])  # 记录无碰撞的 (x, y)

            tentative_g_score = g_score[current] + action_cost(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # 如果是新的节点或者找到更优的路径，更新节点
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                unique_id += 1
                open_set.put((f_score[neighbor], unique_id, neighbor))  # 插入新节点

    return None, collision_free_configs, colliding_configs, float('inf')  # 如果没有找到路径


def normalize_angle(theta):
    """将角度规范化到 [-pi, pi]"""
    theta = np.fmod(theta + np.pi, 2 * np.pi)  # 将角度规范化到 [0, 2π]
    if theta < 0:
        theta += 2 * np.pi
    return theta - np.pi


def get_neighbors_4(current):
    """获取 4 个方向的邻居节点，theta 只允许为 0°, 90°, 180°, 360°"""
    x, y, theta = current
    delta = 0.1  # 平移步长
    theta_values = [0, np.pi / 2, -np.pi / 2, np.pi]  # 允许的角度（0°, 90°, 180°, 360°）

    neighbors = []

    # 遍历四个方向的平移，并保持当前角度不变
    neighbors.append((x + delta, y, normalize_angle(theta)))  # 向右
    neighbors.append((x - delta, y, normalize_angle(theta)))  # 向左
    neighbors.append((x, y + delta, normalize_angle(theta)))  # 向上
    neighbors.append((x, y - delta, normalize_angle(theta)))  # 向下

    # 在四个固定角度间选择下一个角度
    for new_theta in theta_values:
        if new_theta != theta:  # 避免重复当前角度
            neighbors.append((x, y, normalize_angle(new_theta)))

    return neighbors


def get_neighbors_8(current):
    """获取 8 个方向的邻居节点，x 和 y 坐标限制为小数点后一位"""
    x, y, theta = current
    delta = 0.1  # 平移步长
    delta_diagonal = round(delta * np.sqrt(2), 1)  # 计算对角线距离，并限制为小数点后一位
    theta_values = [0, np.pi / 2, -np.pi / 2, np.pi]

    neighbors = [
        (round(x + delta, 1), round(y, 1), normalize_angle(theta)),       # 向右
        (round(x - delta, 1), round(y, 1), normalize_angle(theta)),       # 向左
        (round(x, 1), round(y + delta, 1), normalize_angle(theta)),       # 向上
        (round(x, 1), round(y - delta, 1), normalize_angle(theta)),       # 向下
        (round(x + delta_diagonal, 1), round(y + delta_diagonal, 1), normalize_angle(theta)),  # 右上
        (round(x - delta_diagonal, 1), round(y - delta_diagonal, 1), normalize_angle(theta)),  # 左下
        (round(x + delta_diagonal, 1), round(y - delta_diagonal, 1), normalize_angle(theta)),  # 右下
        (round(x - delta_diagonal, 1), round(y + delta_diagonal, 1), normalize_angle(theta))   # 左上
    ]

    for new_theta in theta_values:
        if new_theta != theta:  # 避免重复当前角度
            neighbors.append((round(x, 1), round(y, 1), normalize_angle(new_theta)))

    return neighbors


def draw_path(path, color=(0, 0, 0, 1)):
    """使用 draw_line 函数绘制路径"""
    for i in range(len(path) - 1):
        draw_line(path[i][:2] + (0.05,), path[i + 1][:2] + (0.05,), width=2, color=color[:3])


def draw_points(points, color, radius=0.05):
    """使用 draw_sphere_marker 绘制点，稍微抬高一点"""
    for point in points:
        draw_sphere_marker(point + (0.05,), radius, color)


def main(screenshot=False):
    # 初始化 PyBullet
    connect(use_gui=True)
    # 加载机器人和障碍物
    robots, obstacles = load_env('pr2doorway.json')

    # 定义活动自由度（DoFs）
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    # 设置起始和目标配置
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi / 2)

    # 使用 4 邻居进行 A* 路径规划
    start_time = time.time()
    print("Using A* with 4 neighbors...")
    path_4, collision_free_4, colliding_4, cost_4 = astar_search(start_config, goal_config, get_neighbors_4, collision_fn)
    print("Planner run time with 4 neighbors: ", time.time() - start_time)
    print("Path cost with 4 neighbors: ", cost_4)

    if path_4 is not None:
        draw_path(path_4, color=(0, 0, 0, 1))  # 使用黑色线条绘制路径
        # draw_points(collision_free_4, color=(0, 0, 1, 1), radius=0.05)  # 蓝色无碰撞点
        # draw_points(colliding_4, color=(1, 0, 0, 1), radius=0.05)  # 红色碰撞点
        execute_trajectory(robots['pr2'], base_joints, path_4, sleep=0.2)
    else:
        print("No path found with 4 neighbors.")

    # 使用 8 邻居进行 A* 路径规划
    # start_time = time.time()
    # print("Using A* with 8 neighbors...")
    # path_8, collision_free_8, colliding_8, cost_8 = astar_search(start_config, goal_config, get_neighbors_8, collision_fn)
    # print("Planner run time with 8 neighbors: ", time.time() - start_time)
    # print("Path cost with 8 neighbors: ", cost_8)
    #
    # if path_8 is not None:
    #     draw_path(path_8, color=(0, 0, 0, 1))  # 使用黑色线条绘制路径
    #     draw_points(collision_free_8, color=(0, 0, 1, 1), radius=0.05)  # 蓝色无碰撞点
    #     draw_points(colliding_8, color=(1, 0, 0, 1), radius=0.05)  # 红色碰撞点
    #     execute_trajectory(robots['pr2'], base_joints, path_8, sleep=0.2)
    # else:
    #     print("No path found with 8 neighbors.")

    # 保持窗口打开
    wait_if_gui()
    disconnect()


if __name__ == '__main__':
    main()
