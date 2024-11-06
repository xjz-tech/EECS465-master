import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))

from robot import Kuka

class World:
    def __init__(self, u=None, pos=None, visualize=True):
        # initialize the simulator and blocks
        if visualize:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF('plane.urdf', useFixedBase=True)
        p.changeDynamics(planeId, -1, lateralFriction=0.99)

        # set camera
        p.resetDebugVisualizerCamera(cameraDistance=1,
                                     cameraYaw=-60,
                                     cameraPitch=-20,
                                     cameraTargetPosition=[0, 0, 0])

        # set gravity
        p.setGravity(0, 0, -10)

        # add the robot
        self.robot = Kuka()

        for _ in range(100):
            p.stepSimulation()

        # drop some blocks in scene
        if u is None:
            u = [0, 0, 0]
        if pos is None:
            pos = np.array([0., 0., 0.2])

    

        # x_b = np.random.randn() / 15.
        # y_b = np.random.randn() / 15.
        # z_b = 0.3
        if len(u) == 3:
            q = [np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
                    np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
                    np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
                    np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])]
        else:
            q = u
        self.block_id = p.loadURDF(os.path.join(file_path, "assets/block.urdf"), [0, 0, 0.05])
        p.changeDynamics(self.block_id, -1, mass=0.2, lateralFriction=0.1)
        p.resetBasePositionAndOrientation(self.block_id, pos, q)

        for _ in range(300):
            p.stepSimulation()

        self.home_pose = np.array([])
    
    def get_block_ids(self):
        return self.block_ids
    
    def get_object_info(self):
        position = p.getBasePositionAndOrientation(self.block_id)[0]

        collision_shape = p.getCollisionShapeData(self.block_id, -1)[0][3]
        collision_shape = np.array(collision_shape)

        max_radius = np.linalg.norm(collision_shape/2)
        dynamics_info = p.getDynamicsInfo(self.block_id, -1)
        mu = dynamics_info[1]

        return position, mu, max_radius

    def grasp(self, gp):
        # plan for robot actions
        # pre-grasp w/ offset
        rotation = np.array([[np.cos(gp[3]), -np.sin(gp[3])],
                             [np.sin(gp[3]), np.cos(gp[3])]])
        go = np.matmul(rotation, np.array([0., -0.02]))
        grasp_pose_robot = np.array([gp[0], gp[1], gp[2], 0., 0.75, 0])
        grasp_offset_a = np.array([go[0], go[1], .42, -gp[3], 0., 0])
        grasp_offset_b = np.array([go[0], go[1], .42 / 1.5, -gp[3], 0., 0])
        robot_command = grasp_pose_robot + grasp_offset_a
        for t in range(300):
            self.robot.applyAction(robot_command)
            time.sleep(.001)
            p.stepSimulation()

        # pre-grasp w/o offset
        robot_command = grasp_pose_robot + grasp_offset_b
        for t in range(300):
            self.robot.applyAction(robot_command)
            time.sleep(.001)
            p.stepSimulation()

        # close fingers
        grasp_pose_robot[-2] = -.7
        grasp_pose_robot[-1] = .5
        robot_command = grasp_pose_robot + grasp_offset_b
        for t in range(300):
            self.robot.applyAction(robot_command)
            time.sleep(.001)
            p.stepSimulation()

        #Get contact points
        contact_points = []
        contact_points_block = p.getContactPoints(bodyA=self.robot.kukaUid, bodyB=self.block_id)
        for i in range(len(contact_points_block)):
            #PyBullet can generate false contact points that have 0 contact force
            if contact_points_block[i][9] > 0:
                contact_points.append(contact_points_block[i])

        return contact_points

if __name__ == '__main__':
    world = World()
    contact_points = world.grasp([0, 0, 0, 0])

        
    
    
