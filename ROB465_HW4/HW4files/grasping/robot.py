import os
import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import pdb
import time

class Kuka():
    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01, base_pos=[0.44, 0, 0], base_ori=[0, 0, 0, 1]):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = .35
        self.maxForce = 100. # 200.
        self.fingerAForce = .3
        self.fingerBForce = 2.5
        self.fingerTipForce = 2
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 21
        self.useOrientation = 1
        self.kukaEndEffectorIndex = 6
        self.kukaGripperIndex = 10
        self.base_pose = [base_pos, base_ori]
        #lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        #upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        #joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        #restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        #joint damping coefficents
        self.jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001
        ]
        self.reset()

    def reset(self):
        # object_ = p.loadSDF(os.path.join(self.urdfRootPath, "kuka_iiwa/kuka_with_gripper2.sdf"))[0]
        object_ = p.loadURDF('assets/kuka_arm.urdf', useFixedBase=True)
        self.kukaUid = object_

        p.resetBasePositionAndOrientation(self.kukaUid, self.base_pose[0], self.base_pose[1])

        # self.jointPositions = [
        #   0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
        #   -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        # ]

        # for i in range(p.getNumJoints(self.kukaUid)):
        #     print(p.getJointInfo(self.kukaUid, i))
        self.jointPositions = [
            0.0, 0.0, 0.0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0, 0, #Arm joints
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #Gripper joints

        self.numJoints = p.getNumJoints(self.kukaUid)


        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
            p.setJointMotorControl2(self.kukaUid,
                                jointIndex,
                                p.POSITION_CONTROL,
                                targetPosition=self.jointPositions[jointIndex],
                                force=self.maxForce)

        # add FT sensor
        p.enableJointForceTorqueSensor(self.kukaUid, self.kukaEndEffectorIndex, True)

        self.endEffectorPos = [0.0, 0.0, 0.4]
        self.endEffectorAngle = 0

        self.motorNames = []
        self.motorIndices = []

        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.kukaUid, i)
            qIndex = jointInfo[3]

            if qIndex > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

    def applyAction(self, motorCommands, with_top=True):

        #print ("self.numJoints")
        #print (self.numJoints)
        if (self.useInverseKinematics):

            x = motorCommands[0]
            y = motorCommands[1]
            z = motorCommands[2]
            a = motorCommands[3]
            fingerAngleBase = motorCommands[4]
            fingerAngleTip = motorCommands[5]

            state = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)
            actualEndEffectorPos = state[0]
            #print("pos[2] (getLinkState(kukaEndEffectorIndex)")
            #print(actualEndEffectorPos[2])

            self.endEffectorPos[0] = x
            self.endEffectorPos[1] = y
            self.endEffectorPos[2] = z

            self.endEffectorAngle = a

            pos = self.endEffectorPos

            if with_top:
                orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
            else:
                orn = p.getQuaternionFromEuler([-math.pi/2, 0, 0])
            if (self.useNullSpace == 1):
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                              self.kukaEndEffectorIndex,
                                                              pos,
                                                              orn,
                                                              # self.ll,
                                                              # self.ul,
                                                              # self.jr,
                                                              self.rp)
                else:
                    jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                              self.kukaEndEffectorIndex,
                                                              pos,
                                                              # lowerLimits=self.ll,
                                                              # upperLimits=self.ul,
                                                              # jointRanges=self.jr,
                                                              restPoses=self.rp)
            else:
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                              self.kukaEndEffectorIndex,
                                                              pos,
                                                              orn,
                                                              jointDamping=self.jd)
                else:
                    jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos)

            if (self.useSimulation):
                for i in range(self.kukaEndEffectorIndex + 1):
                    p.setJointMotorControl2(bodyUniqueId=self.kukaUid,
                                            jointIndex=i,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=jointPoses[i],
                                            targetVelocity=0,
                                            force=self.maxForce,
                                            maxVelocity=self.maxVelocity,
                                            positionGain=0.3,
                                            velocityGain=1)
            else:
                #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.numJoints):
                    p.resetJointState(self.kukaUid, i, jointPoses[i])

            #fingers
            p.setJointMotorControl2(self.kukaUid,
                                    self.kukaGripperIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=self.endEffectorAngle,
                                    force=self.maxForce)

            p.setJointMotorControl2(self.kukaUid,
                                    12,
                                    p.POSITION_CONTROL,
                                    targetPosition=-fingerAngleBase,
                                    force=self.fingerAForce)
            p.setJointMotorControl2(self.kukaUid,
                                    16,
                                    p.POSITION_CONTROL,
                                    targetPosition=-fingerAngleBase,
                                    force=self.fingerAForce)
            p.setJointMotorControl2(self.kukaUid,
                                    20,
                                    p.POSITION_CONTROL,
                                    targetPosition=-fingerAngleBase,
                                    force=self.fingerAForce)
            
            p.setJointMotorControl2(self.kukaUid,
                                12 + 2,
                                p.POSITION_CONTROL,
                                targetPosition=-fingerAngleTip,
                                force=self.fingerAForce)
            p.setJointMotorControl2(self.kukaUid,
                                16 + 2,
                                p.POSITION_CONTROL,
                                targetPosition=-fingerAngleTip,
                                force=self.fingerAForce)
            p.setJointMotorControl2(self.kukaUid,
                                20 + 2,
                                p.POSITION_CONTROL,
                                targetPosition=-fingerAngleTip,
                                force=self.fingerAForce)

        else:
            for action in range(len(motorCommands)):
                motor = self.motorIndices[action]
                p.setJointMotorControl2(self.kukaUid,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=motorCommands[action],
                                        force=self.maxForce)

if __name__=="__main__":
    p.connect(p.GUI)
    robot = Kuka()
    while True:
        time.sleep(10)
