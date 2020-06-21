import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import glob
import time
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
from datetime import datetime

RENDER_HEIGHT = 720
RENDER_WIDTH = 960



class DLOEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0.85, 0.25, 0.3])


        self.EndEffectorIndex = 11
        self.LeftGripperIndex = 9
        self.RightGripperIndex = 10
        self.numJoints = 7
        self.uls = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
        self.lls = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
        self.jrs = [5.9342, 3.6652, 5.9342, 3.1416, 5.9342, 3.9095999999999997, 5.9342]
        self.rps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.RightHandGraspOrientation = p.getQuaternionFromEuler([0., -math.pi*0.75, -math.pi*0.5])
        #self.RightHandGraspOrientation = p.getQuaternionFromEuler([0., math.pi, math.pi * 0.5])

        self.LeftHandGraspOrientation = p.getQuaternionFromEuler([0., math.pi*0.75, -math.pi*0.5])
        print(self.LeftHandGraspOrientation)
        #self.LeftHandGraspOrientation = p.getQuaternionFromEuler([0., math.pi, -math.pi * 0.5])

        self.WorkPos = [0.75, -0.1, 1.2, -0.6532814824381882, -0.6532814824381883, -0.2705980500730985, 0.27059805007309856, 0.04]

        self.StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        #self.ObjStartOrientation = p.getQuaternionFromEuler([0, math.pi*0.5, math.pi*0.5])


        self.ObjStartOrientationEuler = [0, math.pi * 0.5, math.pi*0.5]
        self.ObjStartOrientation = p.getQuaternionFromEuler([0, math.pi * 0.5, math.pi*0.5])

        #self.ringStartOrnEuler = [0, 0, math.pi*0.75]
        self.ringStartOrnEuler = [0, 0, 0]
        self.ringStartOrn = p.getQuaternionFromEuler(self.ringStartOrnEuler)

        self.EndEffectorInitialPos = [0.088, -1.7431537324696934e-12, 1.5210000000000001]
        self.EndEffectorInitialOrn = [0.9238795325113726, 0.38268343236488267, -1.8738412424764667e-12, 4.523852941323606e-12]
        self.OpenGrasp = 0.04
        #0.013
        self.CloseGrasp = 0.013
        self.jointMaxforce = 87.0
        self.maxVelocity=0.5
        #Specified finger max force is 20, but does not work in simulation, need 200
        self.fingerMaxForce = 20



        self.timestep = 1./240.

        self.RightHandInitialPos = [0.75, -0.1, 1.2, -0.6532814824381882, -0.6532814824381883, -0.2705980500730985, 0.27059805007309856, 0.04]
        self.LeftHandInitialPos = [0.75, 0.4, 1.2, 0.6532814824381882, 0.6532814824381883, -0.2705980500730985, 0.27059805007309856, 0.04]


        self.GraspPosOffset = 0.05


    def applyAction(self, action):
        # dv = 0.5
        # dx = action[0] * dv
        # dy = action[1] * dv
        # dz = action[2] * dv

        worker = action[0]
        if(worker == 1):
            workerId = self.pandaId
        else:
            workerId = self.panda2Id

        goal = action[1:4]
        orientation = action[4:8]
        grasp = action[8]

        currentPose = p.getLinkState(self.pandaId, self.EndEffectorIndex)
        currentPosition = currentPose[0]
        # print("Current Position: ", currentPosition)

        # newPosition = [currentPosition[0] + dx, currentPosition[1] + dy, currentPosition[2] + dz]


        # Use Inverse Kinematics with constraints to calculate goal pose of 7 joints
        jointPoses = p.calculateInverseKinematics(workerId, self.EndEffectorIndex, goal, orientation, self.lls,
                                                  self.uls, self.jrs, self.rps, solver=p.IK_DLS)

        # Set joint motor movement one by one with constraints on the position gain
        for i in range(self.numJoints):
            p.setJointMotorControl2(workerId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    positionGain=0.06,
                                    maxVelocity=4)

        time.sleep(1./240.)

        p.setJointMotorControl2(workerId, self.LeftGripperIndex, p.POSITION_CONTROL, grasp, maxVelocity=0.25,
                                force=self.fingerMaxForce)
        p.setJointMotorControl2(workerId, self.RightGripperIndex, p.POSITION_CONTROL, grasp, maxVelocity=0.25,
                                force=self.fingerMaxForce)

    # def get_extended_state(self, state):
    def step_a(self, action):
        self.applyAction(action)

    def step(self, agents, seq_state, action):
        if action == None:
            return agents + [3] + 25*[0]

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        goal = False

        self.applyAction(action)

        p.stepSimulation()

        worker = action[0]
        if (worker == 1):
            workerId = self.pandaId
        else:
            workerId = self.panda2Id

        #Get observations
        #TO DO ----- Orientation -> aligns with movement direction of gripper?
        state_object = list(p.getBasePositionAndOrientation(self.cylindId)[0])

        state_robot = list(p.getLinkState(self.pandaId, self.EndEffectorIndex)[0]) + list(p.getLinkState(self.pandaId, self.EndEffectorIndex)[1])
        state_robot2 = list(p.getLinkState(self.panda2Id, self.EndEffectorIndex)[0]) + list(p.getLinkState(self.panda2Id, self.EndEffectorIndex)[1])
        state_fingers = (p.getJointState(self.pandaId, self.LeftGripperIndex)[0], p.getJointState(self.pandaId, self.RightGripperIndex)[0])
        state_fingers2 = (p.getJointState(self.panda2Id, self.LeftGripperIndex)[0], p.getJointState(self.panda2Id, self.RightGripperIndex)[0])

        state = agents + [seq_state] + state_robot + list(state_fingers) + state_robot2 + list(state_fingers2) + state_object

        return state

    def limits(self):
        uls, lls, jrs, rps = [], [], [], []

        for i in range(7):
            joint_info = p.getJointInfo(self.panda2Id, i)
            name, ll, ul = joint_info[1], joint_info[8], joint_info[9]
            jr = ul - ll
            rp = p.getJointState(self.panda2Id, i)[0]
            uls.append(ul)
            lls.append(ll)
            jrs.append(jr)
            rps.append(rp)

        return uls, lls, jrs, rps

    def reset(self):
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        #p.resetSimulation()
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.025)
        p.setTimestep = self.timestep

        #Enable rendering after we loaded everything
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        urdfRootPath = pybullet_data.getDataPath()
        texture_paths = glob.glob(os.path.join('dtd', '**', '*.jpg'), recursive=True)

        p.setGravity(0, 0, -10)

        #Load plane
        planeId = p.loadURDF("plane.urdf")

        #Generate table with random spawn texture
        tableId = p.loadURDF("table/table.urdf", basePosition = [0.6, 0.25, 0.1], baseOrientation=self.StartOrientation)
        random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]
        textureId = p.loadTexture(random_texture_path)
        p.changeVisualShape(tableId, -1, textureUniqueId=textureId)



        #Spawn Loop
        ring_spawn = [0.7, 0.25, 1.0]
        #ring_spawn = (random.uniform(0.6, 0.75), random.uniform(-0.1, 0.1), 1.0)
        self.spawn_ring(ring_spawn, self.ringStartOrn)
        #state_ring = list(p.getBasePositionAndOrientation(self.torusId)[0]) + list(p.getBasePositionAndOrientation(self.torusId)[1])
        state_ring = list(p.getBasePositionAndOrientation(self.torusId)[0]) + list(
            self.ringStartOrnEuler)

        print(state_ring)
        #Add Height Offset - Due to spawn position = bottom of ring instead of centre
        state_ring[2] = state_ring[2]+0.0025



        #Spawn Deformable Object
        #TO DO - ADDS IN ORIENTATION LATER
        state_object = (random.uniform(0.55, 0.75), random.uniform(-0.25, 0.55), 0.74)
        self.spawn_deformable_object(state_object)
        #self.spawn_rope_50(state_object)




        #Spawn Object at Random Position
        #state_object = [random.uniform(0.4, 0.75), -0.05, 0.74]
        #self.objId = p.loadURDF("000/000.urdf", basePosition=state_object, baseOrientation=self.StartOrientation)
        # state_object = [0.6, 0.0, 0.75]
        # self.spawn_rope_50(state_object)


        #Spawn Robot in initial pose
        self.pandaId = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), basePosition=[0, 0, 0.7], baseOrientation=self.StartOrientation, useFixedBase=True)
        self.panda2Id = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), basePosition=[0, 0.5, 0.7], baseOrientation=self.StartOrientation, useFixedBase=True)


        #Create constraint to keep the fingers at centre
        c = p.createConstraint(self.pandaId, 9, self.pandaId, 10, jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0.7], childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=20)

        c2 = p.createConstraint(self.panda2Id, 9, self.panda2Id, 10, jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0.5, 0.7], childFramePosition=[0, 0, 0])
        p.changeConstraint(c2, gearRatio=-1, erp=0.1, maxForce=20)

        #Initial State
        seq_state = [0]

        #INsert Function to determine main worker
        agents = self.worker_and_helper(state_object, state_ring)

        for _ in range(400):
            self.to_initial_pos()
            p.stepSimulation()


        #Obtain state of robot and fingers

        #TO DO - MERGE TWO ROBOT STATES
        state_robot = list(p.getLinkState(self.pandaId, self.EndEffectorIndex)[0]) + list(p.getLinkState(self.pandaId, self.EndEffectorIndex)[1])
        state_robot2 = list(p.getLinkState(self.panda2Id, self.EndEffectorIndex)[0]) + list(p.getLinkState(self.panda2Id, self.EndEffectorIndex)[1])
        #print(state_robot2)
        state_fingers = (p.getJointState(self.pandaId, self.LeftGripperIndex)[0], p.getJointState(self.pandaId, self.RightGripperIndex)[0])
        state_fingers2 = (p.getJointState(self.panda2Id, self.LeftGripperIndex)[0], p.getJointState(self.panda2Id, self.RightGripperIndex)[0])

        #observation = state_robot + state_fingers + state_robot2 + state_fingers2 + agents

        state = agents + seq_state + state_robot + list(state_fingers) + state_robot2 + list(state_fingers2) + list(state_object)

        #Restart rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        #return observation, state_object, state_ring
        return state, state_ring

    def worker_and_helper(self, state_object, goal):
        if(state_object[1] < goal[1]):
            return [1, 2]
        return [2, 1]

    def render(self, mode='human'):

        viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.75, 0.2, 0.8], distance=1.0, yaw=90, pitch=-90, roll=0, upAxisIndex=2)


        projectionMatrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT, nearVal=0.1, farVal=100.0)

        _, _, rgbImg, _, _ = p.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=viewMatrix,
                                                                   projectionMatrix=projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(rgbImg, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]

        return rgb_array

    def to_initial_pos(self):
        goal = self.RightHandInitialPos[0:3]
        orientation = self.RightHandInitialPos[3:7]
        jointPoses = p.calculateInverseKinematics(self.pandaId, self.EndEffectorIndex, goal, orientation, self.lls,
                                                  self.uls, self.jrs, self.rps)

        # Set joint motor movement one by one / group
        for i in range(self.numJoints):
            p.setJointMotorControl2(self.pandaId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    positionGain=0.08,
                                    maxVelocity=5)

        goal2 = self.LeftHandInitialPos[0:3]
        orientation2 = self.LeftHandInitialPos[3:7]
        jointPoses2 = p.calculateInverseKinematics(self.panda2Id, self.EndEffectorIndex, goal2, orientation2, self.lls,
                                                  self.uls, self.jrs, self.rps)

        # Set joint motor movement one by one / group
        for i in range(self.numJoints):
            p.setJointMotorControl2(self.panda2Id,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses2[i],
                                    positionGain=0.08,
                                    maxVelocity=5)


    # def at_work_pos(self, state_object, state_robot, state_fingers):
    #
    #     if(tuple(self.WorkPos[0:3])[0]-0.05 <= state_robot[0] <= tuple(self.WorkPos[0:3])[0]+0.05 and tuple(self.WorkPos[0:3])[1]-0.05 <= state_robot[1] <= tuple(self.WorkPos[0:3])[1]+0.05 and tuple(self.WorkPos[0:3])[2]-0.05 <= state_robot[2] <= tuple(self.WorkPos[0:3])[2]+0.05):
    #         return True
    #
    #     return False
    #
    #
    # def object_grasp(self, state_object, state_robot, state_fingers):
    #
    #     if(state_object[0]-0.05 <= state_robot[0] <= state_object[0]+0.05 and state_object[1]-0.05 <= state_robot[1] <= state_object[1]+0.05 and state_object[2]-0.05 <= state_robot[2] <= state_object[2]+0.05 and (0.020 <= state_fingers[0] or 0.020 <= state_fingers[1])):
    #         return True
    #
    #     return False
    #
    #
    # def ready_to_grasp(self, state_object, state_robot, state_fingers):
    #
    #     if(state_object[0]-0.05 <= state_robot[0] <= state_object[0]+0.05 and state_object[1]-0.05 <= state_robot[1] <= state_object[1]+0.05 and (state_object[2]-0.05 > state_robot[2] or state_robot[2] > state_object[2]+0.05)):
    #         return True
    #
    #     return False
    #
    # def grasp_in_place(self, state_fingers):
    #
    #     if(0.005 <= state_fingers[0] <= 0.020 and 0.005 <= state_fingers[1] <= 0.020):
    #         return True
    #
    #     return False
    #
    #
    # def ready_to_insert(self, state_object, state_robot, state_fingers, state_ring):
    #
    #     if (state_ring[0] - 0.05 <= state_robot[0] <= state_ring[0] + 0.05 and state_ring[1] - 0.15 <=
    #             state_robot[1] <= state_ring[1] - 0.05 and state_ring[2] + 0.025 <= state_robot[2] <= state_ring[2] + 0.080):
    #         return True
    #
    #     return False
    #
    # def insert_in_place(self, state_object, state_robot, state_fingers, state_ring):
    #
    #     if (state_ring[0] - 0.05 <= state_object[0] <= state_ring[0] + 0.05 and state_ring[1] - 0.05 <=
    #             state_object[1] <= state_ring[1] + 0.05 and state_ring[2] + 0.06 <= state_object[2] <= state_ring[2] + 0.08):
    #         return True
    #
    #     return False
    #
    # def ready_to_pull(self, state_object, state_robot, state_fingers, state_ring):
    #     print(state_robot)
    #     if(state_object[0]-0.05 <= state_robot[0] <= state_object[0]+0.05 and state_object[2]-0.05 <= state_robot[2] <= state_object[2]+0.05):
    #         return True
    #
    #     return False

    def equate(self, a, b):
        for i in range(len(a)):
            if(round(a[i], 10) != round(b[i], 10)):
                return False
        return True

    def compute_grasp_orn(self, state_object):
        #TO DO
        grasp_orn = p.getQuaternionFromEuler([0., -math.pi, math.pi/2.])
        return grasp_orn

    def compute_insert_orn(self, state_ring):
        print(state_ring)
        print(p.getQuaternionFromEuler([0, 0, math.pi*0.75]))
        if(state_ring == p.getQuaternionFromEuler([0, 0, 0])):
            #print("Normal Orientation -> Proceed with current insert orientation")
            return(p.getQuaternionFromEuler([0, -math.pi*0.75, -math.pi*0.5]))

        elif(self.equate(state_ring, p.getQuaternionFromEuler([0, 0, math.pi*0.75]))):
            #print("New Orn")
            return(p.getQuaternionFromEuler([0, -math.pi*0.75, -math.pi*0.75]))

        else:
            #print("LOLLLLLLLLLL")
            return(p.getQuaternionFromEuler([0, -math.pi*0.75, -math.pi*0.5]))

    def compute_pull_orn(self, state_ring):
        if (state_ring == p.getQuaternionFromEuler([0, 0, 0])):
            #print("Normal Orientation -> Proceed with current insert orientation")
            return (p.getQuaternionFromEuler([0, -math.pi * 0.75, math.pi * 0.5]))

    def state_match_high(self, state_a, state_b):
        if(state_a[0]-0.05 <= state_b[0] <= state_a[0]+0.05 and state_a[1]-0.05 <= state_b[1] <= state_a[1]+0.05 and state_a[2]-0.05 <= state_b[2] <= state_a[2]+0.05):
            return True
        return False

    def state_match(self, state_a, state_b):
        if(state_a[0]-0.025 <= state_b[0] <= state_a[0]+0.025 and state_a[1]-0.025 <= state_b[1] <= state_a[1]+0.025 and state_a[2]-0.025 <= state_b[2] <= state_a[2]+0.025):
            return True
        return False

    def at_initial_pos(self, state_robot, workerId):
        if(workerId == 1):
            if(self.state_match_high(self.RightHandInitialPos[0:3], state_robot[0:3])):
                return True
            else:
                return False
        else:
            if (self.state_match_high(self.LeftHandInitialPos[0:3], state_robot[0:3])):
                return True
            else:
                return False

    def height_aligned(self, state_a, state_b):
        if(state_b[2]-0.05 <= state_a[2] <= state_b[2]+0.05):
            return True
        return False

    def on_object(self, state_robot, state_object):
        if(state_object[0]-0.05 <= state_robot[0] <= state_object[0]+0.05 and state_object[1]-0.05 <= state_robot[1] <= state_object[1]+0.05 and state_object[2]-0.05 <= state_robot[2] <= state_object[2]+0.05):
            return True
        return False

    def horizontal_mismatch(self, state_a, state_b):
        if(state_b[1]-0.05 <= state_a[1] <= state_b[1]+0.05):
            return False
        return True

    def align_not_z(self, state_a, state_b):
        if (state_b[0] - 0.05 <= state_a[0] <= state_b[0] + 0.05 and state_b[1] - 0.05 <= state_a[1] <= state_b[1] + 0.05 and not state_b[2] - 0.05 <= state_a[2] <= state_b[2] + 0.05):
            return True
        return False

    def align_not_y(self, state_a, state_b):
        if(state_b[0]-0.05 <= state_a[0] <= state_b[0]+0.05 and not state_b[1]-0.05 <= state_a[1] <= state_b[1]+0.05 and state_b[2]-0.05 <= state_a[2] <= state_b[2]+0.05):
            return True
        return False

    def object_in_hand(self, state_fingers):
        if(0.005 <= state_fingers[0] <= 0.015 and 0.005 <= state_fingers[1] <= 0.015):
            return True
        return False

    def grasp_point(self, state_object, place):
        if(place == 0):
            return state_object
        elif(place == 1):
            return [state_object[0], state_object[1]-0.05, state_object[2]], [state_object[0], state_object[1]+0.05, state_object[2]]
        else:
            return [state_object[0], state_object[1]+0.05, state_object[2]], [state_object[0], state_object[1]-0.05, state_object[2]]

    def grasp_orientation(self, workerId):
        if(workerId == 1):
            return list(self.RightHandGraspOrientation), list(self.LeftHandGraspOrientation)
        else:
            return list(self.LeftHandGraspOrientation), list(self.RightHandGraspOrientation)

    def action_orientation(self, agents, goal):
        if (goal[2] == 0 or goal[2] == math.pi):
            if (agents[0] == 1):
                return list(p.getQuaternionFromEuler([0., -math.pi*0.75, -math.pi*0.5])), list(
                    p.getQuaternionFromEuler([0., math.pi*0.75, -math.pi*0.5]))
            else:
                return list(
                    p.getQuaternionFromEuler([0., math.pi*0.75, -math.pi*0.5])), list(p.getQuaternionFromEuler([0., -math.pi*0.75, -math.pi*0.5]))

        elif (goal[2] == math.pi * 0.25 or goal[2] == -math.pi * 0.75):
            if(agents[0] == 1): return [-1.0, 1.0]
            else: return [1.0, -1.0]

        elif (goal[2] == math.pi * 0.75 or goal[2] == -math.pi * 0.25):
            if(agents[0] == 1): return list(p.getQuaternionFromEuler([-math.pi, -math.pi*0.25, math.pi*0.25])), list(p.getQuaternionFromEuler([-math.pi, math.pi*0.25, math.pi*0.25]))
            else: return list(p.getQuaternionFromEuler([-math.pi, math.pi*0.25, math.pi*0.25])), list(p.getQuaternionFromEuler([-math.pi, -math.pi*0.25, math.pi*0.25]))

    def action_direction(self, workerId):
        if(workerId == 1):
            return 1.0
        else:
            return -1.0

    def movement_direction(self, agents, goal):
        if (goal[2] == 0 or goal[2] == math.pi):
            if (agents[0] == 1):
                return [0.0, 1.0]
            else:
                return [0.0, -1.0]

        elif (goal[2] == math.pi * 0.25 or goal[2] == -math.pi * 0.75):
            if(agents[0] == 1): return [-1.0, 1.0]
            else: return [1.0, -1.0]

        elif (goal[2] == math.pi * 0.75 or goal[2] == -math.pi * 0.25):
            if(agents[0] == 1): return [1.0, 1.0]
            else: return [-1.0, -1.0]

    def temp_position(self, goal, state_object_helper, direction):
        dir_invert =  [-1*direction[0]] + [-1*direction[1]]

        pre_insert = [goal[0] + dir_invert[0]*0.3] + [goal[1] + dir_invert[1]*0.3] + [goal[2]]
        pre_pull = [state_object_helper[0] + direction[0]*0.3] + [state_object_helper[1] + direction[1]*0.3] + [state_object_helper[2]]

        return pre_insert, pre_pull

    #def object_align_hole(self, goal_pos, state_object_pos, direction):

    def align_z(self, state_a, state_b):
        if (state_b[2] - 0.05 <= state_a[2] <= state_b[2] + 0.05):
            return True
        return False

    def state_action_backup(self, agents, seq_state, state_worker, state_helper, state_object, goal):

        workerId = [agents[0]]
        helperId = [agents[1]]

        state_object_worker, state_object_helper = self.grasp_point(state_object, agents[0])

        worker_grasp_orn, helper_grasp_orn = self.grasp_orientation(agents[0])

        dir = self.action_direction(agents[0])

        dirTest = self.movement_direction(agents, goal[3:])
        print("Test Dir: ", dirTest)

        insert_test, pull_test = self.temp_position(goal[0:3], state_object_helper[0:3], dirTest)
        print("Insert Test: ", insert_test)
        print("Pull Test: ", pull_test)

        if(seq_state == 0):

            if(self.at_initial_pos(state_worker, agents[0])):
                #print("Moving to Position above object")
                goal_state = [state_object_worker[0], state_object_worker[1], state_object_worker[2]+0.2]
                action = workerId + goal_state + worker_grasp_orn + [self.OpenGrasp]

            elif(self.align_not_z(state_worker, state_object_worker)):
                #print("Moving to Grasp Position")
                action = workerId + state_object_worker + worker_grasp_orn + [self.OpenGrasp]

            elif(self.on_object(state_worker, state_object_worker) and not self.object_in_hand(state_worker[7:])):
                #print("Grasping")
                action = workerId + state_object_worker + worker_grasp_orn + [self.CloseGrasp]

            elif(self.on_object(state_worker, state_object_worker) and self.object_in_hand(state_worker[7:])):
                #print("Lifting")
                #TO DO - Lift to ring height

                #NEED TO CHANGE PRE POSITION ACCORDING TO ORIENTATION
                #goal_state = [goal[0], goal[1]-(dir*0.2), goal[2]]
                goal_state = [goal[0], goal[1] - (dirTest[1] * 0.2), goal[2]]
                print("Goal State: ", goal_state)
                action = workerId + goal_state + worker_grasp_orn + [self.CloseGrasp]
                seq_state = 1
            else:
                #print("No Match State")
                if(agents[0] == 1): action = workerId + list(self.RightHandInitialPos)
                else: action = workerId + list(self.LeftHandInitialPos)

        elif(seq_state == 1):

            if(self.align_not_y(state_object, goal)):
                #print("Inserting")
                #TO DO - right now we insert to object = ring
                #action = workerId + [state_worker[0], state_worker[1]+(dir*0.15), state_worker[2]] + worker_grasp_orn + [self.CloseGrasp]
                action = workerId + [state_worker[0], state_worker[1] + (dirTest[1] * 0.15),
                                     state_worker[2]] + worker_grasp_orn + [self.CloseGrasp]

            elif(self.state_match(state_object, goal)):
                #print("IN HOLE")
                #action = helperId + [state_object[0], state_object[1]+(dir*0.225), state_object[2]] + helper_grasp_orn + [self.OpenGrasp]
                action = helperId + [state_object[0], state_object[1] + (dirTest[1] * 0.225),
                                     state_object[2]] + helper_grasp_orn + [self.OpenGrasp]
                seq_state = 2

            else:
                #print("No Match State")
                seq_state = 0
                if (agents[0] == 1): action = workerId + list(self.RightHandInitialPos)
                else: action = workerId + list(self.LeftHandInitialPos)

        elif(seq_state == 2):

            if(self.align_not_y(state_helper, state_object_helper)):
                #print("Helper Approaching")
                #TO DO - right now im using state of helper . should be dependent on object ?
                action = helperId + state_object_helper + helper_grasp_orn + [self.OpenGrasp]

            elif(self.state_match(state_helper, state_object_helper) and not self.object_in_hand(state_helper[7:])):
                #print("Helper Grasping")
                action = helperId + state_object_helper + helper_grasp_orn + [self.CloseGrasp]

            elif(self.object_in_hand(state_worker[7:]) and self.object_in_hand(state_helper[7:])):
                #print("Worker Letting Go")
                action = workerId + state_worker[0:3] + worker_grasp_orn + [self.OpenGrasp]

            elif(self.object_in_hand(state_helper[7:]) and not self.object_in_hand(state_worker[7:]) and self.state_match(state_object, goal)):
                #print("Helper Pulling")
                #action = helperId + [state_object_helper[0], state_object_helper[1]+(dir*0.2), state_object_helper[2]] + helper_grasp_orn + [self.CloseGrasp]
                print("Goal Pull State: ", [state_object_helper[0], state_object_helper[1] + (dirTest[1] * 0.2),
                                     state_object_helper[2]])
                action = helperId + [state_object_helper[0], state_object_helper[1] + (dirTest[1] * 0.2),
                                     state_object_helper[2]] + helper_grasp_orn + [self.CloseGrasp]

            elif(self.align_not_y(state_object, goal)):
                #print("Pulling Finished")
                action = helperId + state_object_helper + helper_grasp_orn + [self.OpenGrasp]
                seq_state = 3

            else:
                #print("No Match State")
                if (agents[0] == 1):
                    action = helperId + list(self.LeftHandInitialPos)
                else:
                    action = helperId + list(self.RightHandInitialPos)

        return action, seq_state

    def state_action(self, agents, seq_state, state_worker, state_helper, state_object, goal):

        workerId = [agents[0]]
        helperId = [agents[1]]

        state_object_worker, state_object_helper = self.grasp_point(state_object, agents[0])

        worker_grasp_orn, helper_grasp_orn = self.grasp_orientation(agents[0])
        #print("Helper Grasp: ", helper_grasp_orn)
        worker_orn, helper_orn = self.action_orientation(agents, goal[3:])
       # print("Helper Orn: ", helper_orn)
        dir = self.action_direction(agents[0])

        dirTest = self.movement_direction(agents, goal[3:])
        #print("Test Dir: ", dirTest)

        insert_test, pull_test = self.temp_position(goal[0:3], goal[0:3], dirTest)
        #print("Insert Test: ", insert_test)
        #print("Pull Test: ", pull_test)

        #print("State OBject Helper: ", state_object_helper)

        if(seq_state == 0):

            if(self.at_initial_pos(state_worker, agents[0])):
                print("Moving to Position above object")
                goal_state = [state_object_worker[0], state_object_worker[1], state_object_worker[2]+0.2]
                action = workerId + goal_state + worker_grasp_orn + [self.OpenGrasp]

            elif(self.align_not_z(state_worker, state_object_worker)):
                #print("Moving to Grasp Position")
                action = workerId + state_object_worker + worker_grasp_orn + [self.OpenGrasp]

            elif(self.on_object(state_worker, state_object_worker) and not self.object_in_hand(state_worker[7:])):
                #print("Grasping")
                action = workerId + state_object_worker + worker_grasp_orn + [self.CloseGrasp]

            elif(self.on_object(state_worker, state_object_worker) and self.object_in_hand(state_worker[7:])):
                #print("Lifting")
                #TO DO - Lift to ring height

                #NEED TO CHANGE PRE POSITION ACCORDING TO ORIENTATION
                #goal_state = [goal[0], goal[1]-(dir*0.2), goal[2]]
                goal_state = [goal[0], goal[1] - (dirTest[1] * 0.1), goal[2]]
                print("Goal State: ", goal_state)
                action = workerId + insert_test + worker_orn + [self.CloseGrasp]
                seq_state = 1
            else:
                #print("No Match State")
                if(agents[0] == 1): action = workerId + list(self.RightHandInitialPos)
                else: action = workerId + list(self.LeftHandInitialPos)

        elif(seq_state == 1):

            if(self.align_z(state_object, goal) and not self.state_match(state_object, goal)):
                print("Inserting")
                #TO DO - right now we insert to object = ring
                #action = workerId + [state_worker[0], state_worker[1]+(dir*0.15), state_worker[2]] + worker_grasp_orn + [self.CloseGrasp]
                action = workerId + [state_worker[0] + (dirTest[0] * 0.05), state_worker[1] + (dirTest[1] * 0.05),
                                     state_worker[2]] + worker_orn + [self.CloseGrasp]

            elif(self.state_match(state_object, goal)):
                print("IN HOLE")
                #action = helperId + [state_object[0], state_object[1]+(dir*0.225), state_object[2]] + helper_grasp_orn + [self.OpenGrasp]
                action = helperId + pull_test + helper_orn + [self.OpenGrasp]
                seq_state = 2

            else:
                print("No Match State")
                seq_state = 0
                if (agents[0] == 1): action = workerId + list(self.RightHandInitialPos)
                else: action = workerId + list(self.LeftHandInitialPos)

        elif(seq_state == 2):

            if(self.align_not_y(state_helper, state_object_helper)):
                print("Helper Approaching")
                #TO DO - right now im using state of helper . should be dependent on object ?
                action = helperId + state_object_helper + helper_orn + [self.OpenGrasp]

            elif(self.state_match(state_helper, state_object_helper) and not self.object_in_hand(state_helper[7:])):
                #print("Helper Grasping")
                action = helperId + state_object_helper + helper_orn + [self.CloseGrasp]

            elif(self.object_in_hand(state_worker[7:]) and self.object_in_hand(state_helper[7:])):
                #print("Worker Letting Go")
                action = workerId + state_worker[0:3] + worker_orn + [self.OpenGrasp]

            elif(self.object_in_hand(state_helper[7:]) and not self.object_in_hand(state_worker[7:]) and self.state_match(state_object, goal)):
                #print("Helper Pulling")
                #action = helperId + [state_object_helper[0], state_object_helper[1]+(dir*0.2), state_object_helper[2]] + helper_grasp_orn + [self.CloseGrasp]
                print("Goal Pull State: ", [state_object_helper[0] + (dirTest[0] * 0.2), state_object_helper[1] + (dirTest[1] * 0.2),
                                     state_object_helper[2]])
                action = helperId + pull_test + helper_orn + [self.CloseGrasp]

            elif(self.align_not_y(state_object, goal)):
                #print("Pulling Finished")
                action = helperId + state_object_helper + helper_orn + [self.OpenGrasp]
                seq_state = 3

            else:
                print("No Match State 3")
                if (agents[0] == 1):
                    action = helperId + list(self.LeftHandInitialPos)
                else:
                    action = helperId + list(self.RightHandInitialPos)

        return action, seq_state

    def object_valid(self, state_object):
        if(0.3 <= state_object[0] <= 0.9 and -0.2 <= state_object[1] <= 0.6 and 0.73 <= state_object[2]):
            return True
        return False

    def compute_orientation(self, goal):
        #TO DO - USING LOOK UP TABLE RIGHT NOW >>>> FIND A CALCULATION FOR THIS
        if(goal[2] == 0 or goal[2] == math.pi):
            return p.getQuaternionFromEuler([0, -math.pi*0.75, -math.pi*0.5]), p.getQuaternionFromEuler([0, math.pi*0.75, -math.pi*0.5])
        elif(goal[2] == math.pi*0.25 or goal[2] == -math.pi*0.75):
            return p.getQuaternionFromEuler([math.pi, math.pi * 0.25, math.pi * 0.75]), p.getQuaternionFromEuler(
                [math.pi, -math.pi * 0.25, math.pi * 0.75])
        elif(goal[2] == math.pi*0.75 or goal[2] == -math.pi*0.25):
            return p.getQuaternionFromEuler([-math.pi, -math.pi * 0.25, math.pi * 0.25]), p.getQuaternionFromEuler(
                [-math.pi, math.pi * 0.25, math.pi * 0.25])

    def action_planner(self, state, goal):
        print("Plan")
        #Break down state representations
        agents = state[0:2]
        seq_state = state[2]

        #TO DO state_worker and helper should depend on worker and helper index.
        if(agents[0] == 1):
            state_worker, state_helper = state[3:12], state[12:21]
        elif(agents[0] == 2):
            state_helper, state_worker = state[3:12], state[12:21]

        state_object = state[21:]



        #Change Sequential State
        if not self.object_valid(state_object):
            return agents, 3, None

        action, seq_state = self.state_action(agents, seq_state, state_worker, state_helper, state_object, goal)

        #Planning Failure

        return agents, seq_state, action


    def planner(self, observation, state_object, state_ring):
        #TO DO

        #Determine state of object
        #state_object = state_object
        print("State Robots ", observation)
        print("State Object: ", state_object)
        print("State Ring: ", state_ring)

        #Determine worker based on robot
        worker = [observation[10]]
        helper = [observation[11]]

        #Determine state of robot - joints & fingers
        state_robot = observation[0:3]
        state_fingers = observation[3:5]

        #Determine state

        #Initial State -> Work State
        if(state_robot == tuple(self.EndEffectorInitialPos)):
            print("At Initial State -> Progress to Work Pose")
            action = worker + self.WorkPos

        #Work State if joint poses = start poses
        #Work State -> Pre-Grasp State
        elif(self.at_work_pos(state_object, state_robot, state_fingers)):
            print("At Start State -> Progress to Pre-Grasp Pose")
            fingers = [self.OpenGrasp]
            new_state = [state_object[0], state_object[1], state_object[2]+0.2]
            print("New State: ", new_state)
            action = worker + new_state + list(self.RightHandGraspOrientation) + fingers

        #If EndEffector Pose's X & Y = Object Pose's X & Y with Z+0.2
        #Pre-Grasp State -> Grasp State
        elif(self.ready_to_grasp(state_object, state_robot, state_fingers)):
            print("At Pre-Grasp Pose -> Progress to Grasp Pose")
            fingers = [self.OpenGrasp]
            action = worker + list(state_object) + list(self.RightHandGraspOrientation) + fingers
            #print(action)

        #If End Effector Pos = Object Pos
        #Grasp State -> Close Fingers
        elif(self.object_grasp(state_object, state_robot, state_fingers)):
            print("Grasp State -> Grasping")
            fingers = [self.CloseGrasp]
            action = worker + list(state_object) + list(self.RightHandGraspOrientation) + fingers

        #Close Fingers -> Pre-Insert Pose
        elif(self.grasp_in_place(state_fingers) and not self.ready_to_insert(state_object, state_robot, state_fingers, state_ring[0]) and not self.insert_in_place(state_object, state_robot, state_fingers, state_ring[0])):
            print("Moving Object to Pre-Insert Pose")
            fingers = [self.CloseGrasp]
            new_state = [state_ring[0][0], state_ring[0][1]-0.05, state_ring[0][2]+0.075]
            #insert_orn = self.compute_insert_orn(state_ring[1])
            action = worker + new_state + list(self.compute_insert_orn(state_ring[1])) + fingers
            #print("Action Test: ", action)

        elif(self.ready_to_insert(state_object, state_robot, state_fingers, state_ring[0])):
            print("Inserting")
            fingers = [self.CloseGrasp]
            new_state = [state_robot[0], state_robot[1]+0.07, state_robot[2]]
            action = worker + new_state + list(self.compute_insert_orn(state_ring[1])) + fingers

        elif(self.insert_in_place(state_object, state_robot, state_fingers, state_ring[0]) and not self.ready_to_pull(state_object, observation[5:8], state_fingers, state_ring[0])):
            print("WO CAO NI MA")
            fingers = [self.OpenGrasp]
            new_state = [state_ring[0][0], state_ring[0][1] + 0.05, state_ring[0][2]+0.08]
            action = [2] + new_state + list(self.compute_pull_orn(state_ring[1])) + fingers

        elif (self.insert_in_place(state_object, state_robot, state_fingers, state_ring[0]) and self.ready_to_pull(state_object, observation[5:8], state_fingers, state_ring[0]) and not self.grasp_in_place(observation[8:10])):
            print("PULLLLLLLLL")
            fingers = [self.CloseGrasp]
            new_state = [state_ring[0][0], state_ring[0][1] + 0.05, state_ring[0][2] + 0.08]
            action = [2] + new_state + list(self.compute_pull_orn(state_ring[1])) + fingers

        elif(self.grasp_in_place(state_fingers) and self.grasp_in_place(observation[8:10])):
            print("Worker Let Go")
            fingers = [self.OpenGrasp]
            action = worker + list(state_robot) + list(self.compute_insert_orn(state_ring[1])) + fingers

        elif(self.grasp_in_place(observation[8:10]) and not self.grasp_in_place(state_fingers) and self.insert_in_place(state_object, state_robot, state_fingers, state_ring[0])):
            print("PULLLLLLLLLINGGGGG")
            fingers = [self.CloseGrasp]
            new_state = [observation[5], observation[6] + 0.05, observation[7]]
            action = [2] + new_state + list(self.compute_pull_orn(state_ring[1])) + fingers

        #Not match any current state -> Reset
        else:
            print("No Match State")
            fingers = [self.OpenGrasp]
            action = worker + self.EndEffectorInitialPos + list(self.EndEffectorInitialOrn) + fingers

        #Plan actions

        #Return corresponding actions
        return action

    def simple_grasp(self, state_object):
        initial_action = [1] + self.WorkPos
        state_object[2] = state_object[2] + 0.2

        grasp = [0.04]
        action = [1] + state_object + list(self.RightHandGraspOrientation) + grasp

        state_object[2] = state_object[2] - 0.19
        action2 = [1] + state_object + list(self.RightHandGraspOrientation) + grasp

        grasp = [0.008]
        action3 = [1] + state_object + list(self.RightHandGraspOrientation) + grasp

        state_object[2] = state_object[2] + 0.2
        action4 = [1] + state_object + list(self.RightHandGraspOrientation) + grasp

        state_object[1] = state_object[1] + 0.3
        action5 = [1] + state_object + list(self.RightHandGraspOrientation) + grasp

        spinOrn = p.getQuaternionFromEuler([0., math.pi, -math.pi * 0.5])
        action6 = [1] + state_object + list(spinOrn) + grasp

        state_object[0] = state_object[0] + 0.2
        state_object[1] = state_object[1] - 0.4
        action7 = [1] + state_object + list(spinOrn) + grasp

        grasp = [0.04]
        action8 = [1] + state_object + list(spinOrn) + grasp

        actions = [initial_action, action, action2, action3, action4, action5, action6, action7, action8]

        return actions

    def knot(self, state_object):
        initial_action = [1] + self.WorkPos

        grasp = [0.04]
        action = [1] + [0.6, -0.26, 0.95] + list(self.RightHandGraspOrientation) + grasp

        # state_object[2] = state_object[2] - 0.19
        action2 = [1] + [0.6, -0.26, 0.75] + list(self.RightHandGraspOrientation) + grasp
        #
        grasp = [0.008]
        action3 = [1] + [0.6, -0.26, 0.74] + list(self.RightHandGraspOrientation) + grasp
        #
        # state_object[2] = state_object[2] + 0.2
        action4 = [1] + [0.6, -0.26, 0.8] + list(self.RightHandGraspOrientation) + grasp

        action5 = [1] + [0.6, -0.5, 0.8] + list(self.RightHandGraspOrientation) + grasp
        #
        spinOrn = p.getQuaternionFromEuler([0., math.pi, -math.pi])
        action6 = [1] + [0.5, -0.4, 0.8] + list(spinOrn) + grasp

        action7 = [1] + [0.45, 0.2, 0.8] + list(spinOrn) + grasp
        #
        # spinOrn = p.getQuaternionFromEuler([0., math.pi, -math.pi * 0.5])

        action8 = [1] + [0.6, 0.2, 0.8] + list(spinOrn) + grasp

        grasp = [0.04]
        action9 = [1] + [0.66, 0.2, 0.8] + list(spinOrn) + grasp
        #
        # state_object[0] = state_object[0] + 0.2
        # state_object[1] = state_object[1] - 0.4
        # action7 = [1] + state_object + list(spinOrn) + grasp
        #
        # grasp = [0.04]
        # action8 = [1] + state_object + list(spinOrn) + grasp

        actions = [initial_action, action, action2, action3, action4, action5, action6, action7, action8, action9]

        return actions

    def testOrn(self, pos):
        grabOrn = p.getQuaternionFromEuler([0, math.pi * 0.5, -math.pi * 0.5])
        grasp = [0.04]
        action = [2] + [0.7, 0.4, 0.9] + list(grabOrn) + grasp
        actions = [action]
        return actions

    def knot2(self, state_object):
        initial_action = [1] + self.WorkPos

        graspOrn = p.getQuaternionFromEuler([0., -math.pi, math.pi*0.5])
        spinOrn = p.getQuaternionFromEuler([0., math.pi, -math.pi*0.75])
        spin2Orn = p.getQuaternionFromEuler([0., math.pi, -math.pi*0.5])
        grabOrn = p.getQuaternionFromEuler([0, math.pi * 0.5, -math.pi * 0.5])

        grasp = [0.04]
        action = [1] + [0.5, -0.4, 0.95] + list(graspOrn) + grasp

        # state_object[2] = state_object[2] - 0.19
        action2 = [1] + [0.5, -0.4, 0.74] + list(graspOrn) + grasp
        #
        grasp = [0.004]
        action3 = [1] + [0.5, -0.4, 0.74] + list(graspOrn) + grasp
        #

        action4 = [1] + [0.5, -0.4, 0.9] + list(graspOrn) + grasp
        #
        action5 = [1] + [0.6, -0.35, 0.9] + list(spinOrn) + grasp
        #
        action6 = [1] + [0.7, 0.2, 0.9] + list(spin2Orn) + grasp

        grasp = [0.04]
        # action7 = [1] + [0.7, 0.2, 0.9] + list(spin2Orn) + grasp
        #
        action7 = [2] + [0.7, 0.5, 0.92] + list(grabOrn) + grasp
        action8 = [2] + [0.7, 0.275, 0.92] + list(grabOrn) + grasp

        grasp = [0.004]
        action9 = [2] + [0.7, 0.275, 0.92] + list(grabOrn) + grasp

        grasp = [0.04]
        action10 = [1] + [0.7, 0.2, 0.9] + list(spin2Orn) + grasp

        grasp = [0.004]
        action11 = [2] + [0.7, 0.7, 0.92] + list(grabOrn) + grasp

        actions = [initial_action, action, action2, action3, action4, action5, action6, action7, action8, action9, action10, action11]



        return actions


    def spawn_rope_100(self, state_object):
        # Soft body parameters
        mass = 0.007
        scale = 0.018
        # scale = 0.035
        softBodyId = 0
        useBend = True
        ESt = 0.19
        DSt = 0.0625
        BSt = 0.05
        Rp = 0.01
        cMargin = 0.00475
        friction = 1e99

        self.softBodyId = p.loadSoftBody('cyl_100_1568.vtk', mass=mass, scale=scale, basePosition=state_object,
                                    baseOrientation=p.getQuaternionFromEuler([0, math.pi / 2, -math.pi/2]),
                                    useNeoHookean=0, useBendingSprings=useBend, useMassSpring=1,
                                    springElasticStiffness=ESt,
                                    springDampingStiffness=DSt, springBendingStiffness=BSt, repulsionStiffness=Rp,
                                    useSelfCollision=0,
                                    collisionMargin=cMargin, frictionCoeff=friction, useFaceContact=0)

    def spawn_rope_50(self, state_object):
        # Soft body parameters
        mass = 0.01
        scale = 0.023
        # scale = 0.035
        softBodyId = 0
        useBend = True
        ESt = 0.15
        DSt = 0.05
        BSt = 0.05
        Rp = 0.01
        cMargin = 0.01
        friction = 1e99
        self.softBodyId = p.loadSoftBody('cyl_50_827.vtk', mass=mass, scale=scale, basePosition=state_object,
                                    baseOrientation=p.getQuaternionFromEuler([0, math.pi / 2, -math.pi/2]),
                                    useNeoHookean=0, useBendingSprings=useBend, useMassSpring=1,
                                    springElasticStiffness=ESt,
                                    springDampingStiffness=DSt, springBendingStiffness=BSt, repulsionStiffness=Rp,
                                    useSelfCollision=0,
                                    collisionMargin=cMargin, frictionCoeff=friction, useFaceContact=0)


    def spawn_deformable_object(self, state_object):
        self.cylindId = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.0135, height=0.2)
        #self.cylindId = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.00535, height=0.1)
        self.cylindId = p.createMultiBody(0.001, self.cylindId, -1, state_object, self.ObjStartOrientation)
        p.changeDynamics(self.cylindId, -1, rollingFriction=0.05)

    def spawn_ring(self, state_object, orn):
        self.torusId = p.createCollisionShape(p.GEOM_MESH, fileName="torus/torus_15_8.obj",
                                              meshScale=[0.00225, 0.00225, 0.00225],
                                              flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        self.torusId = p.createMultiBody(0, self.torusId, basePosition=state_object, baseOrientation=orn)


    # def random_perturbation(self, object):
    #     #Object = Action Space
    #     #Object = [[S1], [S2], ...., [Sn]], Every S has Position and Orientation of the segment
    #     action_space = object
    #     target_segment = action_space[random.uniform(0, len(action_space))]
    #
    #     target_pos, target_orn = target_segment[0], target_segment[1]
    #
    #     #Calculate respective approaching Orientation
    #
    #     grasp_action = [1] + target_pos + target_orn + [self.CloseGrasp]
    #     self.applyAction(grasp_action)
    #
    #     #Now we move the string randomly
    #     new_pos[0] = target_pos[0] + random.uniform(-0.1, 0.1)
    #     new_pos[1] = target_pos[1] + random.uniform(-0.1, 0.1)
    #     new_pos[2] = target_pos[2] + random.uniform(-0.1, 0.1)
    #
    #     move_action = [1] + new_pos + target_orn + [self.CloseGrasp]
    #     self.applyAction(move_action)
    #
    #     #Realse Fingers
    #
    #     #Get new states of objects
    #     state_object = ...
    #
    #     return state_object

    def close(self):
        p.disconnect()




