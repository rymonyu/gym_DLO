import gym
import gym_panda
from gym import error, spaces, utils
from gym.utils import seeding

import os
import time
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import glob

#env = gym.make('panda-v0')

p.connect(p.GUI)
p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0.85, 0.25, 0.3])


def spawn_rope_50(state_object):
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
    softBodyId = p.loadSoftBody('cyl_50_827.vtk', mass=mass, scale=scale, basePosition=state_object,
                                     baseOrientation=p.getQuaternionFromEuler([0, math.pi / 2, -math.pi / 2]),
                                     useNeoHookean=0, useBendingSprings=useBend, useMassSpring=1,
                                     springElasticStiffness=ESt,
                                     springDampingStiffness=DSt, springBendingStiffness=BSt, repulsionStiffness=Rp,
                                     useSelfCollision=0,
                                     collisionMargin=cMargin, frictionCoeff=friction, useFaceContact=0)

def to_initial_pos(pandaId):
    print("Here")
    RightHandInitialPos = [0.75, -0.1, 1.2, -0.6532814824381882, -0.6532814824381883, -0.2705980500730985, 0.27059805007309856, 0.04]
    uls = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
    lls = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
    jrs = [5.9342, 3.6652, 5.9342, 3.1416, 5.9342, 3.9095999999999997, 5.9342]
    rps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    goal = RightHandInitialPos[0:3]
    orientation = RightHandInitialPos[3:7]
    jointPoses = p.calculateInverseKinematics(pandaId, 11, goal, orientation, lls,
                                              uls, jrs, rps)

    # Set joint motor movement one by one / group
    for i in range(7):
        p.setJointMotorControl2(pandaId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                positionGain=0.08,
                                maxVelocity=5)


def reset():
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    # p.resetSimulation()
    p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.025)
    #p.setRealTimeSimulation(1)
    p.setTimestep = 1./240.

    # Enable rendering after we loaded everything
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdfRootPath = pybullet_data.getDataPath()
    texture_paths = glob.glob(os.path.join('dtd', '**', '*.jpg'), recursive=True)

    p.setGravity(0, 0, -10)

    # Load plane
    planeId = p.loadURDF("plane.urdf")

    # Generate table with random spawn texture
    tableId = p.loadURDF("table/table.urdf", basePosition=[0.6, 0.25, 0.1], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
    # random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]
    # textureId = p.loadTexture(random_texture_path)
    # p.changeVisualShape(tableId, -1, textureUniqueId=textureId)


    # Spawn Deformable Object
    # TO DO - ADDS IN ORIENTATION LATER
    #state_object = (random.uniform(0.55, 0.75), random.uniform(-0.25, 0.55), 0.74)
    state_object = [0.6, -0.1, 0.75]
    #self.spawn_deformable_object(state_object)
    spawn_rope_50(state_object)

    torusId = p.createCollisionShape(p.GEOM_MESH, fileName="torus/torus_only.obj",
                                          meshScale=[0.05, 0.05, 0.05],
                                          flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    torusId = p.createMultiBody(0, torusId, basePosition=[0.6, 0.0, 0.9], baseOrientation=[0, 0, 0])


    # Spawn Robot in initial pose
    pandaId = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), basePosition=[0, 0, 0.7],
                              baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
    panda2Id = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), basePosition=[0, 0.5, 0.7],
                               baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

    #Create constraint to keep the fingers at centre
    c = p.createConstraint(pandaId, 9, pandaId, 10, jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0], parentFramePosition=[0, 0, 0.7], childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=20)

    c2 = p.createConstraint(panda2Id, 9, panda2Id, 10, jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0],
                           parentFramePosition=[0, 0.5, 0.7], childFramePosition=[0, 0, 0])
    p.changeConstraint(c2, gearRatio=-1, erp=0.1, maxForce=20)



    # Restart rendering
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    for _ in range(200):
        to_initial_pos(pandaId)
        p.stepSimulation()

    return pandaId, state_object


def applyAction(pandaId, action):
        uls = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
        lls = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
        jrs = [5.9342, 3.6652, 5.9342, 3.1416, 5.9342, 3.9095999999999997, 5.9342]
        rps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print(action)
        goal = action[0:3]
        orientation = action[3:7]
        grasp = action[7]
        # print("Current Position: ", currentPosition)

        # newPosition = [currentPosition[0] + dx, currentPosition[1] + dy, currentPosition[2] + dz]
        print(goal)
        print(orientation)

        # Use Inverse Kinematics with constraints to calculate goal pose of 7 joints
        jointPoses = p.calculateInverseKinematics(pandaId, 11, goal, orientation, lls,
                                                  uls, jrs, rps, solver=p.IK_DLS)

        # Set joint motor movement one by one / group
        for i in range(7):
            p.setJointMotorControl2(pandaId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    positionGain=0.06,
                                    maxVelocity=4)

        time.sleep(1./240.)

        p.setJointMotorControl2(pandaId, 9, p.POSITION_CONTROL, grasp, maxVelocity=0.25, force=20)
        p.setJointMotorControl2(pandaId, 10, p.POSITION_CONTROL, grasp, maxVelocity=0.25, force=20)




def actions(state_object):
    orn = p.getQuaternionFromEuler([0., -math.pi*0.75, -math.pi*0.5])


    initial = [0.6, -0.2, 1.0] + list(orn) + [0.04]

    state_object[2] = state_object[2] + 0.2
    action1 = state_object + list(orn) + [0.04]

    state_object[2] = state_object[2] - 0.19
    action2 = state_object + list(orn) + [0.04]

    action3 = state_object + list(orn) + [0.0075]

    state_object[2] = state_object[2] + 0.2
    action4 = state_object + list(orn) + [0.0075]

    # state_object[0] = state_object[0] -0.1
    state_object[1] = state_object[1] + 0.1
    action5 = state_object + list(orn) + [0.0075]
    #
    # action6 = state_object + list(orn) + [0.04]
    #
    # orn = p.getQuaternionFromEuler([0., -math.pi*0.75, -math.pi*0.4])
    # action7 = [0.5, -0.25, 0.95] + list(orn) + [0.04]
    #
    # action8 = [0.5, -0.25, 0.75] + list(orn) + [0.04]
    #
    # action9 = [0.5, -0.25, 0.75] + list(orn) + [0.005]
    #
    # action10 = [0.5, -0.25, 0.95] + list(orn) + [0.005]
    #
    # action11 = [0.6, -0.1, 0.95] + list(orn) + [0.005]
    #
    # action12 = [0.6, -0.1, 0.95] + list(orn) + [0.04]
    #
    # orn = p.getQuaternionFromEuler([0., -math.pi*0.75, -math.pi*0.6])
    # action13 = [0.65, -0.25, 0.95] + list(orn) + [0.04]
    #
    # action14 = [0.6, -0.3, 0.75] + list(orn) + [0.04]
    # action15 = [0.6, -0.3, 0.75] + list(orn) + [0.005]
    # action16 = [0.6, -0.3, 0.95] + list(orn) + [0.005]
    #
    # action17 = [0.7, 0.05, 0.95] + list(orn) + [0.005]
    # action18 = [0.7, 0.05, 0.95] + list(orn) + [0.04]
    #
    # orn = p.getQuaternionFromEuler([0., -math.pi * 0.75, -math.pi * 0.5])
    # final = [0.6, -0.2, 1.0] + list(orn) + [0.04]


    actions = [initial, action1, action2, action3, action4, action5]
               #action5, action6, action7, action8, action9, action10, action11, action12, action13,
               #action14, action15, action16, action17, action18, final]

    return actions




r_Key = ord('r')  # Press r to reset
m_Key = ord('m')

while p.isConnected():
    keys = p.getKeyboardEvents()
    if r_Key in keys and keys[r_Key] & p.KEY_IS_DOWN:
        pandaId, state_object = reset()
        state_object = [0.6, -0.2, 0.75]
        actions = actions(state_object)

        for a in actions:
            print("Here")
            for _ in range(200):
                applyAction(pandaId, a)
                p.stepSimulation()
                #p.stepSimulation()
    # print("LOL")

p.disconnect()

