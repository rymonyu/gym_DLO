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


StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
p.connect(p.GUI)




def reset_env():
    RightHandGraspOrientation = p.getQuaternionFromEuler([0., -math.pi * 0.75, -math.pi * 0.5])
    RightHandInitialPos = [0.6, -0.2, 0.8]
    LeftHandInitialPos = [0.5, 0.5, 1.0, 0.6532814824381882, 0.6532814824381883, -0.2705980500730985,
                          0.27059805007309856, 0.04]

    uls = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
    lls = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
    jrs = [5.9342, 3.6652, 5.9342, 3.1416, 5.9342, 3.9095999999999997, 5.9342]
    rps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    p.resetDebugVisualizerCamera(cameraDistance=0.75, cameraYaw=90, cameraPitch=-89,
                                 cameraTargetPosition=[0.7, 0.25, 0.8])

    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    # p.resetSimulation()
    p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.025)
    p.setRealTimeSimulation(1)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdfRootPath = pybullet_data.getDataPath()
    texture_paths = glob.glob(os.path.join('dtd', '**', '*.jpg'), recursive=True)

    p.setGravity(0, 0, -10)

    # Load plane
    planeId = p.loadURDF("plane.urdf")

    # Generate table with random spawn texture
    tableId = p.loadURDF("table/table.urdf", basePosition=[0.6, 0.25, 0.1], baseOrientation=StartOrientation)
    # random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]
    # textureId = p.loadTexture(random_texture_path)
    # p.changeVisualShape(tableId, -1, textureUniqueId=textureId)

    # pandaId = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), basePosition=[0, 0, 0.7],
    #                      baseOrientation=StartOrientation, useFixedBase=True)
    #
    # panda2Id = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), basePosition=[0, 0.5, 0.7],
    #                            baseOrientation=StartOrientation, useFixedBase=True)

    # YumiId = p.loadURDF("yumi_description/urdf/yumi.urdf", basePosition=[0, 0, 0.7],
    #                      baseOrientation=StartOrientation, useFixedBase=True)

    # jointPoses = p.calculateInverseKinematics(pandaId, 11, RightHandInitialPos, RightHandGraspOrientation, lls, uls, jrs, rps)
    #
    # # Set joint motor movement one by one / group
    # for i in range(7):
    #     p.setJointMotorControl2(pandaId,
    #                             jointIndex=i,
    #                             controlMode=p.POSITION_CONTROL,
    #                             targetPosition=jointPoses[i])

    # goal2 = LeftHandInitialPos[0:3]
    # orientation2 = LeftHandInitialPos[3:7]
    # jointPoses2 = p.calculateInverseKinematics(panda2Id, 11, goal2, orientation2)
    #
    # # Set joint motor movement one by one / group
    # for i in range(7):
    #     p.setJointMotorControl2(panda2Id,
    #                             jointIndex=i,
    #                             controlMode=p.POSITION_CONTROL,
    #                             targetPosition=jointPoses2[i],
    #                             positionGain=0.08,
    #                             maxVelocity=5)
    ring_spawn = [0.7, 0.35, 0.73]
    torusId = p.createCollisionShape(p.GEOM_MESH, fileName="torus/torus_15_6.obj",
                                          meshScale=[0.00225, 0.00225, 0.00225],
                                          flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    torusId = p.createMultiBody(0, torusId, basePosition=ring_spawn, baseOrientation=p.getQuaternionFromEuler([math.pi*0.5, 0, 0]))

    ring_spawn = [0.7, 0.45, 0.73]
    torus2Id = p.createCollisionShape(p.GEOM_MESH, fileName="torus/torus_15_7.obj",
                                     meshScale=[0.00225, 0.00225, 0.00225],
                                     flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    torus2Id = p.createMultiBody(0, torus2Id, basePosition=ring_spawn,
                                baseOrientation=p.getQuaternionFromEuler([math.pi * 0.5, 0, 0]))

    ring_spawn = [0.7, 0.55, 0.73]
    torus3Id = p.createCollisionShape(p.GEOM_MESH, fileName="torus/torus_15_8.obj",
                                     meshScale=[0.00225, 0.00225, 0.00225],
                                     flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    torus3Id = p.createMultiBody(0, torus3Id, basePosition=ring_spawn,
                                baseOrientation=p.getQuaternionFromEuler([math.pi * 0.5, 0, 0]))


    #
    cylindId = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.0135, height=0.2)
    cylindId = p.createMultiBody(0.001, cylindId, -1, [0.7, 0.25, 0.8], p.getQuaternionFromEuler([0, math.pi, 0 ]))
    p.changeDynamics(cylindId, -1, spinningFriction = 50, rollingFriction=50)



pandaId = reset_env()

xId = p.addUserDebugParameter('X', -math.pi, math.pi, 0)
yId = p.addUserDebugParameter('Y', -math.pi, math.pi, 0)
zId = p.addUserDebugParameter('Z', -math.pi, math.pi, 0)

def spawn(pandaId):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdfRootPath = pybullet_data.getDataPath()

    p.removeBody(pandaId)

    x = p.readUserDebugParameter(xId)
    y = p.readUserDebugParameter(yId)
    z = p.readUserDebugParameter(zId)

    orientation = p.getQuaternionFromEuler([x, y, z])

    pandaId = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), basePosition=[0, 0, 0.7],
                         baseOrientation=StartOrientation, useFixedBase=True)

    jointPoses = p.calculateInverseKinematics(pandaId, 11, [1.0, 0.0, 1.5], orientation)

    # Set joint motor movement one by one / group
    for i in range(7):
        p.setJointMotorControl2(pandaId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i])

    time.sleep(1. / 240.)

    return pandaId

q_Key = ord('q')  # Press q to generate softbody
r_Key = ord('r')  # Press r to reset
m_Key = ord('m')

while p.isConnected():
    keys = p.getKeyboardEvents()
    if r_Key in keys and keys[r_Key] & p.KEY_IS_DOWN:

        reset_env()

p.disconnect()
