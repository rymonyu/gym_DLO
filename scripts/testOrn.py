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

    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.85, 0.15, 0.3])

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
    random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]
    textureId = p.loadTexture(random_texture_path)
    p.changeVisualShape(tableId, -1, textureUniqueId=textureId)

    pandaId = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), basePosition=[0, 0, 0.7],
                         baseOrientation=StartOrientation, useFixedBase=True)

    return pandaId

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
    if q_Key in keys and keys[q_Key] & p.KEY_IS_DOWN:

        pandaId = spawn(pandaId)
    elif r_Key in keys and keys[r_Key] & p.KEY_IS_DOWN:

        pandaId = reset_env()

p.disconnect()
