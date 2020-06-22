import pybullet as p
from time import sleep
import pybullet_data
import numpy as np
import math
import os

urdfRootPath = pybullet_data.getDataPath()
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
softBodyPos = [.25, .1, 1]
endEffectorInitialPos = [0.088, -1.7431537324696934e-12, 1.5210000000000001]
endEffectorInitialOrn = [0.9238795325113726, 0.38268343236488267, -1.8738412424764667e-12, 4.523852941323606e-12]
endEffectorIndex = 11
leftGripperIndex = 9
rightGripperIndex = 10
numJoints = 7
pandaId = 0

uls = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
lls = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
jrs = [5.9342, 3.6652, 5.9342, 3.1416, 5.9342, 3.9095999999999997, 5.9342]
rps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
r_Key = ord('r')  # Press r to reset


def set_env():
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.setTimeStep(0.01)
    p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.025)
    p.setRealTimeSimulation(1)
    p.setGravity(0, 0, -10)
    sh_planeId = p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, sh_planeId, basePosition=[0, 0, -0.09])
    pandaId = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), basePosition=[0, 0, 0],
                         baseOrientation=startOrientation, useFixedBase=True)

    '''softBodyId = p.loadSoftBody('cyl_long.vtk', mass = mass, scale = scale, basePosition = softBodyPos, 
                                useNeoHookean = 0, useBendingSprings = useBend, useMassSpring = 1, springElasticStiffness=ESt,
                                springDampingStiffness=DSt, springBendingStiffness=BSt, repulsionStiffness=Rp, useSelfCollision=0, 
                                collisionMargin = cMargin, frictionCoeff=friction, useFaceContact=1)'''
    softBodyId = p.loadSoftBody('cyl_50_827.vtk', mass=mass, scale=scale, basePosition=softBodyPos,
                                baseOrientation=p.getQuaternionFromEuler([0, math.pi / 2, 0]),
                                useNeoHookean=0, useBendingSprings=useBend, useMassSpring=1, springElasticStiffness=ESt,
                                springDampingStiffness=DSt, springBendingStiffness=BSt, repulsionStiffness=Rp,
                                useSelfCollision=0,
                                collisionMargin=cMargin, frictionCoeff=friction, useFaceContact=0)
    # print('Robot state: ', state_robot)
    return pandaId, softBodyId


def reach(robotId, targetId):
    print('Reaching softbody')
    softBodyBasePos, softBodyBaseOrn = p.getBasePositionAndOrientation(targetId)
    softBodyLiftPos = np.array(softBodyBasePos) + np.array([0, 0, 0.4])
    softBodyLiftOrn = p.getQuaternionFromEuler([0, math.pi, 0])
    softBodyBasePos = np.array(softBodyBasePos) - np.array([0, 0, 0.015])
    startPos, startOrn = p.getLinkState(pandaId, endEffectorIndex)[0:2]
    start_jointPoses = p.calculateInverseKinematics(robotId, endEffectorIndex, startPos, startOrn, lls, uls, jrs, rps)
    lift_jointPoses = p.calculateInverseKinematics(robotId, endEffectorIndex, softBodyLiftPos, softBodyLiftOrn,
                                                   lls, uls, jrs, rps)

    for i in range(numJoints):
        p.setJointMotorControl2(pandaId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=lift_jointPoses[i], maxVelocity=1.0
                                )

    p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, 0.04, force=5)
    p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, 0.04, force=5)

    sleep(1.5)

    print('Grasping')
    grab_jointPoses = p.calculateInverseKinematics(robotId, endEffectorIndex, softBodyBasePos, softBodyLiftOrn,
                                                   lls, uls, jrs, rps)
    for i in range(numJoints):
        p.setJointMotorControl2(pandaId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=grab_jointPoses[i],
                                maxVelocity=1.0
                                )

    sleep(1)

    p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, 0.005, force=0.2)
    p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, 0.005, force=0.2)

    sleep(1.5)
    for i in range(numJoints):
        p.setJointMotorControl2(pandaId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=lift_jointPoses[i],
                                maxVelocity=1.0
                                )


p.connect(p.GUI)
pandaId, softBodyId = set_env()
panda_friction = friction
p.changeDynamics(pandaId, 9, lateralFriction=panda_friction, rollingFriction=panda_friction,
                 spinningFriction=panda_friction)
p.changeDynamics(pandaId, 10, lateralFriction=panda_friction, rollingFriction=panda_friction,
                 spinningFriction=panda_friction)
p.changeDynamics(pandaId, 11, lateralFriction=panda_friction, rollingFriction=panda_friction,
                 spinningFriction=panda_friction)
p.resetDebugVisualizerCamera(3.6, 0, -45.2, [1.14, 0.25, -0.21])

while p.isConnected:
    keys = p.getKeyboardEvents()
    if r_Key in keys and keys[r_Key] & p.KEY_IS_DOWN:
        reach(pandaId, softBodyId)
        sleep(0.01)
        continue
    p.stepSimulation()
    sleep(1. / 240.)

p.disconnect()