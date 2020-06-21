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

env = gym.make('panda-v0')

for i in range(10):
    print("Current Iteration Count: ", i+1)
    seq_state = 0
    state, state_ring = env.reset()
    time.sleep(2.)

    while(seq_state != 3):
        #time.sleep(2.)
        agents, seq_state, action = env.action_planner(state, state_ring)
        for i in range(300):
            #env.render()
            state = env.step(agents, seq_state, action)
# for _ in range(10):
#     _, pos = env.reset()
#     actions = env.knot2(pos)
#     #env.to_initial_pos()
#     for a in actions:
#         print(a)
#         for _ in range(200):
#             _, _, _ = env.step(a)


env.close()