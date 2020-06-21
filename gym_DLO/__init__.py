from gym.envs.registration import register

register(
    id='DLO-v0',
    entry_point='gym_DLO.envs:DLOEnv',
)