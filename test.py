import gym
import numpy as np
import pettingzoo

from CustomEnv import parallel_env

env = parallel_env(num_drones=1,num_targets=1)

env.render()
env.reset()
done = False
actions1= {0:np.array([-1.1,0.000001,0.2])}
actions2={0:np.array([0.0,0.0,0.0])}
for i in range(130):
    actions = {i: env.action_space(i).sample() for i in env.agents}
    env.step(actions1)
while not done:
    env.step(actions2)