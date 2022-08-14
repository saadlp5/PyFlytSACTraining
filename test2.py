import gym
import numpy as np
import pettingzoo

from CustomEnv import parallel_env

env = parallel_env(num_drones=3)

env.render()
env.reset()
done = False

while not done:
    actions = {i: env.action_space(i).sample() for i in env.agents}
    observations,rewards,dones,_= env.step(actions)
    print(observations[0],"blegh")
