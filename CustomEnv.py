import functools
import math
import os
from typing import Dict

import gym
import numpy as np
import pybullet as p
from gym import spaces
from gym.spaces import Box
from gym.spaces import Graph, GraphInstance
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from PyFlyt.core.aviary import Aviary


def env():
    env = raw_env()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env():
    env = parallel_env()
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "customdroneenvironment"}

    def __init__(
        self,
        max_steps=1000,
        num_drones=5,
        angle_representation="quaternion",
        num_targets=1,
        flight_dome_size=5.0,
        goal_reach_distance=0.2,
    ):
        # Making a list of drone names numbered from 1 to num_drones
        self.num_drones = num_drones
        self.agents = [i for i in range(self.num_drones)]
        self.possible_agents = [i for i in range(self.num_drones)]

        """ENVIRONMENT CONSTANTS"""
        # environment general settings
        self.enable_render = False
        self.max_steps = max_steps
        self.num_targets = num_targets
        self.goal_reach_distance = goal_reach_distance
        self.flight_dome_size = flight_dome_size
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(
            file_dir,
            "../models/target.urdf",
        )

        # current angle representation
        self.angle_rep = 0 if angle_representation == "euler" else None
        self.angle_rep = 1 if angle_representation == "quaternion" else None
        assert (
            self.angle_rep is not None
        ), f"Something went wrong with setting angle reps, got {angle_representation}"

        # observation size increases by 1 for quaternion
        # self.obs_shape = self.num_drones*3 + 3 if self.angle_rep == 0 else None
        # self.obs_shape = self.num_drones*3 + 3 if self.angle_rep == 1 else None

        # observation space dict
        # space = Box(
        # low=-np.inf,
        # high=np.inf,
        # shape=(self.obs_shape,),
        # dtype=np.float64,
        # )
        # Observation space graph dict
        self.obs_shape = (self.num_drones - 1) * 3
        space = Graph(
            node_space=Box(
                low=-np.inf,
                high=np.inf,
                shape=(6,),
                dtype=np.float64,
            ),
            edge_space=Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        )

        self.observation_spaces = {agent: space for agent in self.agents}

        # action space dict
        space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64)
        self.action_spaces = {agent: space for agent in self.agents}

        """RUNTIME VARIABLES"""
        self.env = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self, mode="human"):
        self.enable_render = True

    def reset(self, seed=None):
        # if we already have an env, disconnect from it
        if self.env is not None:
            self.env.disconnect()

        # reset step count
        self.cumulative_rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in range(self.num_drones)}
        self.step_count = 0

        # init env
        self.env = Aviary(
            start_pos=np.array(
                [
                    range(self.num_drones),
                    [0] * self.num_drones,
                    [1] * self.num_drones,
                ]
            ).T,
            start_orn=np.zeros((self.num_drones, 3)),
            render=self.enable_render,
        )

        # set flight mode
        self.env.set_mode(5)

        # we sample from polar coordinates to generate linear targets
        #self.targets = np.zeros(shape=(self.num_targets, 3))
        #thts = np.random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        #phis = np.random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        #for i, tht, phi in zip(range(self.num_targets), thts, phis):
           # dist = np.random.uniform(low=1.0, high=self.flight_dome_size)
            #x = dist * math.sin(phi) * math.cos(tht)
            #y = dist * math.sin(phi) * math.sin(tht)
            #z = abs(dist * math.cos(phi))
            #self.targets[i] = np.array([x, y, z])
        self.targets = np.array([[-3.5,0.1,1.54]])

        # wait for env to stabilize
        for _ in range(10):

            self.env.step()

            # if we are rendering, load in the targets
            if self.enable_render:
                self.target_visual = []
                for target in self.targets:
                    self.target_visual.append(
                        self.env.loadURDF(
                            self.targ_obj_dir, basePosition=target, useFixedBase=True
                        )
                    )

                for i, visual in enumerate(self.target_visual):
                    p.changeVisualShape(
                        visual,
                        linkIndex=-1,
                        rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                    )

        state = self.compute_observations()
        self.prev_error = self.dis_error_scalar

        return state

    def compute_observations(self):
        """This computes the observation for each agent"""
        self.states = {}
        self.dis_error_scalar = {}
        lin_pos_placeholder = np.zeros(shape=(self.num_drones, 3))
        for k in self.agents:
            raw_state = self.env.states[k]
            lin_pos_placeholder[k] = raw_state[3]

        for i in self.agents:
            raw_state = self.env.states[i]

            # state breakdown
            ang_vel = raw_state[0]
            ang_pos = raw_state[1]
            lin_vel = raw_state[2]
            lin_pos = raw_state[3]
            lin_pos_subtract = lin_pos_placeholder - lin_pos
            lin_pos_subtract = np.delete(lin_pos_subtract, i, 0)
            lin_pos_subtract = np.ndarray.flatten(lin_pos_subtract)

            # quarternion angles
            q_ang_pos = p.getQuaternionFromEuler(ang_pos)

            # rotation matrix
            rotation = np.array(p.getMatrixFromQuaternion(q_ang_pos)).reshape(3, 3).T

            # drone to target
            dis_error = np.matmul(rotation, self.targets[0] - lin_pos)

            # precompute the scalars so we can use it later
            self.dis_error_scalar[i] = np.linalg.norm(dis_error)

            # state has no ang_pos, ang_vel, lin_pos
            nodes = np.array([[*lin_vel, *dis_error]])
            edges = np.reshape(lin_pos_subtract, (self.num_drones - 1, 3))
            edge_links = None
            new_state = GraphInstance(nodes, edges, edge_links)

            self.states[i] = new_state

        return self.states

    def target_reached(self, agent: int):
        if self.dis_error_scalar[agent] < self.goal_reach_distance:
            return True

    def compute_rewards(self):
        """This computes the reward for each agent"""
        rewards = {}
        for i in self.agents:
            if len((self.env.getContactPoints(i + 1))) > 0:
                # collision
                rewards[i] = -100
            elif self.target_reached(i):
                # reached target
                rewards[i] = 100
            else:
                # normal reward
                rewards[i] = np.clip(
                    (self.prev_error[i] - self.dis_error_scalar[i]) * 50.0, -1.0, 1.0
                )
                self.prev_error[i] = self.dis_error_scalar[i]

        return rewards

    def compute_dones(self):
        """This computes the done for each agent"""

        done = {i: False for i in self.agents}
        vardone = [False for h in self.agents]

        for i in self.agents:
            # we were already done
            if self.dones[i]:
                done[i] = True

            if self.step_count > self.max_steps:
                done[i] = True

            if len(self.env.getContactPoints(i + 1)) > 0:
                done[i] = True

            if self.target_reached(i):
                # done[i] = True
                vardone[i] = True
                if all(s == True for s in vardone):
                    for b in self.agents:
                        done[b] = True

            # out of bounds done
            if np.linalg.norm(self.dis_error_scalar[i]) > 2 * self.flight_dome_size:

                done[i] = True

        self.dones = done
        return self.dones

    def step(self, actions: Dict[int, np.ndarray]):
        """Step the entire simulation
        Inputs:
            actions: a dictionary of agent: actions
        Outputs:
            observations: a dictionary of agent: observation
            rewards: a dictionary of agent: reward
            dones: a dictionary of agent: dones
        """
        yaws = np.array([self.env.states[i][1, -1] for i in actions])
        action = np.array(
            [[actions[i][0], actions[i][1], -yaws[i], actions[i][2]] for i in actions]
        )

        # step through env
        self.env.set_setpoints(action)
        self.env.step()

        # step more such that we don't take an action every timestep
        for _ in range(4):
            while self.env.drones[0].steps % self.env.drones[0].update_ratio != 0:
                self.env.step()

        # compute state and done
        observations = self.compute_observations()
        rewards = self.compute_rewards()
        dones = self.compute_dones()

        self.step_count += 1

        for i in self.agents:
            self.cumulative_rewards[i] += rewards[i]

        return observations, rewards, dones, {}
