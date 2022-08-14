import os
from signal import SIGINT, signal

import numpy as np
import torch
import torch.optim as optim
import wandb
from PIL import Image

from CustomEnv import parallel_env
from SAC.SAC import SAC
from shebangs import check_venv, parse_set, shutdown_handler
from utils.helpers import Helpers, cpuize, gpuize
from utils.replay_buffer import ReplayBuffer


def train(set):
    env = setup_env(set)
    net, net_helper, optim_set, optim_helper = setup_nets(set)
    memory = ReplayBuffer(set.buffer_size)

    to_log = dict()
    to_log["epoch"] = 0
    to_log["episodic_reward"] = -np.inf
    to_log["max_episodic_reward"] = -np.inf

    while memory.count <= set.total_steps + 1:
        to_log["epoch"] += 1

        """ENVIRONMENT INTERACTION"""
        env.reset()
        net.eval()
        net.zero_grad()

        with torch.no_grad():
            dne = [False]
            while not any(dne):
                # get observation and vectorize
                obs = env.states
                nodes = np.array([obs[agent].nodes for agent in env.possible_agents])
                edges = np.array([obs[agent].edges for agent in env.possible_agents])

                # if we're exploring, just sample from the space
                if memory.count < set.exploration_steps:
                    action = np.array(
                        [
                            env.action_space(agent).sample()
                            for agent in env.possible_agents
                        ]
                    )
                else:
                    output = net.actor(
                        gpuize(nodes, set.device), gpuize(edges, set.device)
                    )
                    action, _ = net.actor.sample(*output)
                    action = cpuize(action)
                    action_dict = {i: action[i] for i in range(len(action))}

                # get the next state and other stuff
                next_obs, rew, dne, _ = env.step(action_dict)

                # vectorize everything
                next_nodes = np.array(
                    [next_obs[agent].nodes for agent in env.possible_agents]
                )
                next_edges = np.array(
                    [next_obs[agent].edges for agent in env.possible_agents]
                )
                rew = np.array([rew[agent] for agent in env.possible_agents])
                dne = np.array([dne[agent] for agent in env.possible_agents])

                # store stuff in mem
                memory.push(
                    (nodes, edges, action, rew, next_nodes, next_edges, dne), bulk=True
                )

            # record the cumulative reward for this run
            to_log["episodic_reward"] = np.mean(
                np.array(
                    [env.cumulative_rewards[agent] for agent in env.possible_agents]
                )
            )
            to_log["max_episodic_reward"] = max(
                to_log["episodic_reward"], to_log["max_episodic_reward"]
            )

        """TRAINING RUN"""
        dataloader = torch.utils.data.DataLoader(
            memory, batch_size=set.batch_size, shuffle=True, drop_last=False
        )

        for repeat_num in range(set.repeats_per_buffer):
            for batch_num, stuff in enumerate(dataloader):
                net.train()

                nodes = gpuize(stuff[0], set.device)
                edges = gpuize(stuff[1], set.device)
                actions = gpuize(stuff[2], set.device)
                rewards = gpuize(stuff[3], set.device)
                next_nodes = gpuize(stuff[4], set.device)
                next_edges = gpuize(stuff[5], set.device)
                dones = gpuize(stuff[6], set.device)

                # train critic
                for _ in range(set.critic_update_multiplier):
                    net.zero_grad()
                    q_loss, log = net.calc_critic_loss(
                        nodes, edges, actions, rewards, next_nodes, next_edges, dones
                    )
                    to_log = {**to_log, **log}
                    q_loss.backward()
                    optim_set["critic"].step()
                    net.update_q_target()

                # train actor
                for _ in range(set.actor_update_multiplier):
                    net.zero_grad()
                    rnf_loss, log = net.calc_actor_loss(nodes, edges, dones)
                    to_log = {**to_log, **log}
                    rnf_loss.backward()
                    optim_set["actor"].step()

                    # train entropy regularizer
                    if net.use_entropy:
                        net.zero_grad()
                        ent_loss, log = net.calc_alpha_loss(nodes, edges)
                        to_log = {**to_log, **log}
                        ent_loss.backward()
                        optim_set["alpha"].step()

                """WEIGHTS SAVING"""
                net_weights = net_helper.training_checkpoint(
                    loss=-to_log["episodic_reward"], batch=0, epoch=to_log["epoch"]
                )
                net_optim_weights = optim_helper.training_checkpoint(
                    loss=-to_log["episodic_reward"], batch=0, epoch=to_log["epoch"]
                )
                if net_weights != -1:
                    torch.save(net.state_dict(), net_weights)
                if net_optim_weights != -1:
                    optim_dict = dict()
                    for key in optim_set:
                        optim_dict[key] = optim_set[key].state_dict()
                    torch.save(
                        {
                            "optim": optim_dict,
                            "lowest_running_loss": optim_helper.lowest_running_loss,
                            "epoch": to_log["epoch"],
                        },
                        net_optim_weights,
                    )

                """WANDB"""
                if set.wandb and repeat_num == 0 and batch_num == 0:
                    to_log["num_transitions"] = memory.count
                    to_log["buffer_size"] = memory.__len__()
                    wandb.log(to_log)


def display(set):
    env = setup_env(set)
    net, _, _, _ = setup_nets(set)

    env.reset()

    done = False
    while True:
        if done:
            env.reset()

        # get observation and vectorize
        obs = env.states
        nodes = np.array([obs[agent].nodes for agent in env.possible_agents])
        edges = np.array([obs[agent].edges for agent in env.possible_agents])

        # infer from the network
        output = net.actor(gpuize(nodes, set.device), gpuize(edges, set.device))
        action = net.actor.infer(*output)
        action = cpuize(action)

        # dictionarize
        action = {i: action[i] for i in range(len(action))}

        # get the next state and other stuff
        obs, rew, dne, _ = env.step(action)

        # compute dones
        done = any([dne[i] for i in dne])


def evaluate(set):
    env = setup_env(set)
    net, _, _, _ = setup_nets(set)

    raise NotImplementedError


def setup_env(set):
    env = parallel_env(num_drones=3,goal_reach_distance=0.3)
    if set.display:
        env.render()
    set.num_actions = env.action_space(0).shape[0]
    set.node_space = env.observation_space(0).node_space.shape[0]
    set.edge_space = env.observation_space(0).edge_space.shape[0]

    return env


def setup_nets(set):
    net_helper = Helpers(
        mark_number=set.net_number,
        version_number=set.net_version,
        weights_location=set.weights_directory,
        epoch_interval=set.epoch_interval,
        batch_interval=set.batch_interval,
    )
    optim_helper = Helpers(
        mark_number=0,
        version_number=set.net_version,
        weights_location=set.optim_weights_directory,
        epoch_interval=set.epoch_interval,
        batch_interval=set.batch_interval,
        increment=False,
    )

    # set up networks and optimizers
    net = SAC(
        num_actions=set.num_actions,
        node_space=set.node_space,
        edge_space=set.edge_space,
        entropy_tuning=set.use_entropy,
        target_entropy=set.target_entropy,
        discount_factor=set.discount_factor,
    ).to(set.device)
    actor_optim = optim.AdamW(
        net.actor.parameters(), lr=set.learning_rate, amsgrad=True
    )
    critic_optim = optim.AdamW(
        net.critic.parameters(), lr=set.learning_rate, amsgrad=True
    )
    alpha_optim = optim.AdamW([net.log_alpha], lr=0.01, amsgrad=True)

    optim_set = dict()
    optim_set["actor"] = actor_optim
    optim_set["critic"] = critic_optim
    optim_set["alpha"] = alpha_optim

    # get latest weight files
    net_weights = net_helper.get_weight_file()
    if net_weights != -1:
        net.load_state_dict(torch.load(net_weights))

    # get latest optimizer states
    net_optimizer_weights = optim_helper.get_weight_file()
    if net_optimizer_weights != -1:
        checkpoint = torch.load(net_optimizer_weights)

        for opt_key in optim_set:
            optim_set[opt_key].load_state_dict(checkpoint["optim"][opt_key])

        net_helper.lowest_running_loss = checkpoint["lowest_running_loss"]
        optim_helper.lowest_running_loss = checkpoint["lowest_running_loss"]
        print(f"Lowest Running Loss for Net: {net_helper.lowest_running_loss}")

    return net, net_helper, optim_set, optim_helper


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    set = parse_set()
    check_venv()

    """ SCRIPTS HERE """

    if set.display:
        display(set)
    elif set.train:
        train(set)
    elif set.evaluate:
        evaluate(set)
    else:
        print("Guess this is life now.")

    """ SCRIPTS END """

    if set.shutdown:
        os.system("poweroff")
