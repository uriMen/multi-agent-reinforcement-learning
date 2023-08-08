import argparse
from os.path import join, abspath, exists
from os import makedirs
import time
from datetime import datetime

import numpy as np
import pandas as pd
from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3

from maddpg import MADDPG
from sqddpg import SQDDPG
from buffer import MultiAgentReplayBuffer


def obs_list_to_state_vector(observation):
    return np.hstack(observation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main training process')
    parser.add_argument('-a', '--algorithm',
                        help='which algorithm to use',
                        choices=['maddpg', 'sqddpg'], type=str,
                        default='sqddpg')
    parser.add_argument('-s', '--scenario',
                        help='which scenario to use',
                        choices=['simple_adversary', 'simple_spread'], type=str,
                        default='simple_adversary')
    args = parser.parse_args()

    scenario = args.scenario
    alg = args.algorithm
    output_dir = abspath("scores")
    if not exists(output_dir):
        makedirs(output_dir)
    max_cycles = 25
    num_agents = 5  # number of non-adversary agents
    PRINT_INTERVAL = 50  # 500
    N_GAMES = 5000
    MAX_STEPS = 25
    BATCH_SIZE = 1024
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0
    environments = {
        "simple_adversary": simple_adversary_v3.env(
            N=num_agents, max_cycles=max_cycles, render_mode='human'),
        "simple_spread": simple_spread_v3.env(
            N=num_agents, local_ratio=0.5, max_cycles=max_cycles,
            continuous_actions=False)
    }

    env = environments[scenario]
    env.reset()
    agent_names = env.agents
    actor_dims = []
    for k, v in env.observation_spaces.items():
        actor_dims.append(v.shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = next(iter(env.action_spaces.values())).n
    maddpg_agents = MADDPG(actor_dims, critic_dims, agent_names, n_actions,
                        fc1=64, fc2=64,
                        alpha=0.01, beta=0.01, scenario=scenario,
                        chkpt_dir='tmp/maddpg/')

    sqddpg_agents = SQDDPG(actor_dims, critic_dims, agent_names, n_actions,
                           fc1=64, fc2=64,
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                    n_actions, len(agent_names), batch_size=BATCH_SIZE)

    algorithm_type = {"maddpg": maddpg_agents, "sqddpg": sqddpg_agents}
    chosen_alg = algorithm_type[alg]

    if evaluate:
        chosen_alg.load_checkpoint()

    for i in range(N_GAMES):
        score = 0
        env.reset()
        observations = {a: [] for a in env.agents}
        actions = {a: [] for a in env.agents}
        next_obs = {a: [] for a in env.agents}
        rewards = {a: [] for a in env.agents}
        dones = {a: [] for a in env.agents}
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = chosen_alg.agents[agent].choose_action(observation)
                # action = env.action_space(
                #     agent).sample()  # this is where you would insert your policy
            observations[agent].append(observation)
            actions[agent].append(action)
            rewards[agent].append(reward)
            dones[agent].append(termination)
            env.step(action)
            score += reward
            # print(score)
        # time.sleep(0.1) # to slow down the action for the video

        # store transitions
        for j in range(len(next(iter(observations.values()))) - 1):
            _obs = [o[j] for o in observations.values()]
            _next_obs = [o[j+1] for o in observations.values()]
            _actions = [a[j] for a in actions.values()]
            _done = [d[j] for d in dones.values()]
            _reward = [r[j] for r in rewards.values()]
            state = obs_list_to_state_vector(_obs)
            next_state = obs_list_to_state_vector(_next_obs)

            memory.store_transition(_obs, state, _actions, _reward, _next_obs,
                                    next_state, _done)

        if (i + 1) % 10 == 0 and not evaluate:
            chosen_alg.learn(memory)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                # maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            # print(alg, scenario, output_dir)
            print(datetime.now(), 'episode', i, 'average score {:.1f}'.format(avg_score), 'best score {:.1f}'.format(best_score))
            df = pd.DataFrame(score_history)
            df.to_csv(abspath(
                join(output_dir,
                     f"{scenario}_{alg}_{num_agents}_agents_.csv")))

