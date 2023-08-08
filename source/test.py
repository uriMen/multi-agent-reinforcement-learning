import time

from pettingzoo.mpe import simple_adversary_v3

max_cycles = 50
env = simple_adversary_v3.env(max_cycles=max_cycles, render_mode='human')
# for k, v in env.action_spaces.items():
    # print(v.n)

# print(next(iter(env.action_spaces.values())).n)
env.reset()
# print(env.observation_spaces)
# for a in env.agents:
#     print(env.observation_space(a)) #[i].shape)

# print((not any(env.terminations.values()) or not any(env.truncations.values())))
#     print(env.action_space(env.agents[1]).sample())
# print(env.observe(env.agents[1]))
# print(type(env.action_space(env.agents[1]).sample()))
step = 0

# while not any(env.terminations.values()) or not any(env.truncations.values()):
#     print(step, env.terminations.values(), env.truncations.values())
# for step in range(max_cycles):

# for i, agent in enumerate(env.agent_iter()):
#     print(i, agent)

# print(env.)
obs = {a: [] for a in env.agents}
actions = {a: [] for a in env.agents}
next_obs = {a: [] for a in env.agents}
rewards = {a: [] for a in env.agents}
dones = {a: [] for a in env.agents}
for i, agent in enumerate(env.agent_iter()):
    observation, reward, termination, truncation, info = env.last()
    # print(int(i / 3), i, agent)
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy
    obs[agent].append(observation)
    actions[agent].append(action)
    rewards[agent].append(reward)
    dones[agent].append(termination)
    env.step(action)
# print(len(obs), len(actions), len(rewards), len(dones))
print([len(l) for l in obs.values()])
    # step = int(i / 3)
    # time.sleep(1)
#     # env.render()
#     print(step)
#     # env.render()
#     env.close()
# time.sleep(10)