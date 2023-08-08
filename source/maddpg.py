import torch as T
import torch.nn.functional as F
from agent import Agent


class MADDPG:
    def __init__(self, actor_dims, critic_dims, agent_names, n_actions,
                 scenario='simple', alpha=0.01, beta=0.01, fc1=64,
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = dict()
        self.n_agents = len(agent_names)
        self.n_actions = n_actions
        chkpt_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents[agent_names[agent_idx]] = (Agent(actor_dims[agent_idx],
                                                         critic_dims, n_actions,
                                                         len(agent_names),
                                                         agent_idx,
                                                         alpha=alpha, beta=beta,
                                                         chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents.values():
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs: list) -> list:
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
            actor_new_states, states_, dones = memory.sample_buffer()

        device = next(iter(self.agents.values())).actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents.keys()):
            new_states = T.tensor(actor_new_states[agent_idx],
                                  dtype=T.float).to(device)

            new_pi = self.agents[agent].target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx],
                                 dtype=T.float).to(device)
            pi = self.agents[agent].actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents.keys()):
            critic_value_ = self.agents[agent].target_critic.forward(states_,
                                                                     new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0
            critic_value = self.agents[agent].critic.forward(states, old_actions).flatten()
            target = rewards[:, agent_idx] + self.agents[agent].gamma * critic_value_
            # target = target.float()
            critic_loss = F.mse_loss(critic_value, target)
            self.agents[agent].critic.optimizer.zero_grad()
            # T.autograd.set_detect_anomaly(True)
            critic_loss.backward(retain_graph=True,
                                 inputs=list(self.agents[agent].critic.parameters()))
            self.agents[agent].critic.optimizer.step()

            actor_loss = self.agents[agent].critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            self.agents[agent].actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True,
                                inputs=list(self.agents[agent].actor.parameters()))
            self.agents[agent].actor.optimizer.step()

            self.agents[agent].update_network_parameters()
