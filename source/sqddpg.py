import numpy as np
import torch as T
import torch.nn.functional as F

from agent import Agent


class SQDDPG:
    def __init__(self, actor_dims, critic_dims, agent_names, n_actions,
                 scenario='simple', alpha=0.01, beta=0.01, fc1=64,
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/', num_coalition=24):
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
        self.num_coalition = num_coalition

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

    def sample_coalition(self, i):
        """sample a subset of agents which doesn't include the i-th agent"""
        c = np.random.randint(2, size=self.n_agents, dtype=int)
        c[i] = 0
        return c

    def mask_actions(self, actions, coalition):
        """mask actions of agents which aren't in the coalition"""
        mask = T.tensor(np.repeat(coalition, self.n_actions))
        return actions * mask

    def calc_phi(self, agent_idx, agent, coalition, actions, states, hat=False):
        """calculate aprox marginal contribution of agent"""
        masked_actions = self.mask_actions(actions, coalition)
        coalition[agent_idx] = 1
        masked_actions_i = self.mask_actions(actions, coalition)
        if hat:
            critic_value = self.agents[agent].target_critic.forward(states,
                                                                    masked_actions)
            critic_value_i = self.agents[agent].target_critic.forward(states,
                                                                      masked_actions_i)
        else:
            critic_value = self.agents[agent].critic.forward(states,
                                                             masked_actions)
            critic_value_i = self.agents[agent].critic.forward(states,
                                                               masked_actions_i)

        phi = (critic_value_i - critic_value)
        return phi





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

        all_agents_new_actions = []  # a_k_hat
        all_agents_new_mu_actions = []  # a_k
        old_agents_actions = []  # u_k

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

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)  # a_hat
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)  # a
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)  # u

        all_agents_Q_hat = []
        all_agents_Q_u = []

        for agent_idx, agent in enumerate(self.agents.keys()):
            """
            1. sample M ordered coalitions C with probability P(C|N-{agent_idx})
            2. for each coalition:
                3. mask a_k_hat and compute phi over this mask
                4. mask a_k and compute phi over this mask
                5. mask u_k and compute phi over this mask
            6. compute Q_a, Q_a_hat, Q_u (average over all M results for each of the 3 values above)
            7. compute actor_loss with Q_a
            8. compute target: y = r + gamma * sum(Q_a_hat)
            9. compute Q MSELoss with y and sum(Q_u)
            10. update networks parameters
            """
            Q_a = []
            Q_u = []
            Q_a_hat = []
            for m in range(self.num_coalition):
                coalition = self.sample_coalition(agent_idx)
                phi_a = self.calc_phi(agent_idx, agent, coalition, mu, states)
                phi_u = self.calc_phi(agent_idx, agent, coalition, old_actions, states)
                phi_a_hat = self.calc_phi(agent_idx, agent, coalition, new_actions, states, True)

                Q_a.append(phi_a)
                Q_u.append(phi_u)
                Q_a_hat.append(phi_a_hat)

            Q_a = T.mean(T.cat(Q_a, dim=1), dim=1)
            Q_u = T.mean(T.cat(Q_u, dim=1), dim=1)
            Q_a_hat = T.mean(T.cat(Q_a_hat, dim=1), dim=1)

            all_agents_Q_u.append(Q_u.reshape(-1,1))
            all_agents_Q_hat.append(Q_a_hat.reshape(-1,1))

            actor_loss = -T.mean(Q_a)
            self.agents[agent].actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True,
                                inputs=list(
                                    self.agents[agent].actor.parameters()))
            self.agents[agent].actor.optimizer.step()

        all_Q_hat = T.cat(all_agents_Q_hat, dim=1)
        sum_Q_hat = T.sum(all_Q_hat, 1)
        all_Q_u = T.cat(all_agents_Q_u, dim=1)
        sum_Q_u = T.sum(all_Q_u, 1)

        for agent_idx, agent in enumerate(self.agents.keys()):
            sum_Q_hat[dones[:, 0]] = 0.0
            target = rewards[:, agent_idx] + self.agents[agent].gamma * sum_Q_hat
            critic_loss = F.mse_loss((1 / np.sqrt(2)) * sum_Q_u, (1 / np.sqrt(2)) * target)
            self.agents[agent].critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True,
                                 inputs=list(
                                     self.agents[agent].critic.parameters()))
            self.agents[agent].critic.optimizer.step()

            self.agents[agent].update_network_parameters()


