import numpy as np
import torch
import torch.nn as nn
from einops import reduce
from rlf.envs.pointmass import LinearPointMassEnv
from torch.distributions import Normal
from tqdm import tqdm

num_envs = 128
H = 5
lr = 0.01
n_updates = 100
gamma = 0.99
state_dim = 2
action_dim = 1


env = LinearPointMassEnv(num_envs)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(0.1 * torch.randn(state_dim, 1))
        # self.weight = nn.Parameter(torch.zeros(state_dim, 1))
        self.logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.net = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
        )

    def forward(self, state):
        # mean = state @ self.weight
        mean = self.net(state)
        logstd = self.logstd.expand_as(mean)
        return Normal(mean, logstd.exp())


def calculate_returns(rollout_returns, rollout_rewards):
    for step in reversed(range(H)):
        if step == H - 1:
            rollout_returns[step] = rollout_rewards[step]
        else:
            rollout_returns[step] = (
                rollout_returns[step + 1] * gamma + rollout_rewards[step]
            )


policy = Policy()

opt = torch.optim.SGD(policy.parameters(), lr=lr)

rollout_obs = torch.zeros(H + 1, num_envs, state_dim)
rollout_actions = torch.zeros(H, num_envs, action_dim)
rollout_rewards = torch.zeros(H, num_envs, 1)
rollout_returns = torch.zeros(H, num_envs, 1)

for update_i in range(n_updates):
    obs = env.reset()
    rollout_obs[0].copy_(obs)
    all_ep_rewards = []
    print("")

    with torch.no_grad():
        for i in range(H):
            pi = policy(obs)
            action = pi.sample()
            obs, reward, done, info = env.step(action)

            rollout_obs[i].copy_(obs)
            rollout_actions[i].copy_(action)
            rollout_rewards[i].copy_(reward)

        calculate_returns(rollout_returns, rollout_rewards)
        avg_ep_reward = reduce(rollout_rewards, "h n 1 -> n", "sum").mean()
        avg_dist_to_goal = np.mean([x["ep_dist_to_goal"] for x in info])

        print(avg_ep_reward)
        print(avg_dist_to_goal)
        all_ep_rewards.append(avg_ep_reward)

    opt.zero_grad()

    pi = policy(rollout_obs[:-1])
    log_probs = reduce(pi.log_prob(rollout_actions), "h n a -> h n 1", "sum")
    J = torch.mean(-log_probs * rollout_returns)
    J.backward()

    opt.step()
    print(policy.weight, policy.logstd.exp())
