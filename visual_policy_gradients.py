# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from rl_helper.envs import create_vectorized_envs
from rl_helper.envs.pointmass.pointmass_env import PointMassParams
from torch.distributions import Normal

use_params = PointMassParams(clip_actions=True, radius=1.0)


# %%
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        noise_scale = 0.1
        self.weight = nn.Parameter(noise_scale * torch.randn(2))
        self.logstd = nn.Parameter(torch.zeros(1, 1))

    def forward(self, state):
        mean = state * self.weight
        logstd = self.logstd.expand_as(mean)
        return Normal(mean, logstd.exp())


# %%
def evaluate(num_eval_episodes, policy, envs):
    obs = envs.reset()
    total_rewards = []
    while len(total_rewards) < num_eval_episodes:
        action_distrib = policy(obs)
        use_action = action_distrib.mean
        obs, _, done, info = envs.step(use_action)
        for done_i in torch.nonzero(done):
            # dists_to_goal.append(info[done_i]["dist_to_goal"])
            total_rewards.append(info[done_i]["episode"]["r"])
    return sum(total_rewards) / len(total_rewards)


# %%
def plot_true_performance(envs):
    grid_density = 20
    min_range = -2.0
    max_range = 2.0

    policy_values = torch.zeros(grid_density, grid_density)
    weight_X = torch.linspace(min_range, max_range, grid_density)
    weight_Y = torch.linspace(min_range, max_range, grid_density)
    policy = Policy()
    # Grid over [-2,0]^2
    for i, weight_x in enumerate(weight_X):
        for j, weight_y in enumerate(weight_Y):
            policy.weight.data.copy_(torch.tensor([weight_x, weight_y]))
            policy_values[i, j] = evaluate(1, policy, envs)

    fig = plt.imshow(
        policy_values,
        extent=[min_range, max_range, min_range, max_range],
        origin="lower",
    )
    print("Maximum possible reward is ", policy_values.max())


envs = create_vectorized_envs(
    "PointMass-v0",
    32,
    params=use_params,
)
plot_true_performance(envs)
plt.colorbar()
plt.savefig("data/perf_gt.png")
plt.clf()

# %%


def compute_returns(rewards, masks, gamma):
    returns = torch.zeros(rewards.shape[0] + 1, *rewards.shape[1:])
    for step in reversed(range(rewards.size(0))):
        returns[step] = returns[step + 1] * gamma * masks[step + 1] + rewards[step]
    return returns


def rollout_policy(policy, envs, num_steps):
    all_obs = torch.zeros(num_steps + 1, envs.num_envs, envs.observation_space.shape[0])
    all_rewards = torch.zeros(num_steps, envs.num_envs, 1)
    all_actions = torch.zeros(num_steps, envs.num_envs, envs.action_space.shape[0])
    all_masks = torch.zeros(num_steps + 1, envs.num_envs, 1)

    obs = envs.reset()
    all_obs[0].copy_(obs)

    for step_idx in range(num_steps):
        with torch.no_grad():
            action_distrib = policy(obs)
            take_action = action_distrib.sample()

        obs, reward, done, info = envs.step(take_action)
        all_obs[step_idx + 1].copy_(obs)
        all_rewards[step_idx].copy_(reward)
        all_actions[step_idx].copy_(take_action)
        all_masks[step_idx].copy_((~done).float().view(-1, 1))
    return all_obs, all_rewards, all_actions, all_masks


# %%

num_steps = 5
num_updates = 10
num_envs = 256
gamma = 0.99
envs = create_vectorized_envs(
    "PointMass-v0",
    num_envs,
    params=use_params,
)
policy = Policy()
opt = torch.optim.Adam(lr=1e-2, params=policy.parameters())
log_interval = 20

weight_seq = [policy.weight.data.detach().clone()]
for update_i in range(num_updates):
    obs, rewards, actions, masks = rollout_policy(policy, envs, num_steps)
    returns = compute_returns(rewards, masks, gamma)
    log_probs = policy(obs[:-1]).log_prob(actions).sum(-1, keepdim=True)
    loss = (-returns[:-1] * log_probs).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    weight_seq.append(policy.weight.detach().clone())

    if update_i % log_interval == 0:
        eval_envs = create_vectorized_envs(
            "PointMass-v0",
            num_envs,
            params=use_params,
        )
        total_reward = evaluate(10, policy, eval_envs)
        print(f"Update #{update_i}: Reward {total_reward:.4f}")

weight_seq = torch.stack(weight_seq, dim=0)
plot_true_performance(envs)
fig = plt.scatter(
    weight_seq[:, 0],
    weight_seq[:, 1],
    c=torch.arange(weight_seq.size(0)),
    s=4,
    cmap=plt.get_cmap("Reds"),
)
plt.savefig("data/perf_reinforce_opt.png")
plt.clf()

# %%
# Compute the loss function estimate as a function of the batch size.
grid_density = 50
min_range = -2.0
max_range = 2.0

weight_X = torch.linspace(min_range, max_range, grid_density)
weight_Y = torch.linspace(min_range, max_range, grid_density)

for num_envs in [1, 2, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    envs = create_vectorized_envs(
        "PointMass-v0",
        num_envs,
        params=use_params,
    )
    policy_values = torch.zeros(grid_density, grid_density)
    policy = Policy()
    for i, weight_x in enumerate(weight_X):
        for j, weight_y in enumerate(weight_Y):
            policy.weight.data.copy_(torch.tensor([weight_x, weight_y]))
            obs, rewards, actions, masks = rollout_policy(policy, envs, num_steps=5)
            returns = compute_returns(rewards, masks, gamma=0.99)

            log_probs = policy(obs[:-1]).log_prob(actions).sum(-1, keepdim=True)
            loss = (-returns[:-1] * log_probs).mean()
            policy_values[i, j] = loss.item()

    fig = plt.imshow(
        policy_values,
        extent=[min_range, max_range, min_range, max_range],
        origin="lower",
    )
    plt.savefig(f"data/perf_est_{num_envs}.png")
    plt.clf()

# %%
# Compute the gradient accuracy/variance
def get_loss(policy, envs):
    obs, rewards, actions, masks = rollout_policy(policy, envs, num_steps=5)
    returns = compute_returns(rewards, masks, gamma=0.99)
    log_probs = policy(obs[:-1]).log_prob(actions).sum(-1, keepdim=True)
    return (-returns[:-1] * log_probs).mean()


def compute_grad_mean_vars(env_params, append_name, env_sizes):
    policy = Policy()
    envs = create_vectorized_envs(
        "PointMass-v0",
        10000,
        params=env_params,
    )
    true_loss = get_loss(policy, envs)
    true_loss.backward(retain_graph=True)
    true_grad = policy.weight.grad.detach().clone()

    cosine_sim = nn.CosineSimilarity(dim=-1)
    gt_sims = []
    all_pairwise_sims = []
    n_var_samples = 100

    for num_envs in env_sizes:
        print("Computing for ", num_envs)
        envs = create_vectorized_envs(
            "PointMass-v0",
            num_envs,
            params=use_params,
        )

        all_grads = []
        for _ in range(n_var_samples):
            loss = get_loss(policy, envs)
            loss.backward(retain_graph=True)
            all_grads.append(policy.weight.grad.detach().clone())
        all_grads = torch.stack(all_grads, dim=0)

        gt_sims.append(cosine_sim(all_grads[0], true_grad).mean())
        all_pairwise_sims.append(all_grads.std(dim=0).mean())
    return gt_sims, all_pairwise_sims


stochastic_use_params = PointMassParams(
    clip_actions=True, radius=1.0, transition_noise=0.2
)

env_sizes = torch.tensor([1, 2, 4, 16, 32, 64, 128])

grad_acc, grad_var = compute_grad_mean_vars(use_params, "", env_sizes)
stoch_grad_acc, stoch_grad_var = compute_grad_mean_vars(
    stochastic_use_params, "stochastic_", env_sizes
)

plt.plot(env_sizes * 5, grad_acc, label="Deterministic")
plt.plot(env_sizes * 5, stoch_grad_acc, label="Stochastic")
plt.xscale("log", base=2)
plt.legend()
plt.savefig(f"data/grad_accuracy.png")
plt.clf()

plt.plot(env_sizes * 5, grad_var, label="Deterministic")
plt.plot(env_sizes * 5, stoch_grad_var, label="Stochastic")
plt.xscale("log", base=2)
plt.legend()
plt.savefig(f"data/grad_var.png")
plt.clf()


# %%
# Optimization (create a second area of high reward in different area, put
# penalty ring around the goal, what does the reward landscape look like for
# sparse reward?)
