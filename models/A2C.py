import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        layers.append(nn.Linear(last_dim, act_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, obs):
        return self.model(obs)


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, obs):
        return self.model(obs).squeeze(-1)


class A2CAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes=(128, 128),
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        device="cpu",
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = device

        self.actor = ActorNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.critic = CriticNetwork(obs_dim, hidden_sizes).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # buffers
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.logp_buf = []
        self.value_buf = []   # STORES TENSORS

    # -------------------------------------------------------------
    def select_action(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)

        logits = self.actor(obs_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        logp = dist.log_prob(action)

        value = self.critic(obs_t)   # TENSOR WITH GRAD

        return (
            action.item(),
            logp.detach(),          # store detached version
            value.squeeze(0)        # store tensor, NOT a float
        )

    # -------------------------------------------------------------
    def store(self, obs, act, rew, done, logp, value):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.done_buf.append(done)
        self.logp_buf.append(logp)
        self.value_buf.append(value)   # STORES TENSOR

    # -------------------------------------------------------------
    def compute_returns(self):
        returns = []
        R = 0
        for reward, done in zip(reversed(self.rew_buf), reversed(self.done_buf)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    # -------------------------------------------------------------
    def update(self):
        obs = torch.tensor(np.array(self.obs_buf), dtype=torch.float32, device=self.device)
        acts = torch.tensor(self.act_buf, dtype=torch.long, device=self.device)

        values = torch.stack(self.value_buf).to(self.device)   # KEEPS GRAD

        returns = self.compute_returns()
        advantage = returns - values.detach()

        # Actor forward
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        logps = dist.log_prob(acts)
        entropy = dist.entropy().mean()

        actor_loss = -(logps * advantage).mean() - self.entropy_coef * entropy
        critic_loss = F.mse_loss(values, returns)

        # Update actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # clear buffers
        self.obs_buf.clear()
        self.act_buf.clear()
        self.rew_buf.clear()
        self.done_buf.clear()
        self.logp_buf.clear()
        self.value_buf.clear()

        return actor_loss.item(), critic_loss.item()
