"""Minimal TD3 network and replay helpers for fidelity-only calibration."""

from __future__ import annotations

import numpy as np


def _torch_stack():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - cluster environment dependency.
        raise RuntimeError("fidelity-only TD3 requires torch") from exc
    return torch, nn, functional


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int | None = None):
        if capacity < 1:
            raise ValueError("replay buffer capacity must be positive")
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.modes = np.zeros(self.capacity, dtype=np.int64)
        self.next_modes = np.zeros(self.capacity, dtype=np.int64)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, mode_index, action, reward, next_obs, next_mode_index, done) -> None:
        index = self.position
        self.obs[index] = np.asarray(obs, dtype=np.float32)
        self.modes[index] = int(mode_index)
        self.actions[index] = np.asarray(action, dtype=np.float32)
        self.rewards[index, 0] = float(reward)
        self.next_obs[index] = np.asarray(next_obs, dtype=np.float32)
        self.next_modes[index] = int(next_mode_index)
        self.dones[index, 0] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self.size < 1:
            raise RuntimeError("cannot sample from an empty replay buffer")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[indices],
            "modes": self.modes[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "next_modes": self.next_modes[indices],
            "dones": self.dones[indices],
        }


def make_generic_actor(input_dim: int, action_dim: int, hidden_layers: list[int]):
    torch, nn, _ = _torch_stack()

    class GenericActor(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            width = int(input_dim)
            for next_width in hidden_layers:
                layers.extend((nn.Linear(width, int(next_width)), nn.ReLU()))
                width = int(next_width)
            self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
            self.output = nn.Linear(width, int(action_dim))
            self.output.weight.data.zero_()
            self.output.bias.data.zero_()

        def forward(self, obs, mode_indices=None):
            del mode_indices
            return self.output(self.trunk(obs)).tanh()

    return GenericActor()


def make_twin_critic(input_dim: int, action_dim: int, hidden_layers: list[int]):
    torch, nn, _ = _torch_stack()

    def make_q_net():
        layers = []
        width = int(input_dim) + int(action_dim)
        for next_width in hidden_layers:
            layers.extend((nn.Linear(width, int(next_width)), nn.ReLU()))
            width = int(next_width)
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    class TwinCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.q1 = make_q_net()
            self.q2 = make_q_net()

        def forward(self, obs, action):
            values = torch.cat([obs, action], dim=1)
            return self.q1(values), self.q2(values)

        def q1_value(self, obs, action):
            return self.q1(torch.cat([obs, action], dim=1))

    return TwinCritic()


def _soft_update(source, target, tau: float) -> None:
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(1.0 - float(tau)).add_(float(tau) * source_param.data)


def _tensor_batch(torch, batch: dict[str, np.ndarray], device) -> dict:
    return {
        "obs": torch.as_tensor(batch["obs"], dtype=torch.float32, device=device),
        "modes": torch.as_tensor(batch["modes"], dtype=torch.long, device=device),
        "actions": torch.as_tensor(batch["actions"], dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device),
        "next_obs": torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device),
        "next_modes": torch.as_tensor(batch["next_modes"], dtype=torch.long, device=device),
        "dones": torch.as_tensor(batch["dones"], dtype=torch.float32, device=device),
    }
