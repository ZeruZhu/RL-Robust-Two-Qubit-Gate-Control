"""Cache-backed TD3 training for fidelity-only heterogeneous policies."""

from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

from ..config import dump_config, load_config
from ..envs.fidelity_only_multi_agent_env import (
    FIDELITY_ONLY_STAGE_ORDER,
    FidelityOnlyEightAgentEnv,
    fidelity_only_observation_dim,
    parse_fidelity_only_stage,
)
from ..fidelity_only_controls import log10_infidelity
from ..envs.two_qubit_env import TwoQubitWalshEnv
from .logging import save_json
from .fidelity_only_td3 import (
    ReplayBuffer,
    _soft_update,
    _tensor_batch,
    _torch_stack,
    make_generic_actor,
    make_twin_critic,
)


POLICY_FILENAME = "policy.pt"
CONTEXT_ARRAY_KEYS = (
    "seeds",
    "targets",
    "noise",
    "analytic_coefficients",
    "analytic_agfs",
    "alpha_commands",
    "g_commands",
    "x_commands",
    "y_commands",
    "current_coefficients",
    "current_superoperators",
    "current_agfs",
    "cumulative_pulse_evaluations",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _environment_signature(config: dict) -> str:
    keys = (
        "system",
        "control_model",
        "action_mode",
        "base_mode",
        "residual_scale",
        "target_gate",
        "target_sampling",
        "initial_gate",
        "segment_time",
        "kak_exchange_basis_convention",
        "num_walsh_modes",
        "num_time_bins",
        "walsh_order",
        "noise_observability",
        "reward_eps",
        "reward_mode",
        "control_bounds",
        "noise",
        "fidelity_only_eight_agent",
    )
    return json.dumps(
        {key: copy.deepcopy(config.get(key)) for key in keys},
        sort_keys=True,
        separators=(",", ":"),
    )


class FidelityOnlyContextDataset:
    """Read one immutable split from a materialized fidelity-only cache."""

    def __init__(self, cache_dir: str | Path, split: str):
        self.cache_dir = Path(cache_dir)
        self.split = str(split)
        self.env_config = load_config(self.cache_dir / "env_config.yaml")
        with np.load(
            self.cache_dir / f"{self.split}.npz", allow_pickle=False
        ) as payload:
            missing = set(CONTEXT_ARRAY_KEYS) - set(payload.files)
            if missing:
                raise ValueError(f"context cache is missing arrays: {sorted(missing)}")
            self.arrays = {key: payload[key] for key in CONTEXT_ARRAY_KEYS}
        self.size = int(self.arrays["seeds"].shape[0])
        if self.size < 1:
            raise ValueError("context cache split must be nonempty")
        for key, values in self.arrays.items():
            if values.shape[0] != self.size:
                raise ValueError(f"context array {key} has inconsistent leading size")
        metadata_path = self.cache_dir / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        expected_signature = self.metadata.get("environment_signature")
        if expected_signature and expected_signature != _environment_signature(self.env_config):
            raise ValueError("context cache environment signature mismatch")

    def row(self, index: int) -> dict:
        index = int(index)
        if index < 0 or index >= self.size:
            raise IndexError("context row out of range")
        return {
            "seed": int(self.arrays["seeds"][index]),
            "target": self.arrays["targets"][index].copy(),
            "noise": self.arrays["noise"][index].copy(),
            "analytic_coefficients": self.arrays["analytic_coefficients"][index].copy(),
            "analytic_agf": float(self.arrays["analytic_agfs"][index]),
            "alpha_commands": self.arrays["alpha_commands"][index].copy(),
            "g_commands": self.arrays["g_commands"][index].copy(),
            "x_commands": self.arrays["x_commands"][index].copy(),
            "y_commands": self.arrays["y_commands"][index].copy(),
            "current_coefficients": self.arrays["current_coefficients"][index].copy(),
            "current_superoperator": self.arrays["current_superoperators"][index].copy(),
            "current_agf": float(self.arrays["current_agfs"][index]),
            "cumulative_pulse_evaluations": int(
                self.arrays["cumulative_pulse_evaluations"][index]
            ),
            "stage_provenance": str(self.metadata.get("stage_provenance", "unknown")),
        }


def _rows_to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("cannot save an empty context cache")
    return {
        "seeds": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "targets": np.stack([row["target"] for row in rows]),
        "noise": np.stack([row["noise"] for row in rows]),
        "analytic_coefficients": np.stack(
            [row["analytic_coefficients"] for row in rows]
        ),
        "analytic_agfs": np.asarray(
            [row["analytic_agf"] for row in rows], dtype=float
        ),
        "alpha_commands": np.stack([row["alpha_commands"] for row in rows]),
        "g_commands": np.stack([row["g_commands"] for row in rows]),
        "x_commands": np.stack([row["x_commands"] for row in rows]),
        "y_commands": np.stack([row["y_commands"] for row in rows]),
        "current_coefficients": np.stack(
            [row["current_coefficients"] for row in rows]
        ),
        "current_superoperators": np.stack(
            [row["current_superoperator"] for row in rows]
        ),
        "current_agfs": np.asarray(
            [row["current_agf"] for row in rows], dtype=float
        ),
        "cumulative_pulse_evaluations": np.asarray(
            [row["cumulative_pulse_evaluations"] for row in rows], dtype=np.int64
        ),
    }


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def save_context_cache(
    cache_dir: str | Path,
    env_config: dict,
    split_rows: dict[str, list[dict]],
    *,
    stage_provenance: str,
    source_cache_dir: str | None = None,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dump_config(env_config, cache_dir / "env_config.yaml")
    split_sizes = {}
    for split, rows in split_rows.items():
        arrays = _rows_to_arrays(rows)
        _atomic_save_npz(cache_dir / f"{split}.npz", arrays)
        split_sizes[split] = int(arrays["seeds"].shape[0])
    save_json(
        {
            "version": 1,
            "stage_provenance": str(stage_provenance),
            "source_cache_dir": source_cache_dir,
            "environment_signature": _environment_signature(env_config),
            "split_sizes": split_sizes,
        },
        cache_dir / "metadata.json",
    )


def generate_base_context_cache(
    env_config: dict,
    output_dir: str | Path,
    split_specs: dict[str, tuple[int, int]],
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "version": 1,
        "environment_signature": _environment_signature(env_config),
        "split_specs": {
            split: [int(seed_start), int(count)]
            for split, (seed_start, count) in split_specs.items()
        },
    }
    request_path = output_dir / "generation_request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("base context generation request mismatch")
    else:
        save_json(request, request_path)
    dump_config(env_config, output_dir / "env_config.yaml")
    env = TwoQubitWalshEnv(env_config)
    split_sizes = {}
    chunk_dir = output_dir / ".partial"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for split, (seed_start, count) in split_specs.items():
        count = int(count)
        expected_seeds = np.arange(
            int(seed_start), int(seed_start) + count, dtype=np.int64
        )
        final_path = output_dir / f"{split}.npz"
        if final_path.exists():
            with np.load(final_path, allow_pickle=False) as payload:
                if not np.array_equal(payload["seeds"], expected_seeds):
                    raise ValueError(f"completed {split} context seeds do not match")
            split_sizes[split] = count
            continue
        chunks = []
        for start in range(0, count, 128):
            stop = min(count, start + 128)
            chunk_path = chunk_dir / f"{split}_{start:06d}_{stop:06d}.npz"
            if not chunk_path.exists():
                rows = []
                for offset in range(start, stop):
                    seed = int(seed_start) + offset
                    env.reset(seed=seed)
                    analytic = env.base_physical_coefficients.copy()
                    info = env.evaluate_physical_coefficients(analytic)
                    rows.append(
                        {
                            "seed": seed,
                            "target": env.target.copy(),
                            "noise": env.noise_sample.as_vector().copy(),
                            "analytic_coefficients": analytic,
                            "analytic_agf": float(info["average_gate_fidelity"]),
                            "alpha_commands": np.zeros((2, 8), dtype=float),
                            "g_commands": np.zeros((2, 3), dtype=float),
                            "x_commands": np.zeros((2, 8), dtype=float),
                            "y_commands": np.zeros((2, 8), dtype=float),
                            "current_coefficients": analytic.copy(),
                            "current_superoperator": env.current_operator.copy(),
                            "current_agf": float(info["average_gate_fidelity"]),
                            "cumulative_pulse_evaluations": 1,
                        }
                    )
                _atomic_save_npz(chunk_path, _rows_to_arrays(rows))
            with np.load(chunk_path, allow_pickle=False) as payload:
                values = {key: payload[key] for key in CONTEXT_ARRAY_KEYS}
            if not np.array_equal(values["seeds"], expected_seeds[start:stop]):
                raise ValueError(f"partial {split} context seeds do not match")
            chunks.append(values)
        combined = {
            key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
            for key in CONTEXT_ARRAY_KEYS
        }
        _atomic_save_npz(final_path, combined)
        for path in chunk_dir.glob(f"{split}_*.npz"):
            path.unlink()
        split_sizes[split] = count
    save_json(
        {
            "version": 1,
            "stage_provenance": "analytic",
            "source_cache_dir": None,
            "environment_signature": _environment_signature(env_config),
            "split_sizes": split_sizes,
        },
        output_dir / "metadata.json",
    )
    return split_sizes


def validate_base_context_cache_match(
    candidate_dir: str | Path,
    reference_dir: str | Path,
    *,
    canonicalize_numerical_drift: bool = False,
) -> dict:
    """Require two analytic caches to contain identical physical contexts."""
    candidate_dir = Path(candidate_dir)
    reference_dir = Path(reference_dir)
    candidate_metadata = json.loads(
        (candidate_dir / "metadata.json").read_text(encoding="utf-8")
    )
    reference_metadata = json.loads(
        (reference_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if candidate_metadata.get("stage_provenance") != "analytic":
        raise ValueError("candidate cache is not an analytic base cache")
    if reference_metadata.get("stage_provenance") != "analytic":
        raise ValueError("reference cache is not an analytic base cache")
    if candidate_metadata.get("split_sizes") != reference_metadata.get("split_sizes"):
        raise ValueError("candidate and reference cache split sizes differ")

    split_summaries = {}
    exact_keys = set(CONTEXT_ARRAY_KEYS) - {
        "analytic_coefficients",
        "current_coefficients",
        "current_superoperators",
    }
    numerical_tolerances = {
        "analytic_coefficients": 1.0e-7,
        "current_coefficients": 1.0e-7,
        "current_superoperators": 1.0e-12,
    }
    for split in sorted(candidate_metadata["split_sizes"]):
        with np.load(candidate_dir / f"{split}.npz", allow_pickle=False) as payload:
            candidate = {key: payload[key] for key in CONTEXT_ARRAY_KEYS}
        with np.load(reference_dir / f"{split}.npz", allow_pickle=False) as payload:
            reference = {key: payload[key] for key in CONTEXT_ARRAY_KEYS}
        mismatches = [
            key
            for key in CONTEXT_ARRAY_KEYS
            if (
                candidate[key].shape != reference[key].shape
                or candidate[key].dtype != reference[key].dtype
                or candidate[key].tobytes(order="C")
                != reference[key].tobytes(order="C")
            )
        ]
        strict_mismatches = sorted(set(mismatches) & exact_keys)
        if strict_mismatches:
            raise ValueError(
                f"{split} base context arrays differ from the legacy cache: "
                f"{strict_mismatches}"
            )
        numerical_mismatches = sorted(set(mismatches) - exact_keys)
        maximum_differences = {}
        for key in numerical_mismatches:
            maximum = float(np.max(np.abs(candidate[key] - reference[key])))
            maximum_differences[key] = maximum
            if not np.allclose(
                candidate[key],
                reference[key],
                rtol=0.0,
                atol=numerical_tolerances[key],
            ):
                raise ValueError(
                    f"{split} {key} differs beyond the numerical-drift "
                    f"tolerance: {maximum}"
                )
        canonicalized = []
        if numerical_mismatches and canonicalize_numerical_drift:
            for key in numerical_mismatches:
                candidate[key] = reference[key].copy()
                canonicalized.append(key)
            _atomic_save_npz(candidate_dir / f"{split}.npz", candidate)
            mismatches = []
        split_summaries[split] = {
            "samples": int(candidate["seeds"].shape[0]),
            "seed_start": int(candidate["seeds"][0]),
            "seed_stop": int(candidate["seeds"][-1]),
            "all_arrays_bitwise_equal": not mismatches,
            "numerical_drift_arrays": numerical_mismatches,
            "maximum_absolute_differences": maximum_differences,
            "canonicalized_arrays": canonicalized,
        }
    all_equal = all(
        item["all_arrays_bitwise_equal"] for item in split_summaries.values()
    )
    return {
        "candidate_cache_dir": str(candidate_dir.resolve()),
        "reference_cache_dir": str(reference_dir.resolve()),
        "all_arrays_bitwise_equal": all_equal,
        "canonicalize_numerical_drift": bool(canonicalize_numerical_drift),
        "splits": split_summaries,
    }


class FidelityOnlyPolicy:
    """Deterministic adapter for a saved heterogeneous TD3 actor."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu"):
        torch, _, _ = _torch_stack()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        payload = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
        if int(payload.get("version", -1)) != 1 or payload.get("kind") != "fidelity_only":
            raise ValueError("unsupported fidelity-only policy checkpoint")
        self.stage_key = str(payload["stage_key"])
        self.spec = parse_fidelity_only_stage(self.stage_key)
        self.obs_dim = int(payload["obs_dim"])
        self.action_dim = int(payload["action_dim"])
        self.hidden_layers = tuple(int(value) for value in payload["hidden_layers"])
        self.environment_signature = str(payload["environment_signature"])
        if self.obs_dim < self.spec.observation_dim or self.action_dim != self.spec.action_dim:
            raise ValueError("fidelity-only checkpoint dimensions do not match its stage")
        self.actor = make_generic_actor(
            self.obs_dim, self.action_dim, list(self.hidden_layers)
        ).to(device)
        self.actor.load_state_dict(payload["actor"], strict=True)
        self.actor.eval()
        self.device = torch.device(device)
        digest = hashlib.sha256()
        with self.checkpoint_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.checkpoint_sha256 = digest.hexdigest()

    def predict(self, observation: np.ndarray) -> np.ndarray:
        torch, _, _ = _torch_stack()
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (self.obs_dim,):
            raise ValueError(f"{self.stage_key} observation shape mismatch")
        with torch.no_grad():
            tensor = torch.as_tensor(values[None, :], device=self.device)
            action = self.actor(tensor, None).cpu().numpy()[0]
        return np.clip(action, -1.0, 1.0)


@dataclass
class FidelityOnlyTrainingConfig:
    env_config: dict
    active_stage: str
    input_cache_dir: str
    output_dir: str
    stage_episodes: int = 20000
    learning_starts: int = 1000
    buffer_size: int = 100000
    batch_size: int = 512
    actor_hidden_layers: tuple[int, ...] = (300, 300, 300, 300, 300)
    critic_hidden_layers: tuple[int, ...] = (300, 300, 300, 300, 300)
    actor_lr: float = 4.0e-5
    critic_lr: float = 3.0e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    action_noise_sigma: float = 0.10
    target_policy_noise: float = 0.05
    target_noise_clip: float = 0.10
    checkpoint_every_episodes: int = 500
    eval_every_episodes: int = 1000
    checkpoint_eval_episodes: int = 256
    seed: int = 17
    device: str = "auto"
    save_replay_buffer: bool = True
    resume_checkpoint: str | None = None


@dataclass
class _Agent:
    actor: object
    actor_target: object
    critic: object
    critic_target: object
    actor_optimizer: object
    critic_optimizer: object
    replay: ReplayBuffer
    rng: np.random.Generator
    transitions: int = 0
    updates: int = 0
    last_actor_loss: float = np.nan
    last_critic_loss: float = np.nan


def _device(config: FidelityOnlyTrainingConfig):
    torch, _, _ = _torch_stack()
    return torch.device(
        config.device
        if config.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def _make_agent(config: FidelityOnlyTrainingConfig, spec, device) -> _Agent:
    torch, _, _ = _torch_stack()
    observation_dim = fidelity_only_observation_dim(config.env_config, spec.key)
    actor = make_generic_actor(
        observation_dim, spec.action_dim, list(config.actor_hidden_layers)
    ).to(device)
    actor_target = copy.deepcopy(actor).to(device)
    critic = make_twin_critic(
        observation_dim, spec.action_dim, list(config.critic_hidden_layers)
    ).to(device)
    critic_target = copy.deepcopy(critic).to(device)
    return _Agent(
        actor=actor,
        actor_target=actor_target,
        critic=critic,
        critic_target=critic_target,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=float(config.actor_lr)),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=float(config.critic_lr)),
        replay=ReplayBuffer(
            int(config.buffer_size),
            observation_dim,
            spec.action_dim,
            seed=int(config.seed) + 101,
        ),
        rng=np.random.default_rng(int(config.seed) + 202),
    )


def _action(agent: _Agent, observation: np.ndarray, device) -> np.ndarray:
    torch, _, _ = _torch_stack()
    with torch.no_grad():
        tensor = torch.as_tensor(observation[None, :], dtype=torch.float32, device=device)
        return np.clip(agent.actor(tensor, None).cpu().numpy()[0], -1.0, 1.0)


def _exploration_action(
    agent: _Agent,
    observation: np.ndarray,
    sigma: float,
    device,
) -> np.ndarray:
    action = _action(agent, observation, device)
    action += agent.rng.normal(0.0, float(sigma), size=action.shape)
    return np.clip(action, -1.0, 1.0)


def _update_agent(agent: _Agent, config: FidelityOnlyTrainingConfig, device) -> None:
    torch, _, functional = _torch_stack()
    batch = _tensor_batch(torch, agent.replay.sample(config.batch_size), device)
    with torch.no_grad():
        noise = torch.randn_like(batch["actions"]) * float(config.target_policy_noise)
        noise = noise.clamp(-float(config.target_noise_clip), float(config.target_noise_clip))
        next_actions = (agent.actor_target(batch["next_obs"], None) + noise).clamp(-1.0, 1.0)
        q1_target, q2_target = agent.critic_target(batch["next_obs"], next_actions)
        target = batch["rewards"] + (1.0 - batch["dones"]) * float(config.gamma) * torch.minimum(
            q1_target, q2_target
        )
    q1, q2 = agent.critic(batch["obs"], batch["actions"])
    critic_loss = functional.mse_loss(q1, target) + functional.mse_loss(q2, target)
    agent.critic_optimizer.zero_grad()
    critic_loss.backward()
    agent.critic_optimizer.step()
    agent.last_critic_loss = float(critic_loss.detach().cpu().item())
    if agent.updates % int(config.policy_delay) == 0:
        actor_loss = -agent.critic.q1_value(
            batch["obs"], agent.actor(batch["obs"], None)
        ).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()
        _soft_update(agent.actor, agent.actor_target, config.tau)
        _soft_update(agent.critic, agent.critic_target, config.tau)
        agent.last_actor_loss = float(actor_loss.detach().cpu().item())
    agent.updates += 1


def _policy_payload(
    agent: _Agent,
    config: FidelityOnlyTrainingConfig,
    *,
    episode: int,
    summary: dict | None = None,
) -> dict:
    spec = parse_fidelity_only_stage(config.active_stage)
    observation_dim = fidelity_only_observation_dim(config.env_config, spec.key)
    return {
        "version": 1,
        "kind": "fidelity_only",
        "stage_key": spec.key,
        "obs_dim": observation_dim,
        "action_dim": spec.action_dim,
        "hidden_layers": list(config.actor_hidden_layers),
        "actor": copy.deepcopy(agent.actor.state_dict()),
        "episode": int(episode),
        "summary": summary or {},
        "environment_signature": _environment_signature(config.env_config),
    }


def _save_policy(
    directory: Path,
    agent: _Agent,
    config: FidelityOnlyTrainingConfig,
    *,
    episode: int,
    summary: dict | None = None,
) -> None:
    torch, _, _ = _torch_stack()
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"{POLICY_FILENAME}.tmp"
    torch.save(
        _policy_payload(agent, config, episode=episode, summary=summary), temporary
    )
    os.replace(temporary, directory / POLICY_FILENAME)
    save_json(
        {
            "stage_key": config.active_stage,
            "episode": int(episode),
            "summary": summary or {},
        },
        directory / "metadata.json",
    )


def _evaluate_policy(
    agent: _Agent,
    config: FidelityOnlyTrainingConfig,
    dataset: FidelityOnlyContextDataset,
    *,
    episodes: int,
    save_traces: bool,
) -> dict:
    count = min(int(episodes), dataset.size)
    env = FidelityOnlyEightAgentEnv(config.env_config, active_stage=config.active_stage)
    device = _device(config)
    incoming = np.zeros(count, dtype=float)
    final = np.zeros(count, dtype=float)
    analytic = np.zeros(count, dtype=float)
    actions = (
        np.zeros((count, env.spec.steps, env.spec.action_dim), dtype=np.float32)
        if save_traces
        else None
    )
    score_changes = np.zeros((count, env.spec.steps), dtype=float) if save_traces else None
    accepted = np.zeros((count, env.spec.steps), dtype=bool) if save_traces else None
    coefficients = np.zeros((count, 2, 27), dtype=float) if save_traces else None
    pulse_evaluations = 0
    acceptance_rates = []
    gamma_clip_rates = []
    g_exceedance_rates = []
    observation_out_of_range_rates = []
    monotonicity_violations = 0
    for index in range(count):
        observation, info = env.reset_from_context(dataset.row(index))
        incoming[index] = info["incoming_average_gate_fidelity"]
        analytic[index] = info["analytic_average_gate_fidelity"]
        done = False
        while not done:
            step_index = env.step_index
            action = _action(agent, observation, device)
            observation, _, done, _, info = env.step(action)
            if save_traces:
                actions[index, step_index] = action
                score_changes[index, step_index] = info["score_changes"][-1]
                accepted[index, step_index] = info["accepted_history"][-1]
        final[index] = info["average_gate_fidelity"]
        pulse_evaluations += info["pulse_evaluations"]
        acceptance_rates.append(info["acceptance_rate"])
        gamma_clip_rates.append(info["gamma_clip_rate"])
        g_exceedance_rates.append(info["g_nominal_exceedance_rate"])
        observation_out_of_range_rates.append(info["observation_out_of_range_rate"])
        monotonicity_violations += int(info["monotonicity_violations"])
        if save_traces:
            coefficients[index] = info["physical_coefficients"]
    summary = {
        "active_stage": config.active_stage,
        "eval_episodes": count,
        "observation_dim": env.observation_dim,
        "action_dim": parse_fidelity_only_stage(config.active_stage).action_dim,
        "analytic_average_gate_fidelity": float(np.mean(analytic)),
        "analytic_average_gate_fidelity_std": float(np.std(analytic)),
        "incoming_average_gate_fidelity": float(np.mean(incoming)),
        "incoming_average_gate_fidelity_std": float(np.std(incoming)),
        "final_average_gate_fidelity": float(np.mean(final)),
        "final_average_gate_fidelity_std": float(np.std(final)),
        "final_log10_infidelity": log10_infidelity(float(np.mean(final)), 1.0e-8),
        "mean_improvement_over_incoming": float(np.mean(final - incoming)),
        "win_rate_over_incoming": float(np.mean(final > incoming)),
        "acceptance_rate": float(np.mean(acceptance_rates)),
        "gamma_clip_rate": float(np.mean(gamma_clip_rates)),
        "g_nominal_exceedance_rate": float(np.mean(g_exceedance_rates)),
        "observation_out_of_range_rate": float(
            np.mean(observation_out_of_range_rates)
        ),
        "monotonicity_violations": int(monotonicity_violations),
        "pulse_evaluations": int(pulse_evaluations),
    }
    return {
        "summary": summary,
        "analytic": analytic,
        "incoming": incoming,
        "final": final,
        "actions": actions,
        "score_changes": score_changes,
        "accepted": accepted,
        "coefficients": coefficients,
    }


def materialize_policy_cache(
    policy: FidelityOnlyPolicy,
    env_config: dict,
    input_cache_dir: str | Path,
    output_cache_dir: str | Path,
) -> dict:
    if policy.environment_signature != _environment_signature(env_config):
        raise ValueError("materialization policy environment mismatch")
    output_cache_dir = Path(output_cache_dir)
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "version": 1,
        "environment_signature": _environment_signature(env_config),
        "stage_key": policy.stage_key,
        "policy_sha256": policy.checkpoint_sha256,
        "source_cache_dir": str(Path(input_cache_dir).resolve()),
    }
    request_path = output_cache_dir / "materialization_request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("materialized context request mismatch")
    else:
        save_json(request, request_path)
    dump_config(env_config, output_cache_dir / "env_config.yaml")
    chunk_dir = output_cache_dir / ".partial"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pulse_counts = {}
    split_sizes = {}
    for split in ("train", "validation", "test"):
        dataset = FidelityOnlyContextDataset(input_cache_dir, split)
        if _environment_signature(dataset.env_config) != _environment_signature(env_config):
            raise ValueError("materialization input-cache environment mismatch")
        expected_provenance = _expected_input_provenance(policy.stage_key)
        if dataset.metadata.get("stage_provenance") != expected_provenance:
            raise ValueError("materialization input-cache provenance mismatch")
        final_path = output_cache_dir / f"{split}.npz"
        if final_path.exists():
            with np.load(final_path, allow_pickle=False) as payload:
                if not np.array_equal(payload["seeds"], dataset.arrays["seeds"]):
                    raise ValueError(f"materialized {split} seeds do not match input")
            split_sizes[split] = dataset.size
            pulse_counts[split] = dataset.size * policy.spec.steps
            continue
        env = FidelityOnlyEightAgentEnv(env_config, active_stage=policy.stage_key)
        chunks = []
        for start in range(0, dataset.size, 128):
            stop = min(dataset.size, start + 128)
            chunk_path = chunk_dir / f"{split}_{start:06d}_{stop:06d}.npz"
            if not chunk_path.exists():
                rows = []
                for index in range(start, stop):
                    observation, _ = env.reset_from_context(dataset.row(index))
                    done = False
                    while not done:
                        observation, _, done, _, info = env.step(
                            policy.predict(observation)
                        )
                    if int(info["pulse_evaluations"]) != policy.spec.steps:
                        raise RuntimeError("materialization pulse count mismatch")
                    rows.append(env.export_context())
                _atomic_save_npz(chunk_path, _rows_to_arrays(rows))
            with np.load(chunk_path, allow_pickle=False) as payload:
                values = {key: payload[key] for key in CONTEXT_ARRAY_KEYS}
            if not np.array_equal(
                values["seeds"], dataset.arrays["seeds"][start:stop]
            ):
                raise ValueError(f"partial materialized {split} seeds do not match")
            chunks.append(values)
        combined = {
            key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
            for key in CONTEXT_ARRAY_KEYS
        }
        _atomic_save_npz(final_path, combined)
        for path in chunk_dir.glob(f"{split}_*.npz"):
            path.unlink()
        split_sizes[split] = dataset.size
        pulse_counts[split] = dataset.size * policy.spec.steps
    save_json(
        {
            "version": 1,
            "stage_provenance": policy.stage_key,
            "source_cache_dir": str(Path(input_cache_dir).resolve()),
            "environment_signature": _environment_signature(env_config),
            "split_sizes": split_sizes,
            "policy_sha256": policy.checkpoint_sha256,
        },
        output_cache_dir / "metadata.json",
    )
    return pulse_counts


def _save_replay(path: Path, replay: ReplayBuffer) -> None:
    size = int(replay.size)
    np.savez_compressed(
        path,
        obs=replay.obs[:size],
        next_obs=replay.next_obs[:size],
        actions=replay.actions[:size],
        rewards=replay.rewards[:size],
        dones=replay.dones[:size],
    )


def _expected_input_provenance(stage_key: str) -> str:
    index = FIDELITY_ONLY_STAGE_ORDER.index(str(stage_key))
    return "analytic" if index == 0 else FIDELITY_ONLY_STAGE_ORDER[index - 1]


def _validate_input_cache(
    config: FidelityOnlyTrainingConfig,
    *datasets: FidelityOnlyContextDataset,
) -> None:
    expected_signature = _environment_signature(config.env_config)
    expected_provenance = _expected_input_provenance(config.active_stage)
    for dataset in datasets:
        if _environment_signature(dataset.env_config) != expected_signature:
            raise ValueError("training cache environment mismatch")
        actual = str(dataset.metadata.get("stage_provenance", "unknown"))
        if actual != expected_provenance:
            raise ValueError(
                f"{config.active_stage} requires a {expected_provenance!r} cache, "
                f"got {actual!r}"
            )


def _training_signature(config: FidelityOnlyTrainingConfig) -> str:
    values = asdict(config)
    values.pop("output_dir", None)
    values.pop("resume_checkpoint", None)
    values["input_cache_dir"] = str(Path(config.input_cache_dir).resolve())
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _replay_state(replay: ReplayBuffer) -> dict:
    size = int(replay.size)
    return {
        "size": size,
        "position": int(replay.position),
        "obs": replay.obs[:size].copy(),
        "next_obs": replay.next_obs[:size].copy(),
        "actions": replay.actions[:size].copy(),
        "rewards": replay.rewards[:size].copy(),
        "dones": replay.dones[:size].copy(),
        "modes": replay.modes[:size].copy(),
        "next_modes": replay.next_modes[:size].copy(),
        "rng_state": copy.deepcopy(replay.rng.bit_generator.state),
    }


def _restore_replay(replay: ReplayBuffer, state: dict) -> None:
    size = int(state["size"])
    if size < 0 or size > replay.capacity:
        raise ValueError("resume replay size exceeds configured capacity")
    destinations = {
        "obs": replay.obs,
        "next_obs": replay.next_obs,
        "actions": replay.actions,
        "rewards": replay.rewards,
        "dones": replay.dones,
        "modes": replay.modes,
        "next_modes": replay.next_modes,
    }
    for key, destination in destinations.items():
        values = np.asarray(state[key])
        if values.shape != destination[:size].shape:
            raise ValueError(f"resume replay {key} shape mismatch")
        destination[:size] = values
    replay.size = size
    replay.position = int(state["position"])
    if replay.position < 0 or replay.position >= replay.capacity:
        raise ValueError("resume replay position is invalid")
    replay.rng.bit_generator.state = copy.deepcopy(state["rng_state"])


def _save_resume_checkpoint(
    path: Path,
    agent: _Agent,
    config: FidelityOnlyTrainingConfig,
    *,
    episode: int,
    training_rows: list[dict],
    evaluation_rows: list[dict],
    best_mean: float,
    best_episode: int,
    order: np.ndarray,
    permutation_rng: np.random.Generator,
) -> None:
    torch, _, _ = _torch_stack()
    payload = {
        "version": 1,
        "kind": "fidelity_only_training_resume",
        "training_signature": _training_signature(config),
        "episode": int(episode),
        "actor": agent.actor.state_dict(),
        "actor_target": agent.actor_target.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "replay": _replay_state(agent.replay),
        "agent_rng_state": copy.deepcopy(agent.rng.bit_generator.state),
        "transitions": int(agent.transitions),
        "updates": int(agent.updates),
        "last_actor_loss": float(agent.last_actor_loss),
        "last_critic_loss": float(agent.last_critic_loss),
        "training_rows": training_rows,
        "evaluation_rows": evaluation_rows,
        "best_mean": float(best_mean),
        "best_episode": int(best_episode),
        "order": np.asarray(order, dtype=np.int64),
        "permutation_rng_state": copy.deepcopy(
            permutation_rng.bit_generator.state
        ),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _restore_resume_checkpoint(
    path: Path,
    agent: _Agent,
    config: FidelityOnlyTrainingConfig,
    permutation_rng: np.random.Generator,
    device,
) -> dict:
    torch, _, _ = _torch_stack()
    # Resume files are trusted local training state, including NumPy RNG/replay arrays.
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        int(payload.get("version", -1)) != 1
        or payload.get("kind") != "fidelity_only_training_resume"
    ):
        raise ValueError("unsupported fidelity-only resume checkpoint")
    if payload.get("training_signature") != _training_signature(config):
        raise ValueError("fidelity-only resume configuration mismatch")
    agent.actor.load_state_dict(payload["actor"], strict=True)
    agent.actor_target.load_state_dict(payload["actor_target"], strict=True)
    agent.critic.load_state_dict(payload["critic"], strict=True)
    agent.critic_target.load_state_dict(payload["critic_target"], strict=True)
    agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    _restore_replay(agent.replay, payload["replay"])
    agent.rng.bit_generator.state = copy.deepcopy(payload["agent_rng_state"])
    agent.transitions = int(payload["transitions"])
    agent.updates = int(payload["updates"])
    agent.last_actor_loss = float(payload["last_actor_loss"])
    agent.last_critic_loss = float(payload["last_critic_loss"])
    permutation_rng.bit_generator.state = copy.deepcopy(
        payload["permutation_rng_state"]
    )
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"].cpu())
    if torch.cuda.is_available() and "torch_cuda_random_state" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda_random_state"])
    return payload


def train_fidelity_only_stage(config: FidelityOnlyTrainingConfig) -> dict:
    torch, _, _ = _torch_stack()
    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    random.seed(int(config.seed))
    spec = parse_fidelity_only_stage(config.active_stage)
    input_cache = FidelityOnlyContextDataset(config.input_cache_dir, "train")
    validation_cache = FidelityOnlyContextDataset(config.input_cache_dir, "validation")
    _validate_input_cache(config, input_cache, validation_cache)
    completed_summary = run_dir / "stage_summary.json"
    completed_cache = run_dir / "materialized_cache"
    if completed_summary.exists() and all(
        (completed_cache / name).exists()
        for name in ("train.npz", "validation.npz", "test.npz", "metadata.json")
    ):
        metadata = json.loads((completed_cache / "metadata.json").read_text())
        if metadata.get("environment_signature") != _environment_signature(
            config.env_config
        ):
            raise ValueError("completed stage cache environment mismatch")
        if metadata.get("stage_provenance") != config.active_stage:
            raise ValueError("completed stage cache provenance mismatch")
        return json.loads(completed_summary.read_text())
    device = _device(config)
    agent = _make_agent(config, spec, device)
    env = FidelityOnlyEightAgentEnv(config.env_config, active_stage=spec.key)
    permutation_rng = np.random.default_rng(int(config.seed) + 303)
    order = permutation_rng.permutation(input_cache.size)
    resume_path = (
        Path(config.resume_checkpoint)
        if config.resume_checkpoint
        else run_dir / "resume_checkpoint.pt"
    )
    if resume_path.exists():
        payload = _restore_resume_checkpoint(
            resume_path, agent, config, permutation_rng, device
        )
        training_rows = list(payload["training_rows"])
        evaluation_rows = list(payload["evaluation_rows"])
        best_mean = float(payload["best_mean"])
        best_episode = int(payload["best_episode"])
        order = np.asarray(payload["order"], dtype=np.int64)
        start_episode = int(payload["episode"]) + 1
        if not (run_dir / "best_policy" / POLICY_FILENAME).exists():
            raise ValueError("resume checkpoint exists but best policy is missing")
    else:
        training_rows = []
        evaluation_rows = []
        best_episode = 0
        initial = _evaluate_policy(
            agent,
            config,
            validation_cache,
            episodes=config.checkpoint_eval_episodes,
            save_traces=False,
        )
        initial_summary = initial["summary"]
        evaluation_rows.append({"training_episode": 0, **initial_summary})
        best_mean = float(initial_summary["final_average_gate_fidelity"])
        _save_policy(
            run_dir / "best_policy",
            agent,
            config,
            episode=0,
            summary=initial_summary,
        )
        start_episode = 1
        _save_resume_checkpoint(
            resume_path,
            agent,
            config,
            episode=0,
            training_rows=training_rows,
            evaluation_rows=evaluation_rows,
            best_mean=best_mean,
            best_episode=best_episode,
            order=order,
            permutation_rng=permutation_rng,
        )

    for episode in range(start_episode, int(config.stage_episodes) + 1):
        if (episode - 1) % input_cache.size == 0 and episode > 1:
            order = permutation_rng.permutation(input_cache.size)
        row_index = int(order[(episode - 1) % input_cache.size])
        observation, reset_info = env.reset_from_context(input_cache.row(row_index))
        done = False
        info = reset_info
        total_reward = 0.0
        while not done:
            action = _exploration_action(
                agent, observation, config.action_noise_sigma, device
            )
            next_observation, reward, done, _, info = env.step(action)
            agent.replay.add(
                observation, 0, action, reward, next_observation, 0, done
            )
            agent.transitions += 1
            total_reward += float(reward)
            if agent.transitions >= int(config.learning_starts):
                _update_agent(agent, config, device)
            observation = next_observation

        training_rows.append(
            {
                "training_episode": episode,
                "total_interactions": agent.transitions,
                "total_reward": total_reward,
                "incoming_average_gate_fidelity": info["incoming_average_gate_fidelity"],
                "final_average_gate_fidelity": info["average_gate_fidelity"],
                "improvement_over_incoming": (
                    info["average_gate_fidelity"] - info["incoming_average_gate_fidelity"]
                ),
                "acceptance_rate": info["acceptance_rate"],
                "monotonicity_violations": info["monotonicity_violations"],
                "gamma_clip_rate": info["gamma_clip_rate"],
                "g_nominal_exceedance_rate": info["g_nominal_exceedance_rate"],
                "pulse_evaluations": info["pulse_evaluations"],
                "actor_loss": agent.last_actor_loss,
                "critic_loss": agent.last_critic_loss,
            }
        )

        if episode % int(config.eval_every_episodes) == 0 or episode == int(config.stage_episodes):
            evaluation = _evaluate_policy(
                agent,
                config,
                validation_cache,
                episodes=config.checkpoint_eval_episodes,
                save_traces=False,
            )
            summary = evaluation["summary"]
            evaluation_rows.append({"training_episode": episode, **summary})
            score = float(summary["final_average_gate_fidelity"])
            if score > best_mean:
                best_mean = score
                best_episode = episode
                _save_policy(
                    run_dir / "best_policy",
                    agent,
                    config,
                    episode=episode,
                    summary=summary,
                )
            _write_csv(run_dir / "training_eval.csv", evaluation_rows)

        if episode % int(config.checkpoint_every_episodes) == 0:
            _write_csv(run_dir / "training_metrics.csv", training_rows)
            _save_policy(run_dir / "latest_policy", agent, config, episode=episode)
            _save_resume_checkpoint(
                resume_path,
                agent,
                config,
                episode=episode,
                training_rows=training_rows,
                evaluation_rows=evaluation_rows,
                best_mean=best_mean,
                best_episode=best_episode,
                order=order,
                permutation_rng=permutation_rng,
            )

    _save_policy(
        run_dir / "final_policy",
        agent,
        config,
        episode=int(config.stage_episodes),
    )
    _write_csv(run_dir / "training_metrics.csv", training_rows)
    _write_csv(run_dir / "training_eval.csv", evaluation_rows)
    if config.save_replay_buffer:
        _save_replay(run_dir / "replay_buffer.npz", agent.replay)
    _save_resume_checkpoint(
        resume_path,
        agent,
        config,
        episode=int(config.stage_episodes),
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        best_mean=best_mean,
        best_episode=best_episode,
        order=order,
        permutation_rng=permutation_rng,
    )

    best_policy = FidelityOnlyPolicy(
        run_dir / "best_policy" / POLICY_FILENAME, device=str(device)
    )
    materialization_counts = materialize_policy_cache(
        best_policy,
        config.env_config,
        config.input_cache_dir,
        run_dir / "materialized_cache",
    )
    final_cache = FidelityOnlyContextDataset(run_dir / "materialized_cache", "test")
    final = final_cache.arrays["current_agfs"]
    incoming_test = FidelityOnlyContextDataset(config.input_cache_dir, "test")
    incoming = incoming_test.arrays["current_agfs"]
    analytic = final_cache.arrays["analytic_agfs"]
    best_eval_agent = _make_agent(config, spec, device)
    best_eval_agent.actor.load_state_dict(best_policy.actor.state_dict(), strict=True)
    test_evaluation = _evaluate_policy(
        best_eval_agent,
        config,
        incoming_test,
        episodes=incoming_test.size,
        save_traces=True,
    )
    if not np.allclose(
        test_evaluation["final"], final, rtol=0.0, atol=3.0e-10
    ):
        raise RuntimeError("final evaluation does not reproduce materialized cache")
    np.save(run_dir / "test_actions.npy", test_evaluation["actions"])
    np.save(run_dir / "test_score_changes.npy", test_evaluation["score_changes"])
    np.save(run_dir / "test_accepted_proposals.npy", test_evaluation["accepted"])
    np.save(run_dir / "test_physical_coefficients.npy", test_evaluation["coefficients"])
    training_pulses = int(sum(row["pulse_evaluations"] for row in training_rows))
    checkpoint_eval_pulses = int(
        sum(row["pulse_evaluations"] for row in evaluation_rows)
    )
    summary = {
        "active_stage": spec.key,
        "best_checkpoint_episode": int(best_episode),
        "best_checkpoint_validation_agf": float(best_mean),
        "training_episodes": int(config.stage_episodes),
        "training_interactions": int(agent.transitions),
        "observation_dim": fidelity_only_observation_dim(
            config.env_config, spec.key
        ),
        "action_dim": spec.action_dim,
        "test_episodes": int(final_cache.size),
        "analytic_average_gate_fidelity": float(np.mean(analytic)),
        "incoming_average_gate_fidelity": float(np.mean(incoming)),
        "incoming_average_gate_fidelity_std": float(np.std(incoming)),
        "final_average_gate_fidelity": float(np.mean(final)),
        "final_average_gate_fidelity_std": float(np.std(final)),
        "final_average_gate_fidelity_median": float(np.median(final)),
        "final_average_gate_fidelity_p10": float(np.quantile(final, 0.10)),
        "final_average_gate_fidelity_minimum": float(np.min(final)),
        "final_log10_infidelity": log10_infidelity(float(np.mean(final)), 1.0e-8),
        "mean_improvement_over_incoming": float(np.mean(final - incoming)),
        "win_rate_over_incoming": float(np.mean(final > incoming)),
        "loss_rate_over_incoming": float(np.mean(final < incoming)),
        "tie_rate_over_incoming": float(np.mean(final == incoming)),
        "acceptance_rate": float(test_evaluation["summary"]["acceptance_rate"]),
        "rejected_proposal_count": int(
            np.count_nonzero(~test_evaluation["accepted"])
        ),
        "rejected_proposal_reward_mean": float(
            np.mean(
                test_evaluation["score_changes"][~test_evaluation["accepted"]]
            )
            if np.any(~test_evaluation["accepted"])
            else 0.0
        ),
        "rejected_proposal_reward_std": float(
            np.std(
                test_evaluation["score_changes"][~test_evaluation["accepted"]]
            )
            if np.any(~test_evaluation["accepted"])
            else 0.0
        ),
        "monotonicity_violations": int(
            test_evaluation["summary"]["monotonicity_violations"]
        ),
        "gamma_clip_rate": float(test_evaluation["summary"]["gamma_clip_rate"]),
        "g_nominal_exceedance_rate": float(
            test_evaluation["summary"]["g_nominal_exceedance_rate"]
        ),
        "observation_out_of_range_rate": float(
            test_evaluation["summary"]["observation_out_of_range_rate"]
        ),
        "training_pulse_evaluations": training_pulses,
        "checkpoint_evaluation_pulse_evaluations": checkpoint_eval_pulses,
        "final_evaluation_pulse_evaluations": int(
            test_evaluation["summary"]["pulse_evaluations"]
        ),
        "materialization_pulse_evaluations": materialization_counts,
        "input_cache_dir": str(Path(config.input_cache_dir).resolve()),
        "materialized_cache_dir": str((run_dir / "materialized_cache").resolve()),
    }
    save_json(summary, run_dir / "stage_summary.json")
    np.save(run_dir / "test_analytic_average_gate_fidelities.npy", analytic)
    np.save(run_dir / "test_incoming_average_gate_fidelities.npy", incoming)
    np.save(run_dir / "test_final_average_gate_fidelities.npy", final)
    return summary


__all__ = [
    "CONTEXT_ARRAY_KEYS",
    "FidelityOnlyContextDataset",
    "FidelityOnlyPolicy",
    "FidelityOnlyTrainingConfig",
    "generate_base_context_cache",
    "validate_base_context_cache_match",
    "materialize_policy_cache",
    "save_context_cache",
    "train_fidelity_only_stage",
]
