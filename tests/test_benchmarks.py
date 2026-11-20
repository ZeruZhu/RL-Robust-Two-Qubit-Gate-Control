from pathlib import Path

import numpy as np
import pytest

from walsh_rl.config import load_config
from walsh_rl.envs.two_qubit_env import TwoQubitWalshEnv
from walsh_rl.benchmarks import grape, fourier
from walsh_rl.benchmarks.continuous_adjoint import fidelity_gradient

ROOT = Path(__file__).resolve().parents[1]


def context():
    config = load_config(ROOT/"configs/cnot_segment.yaml")
    env = TwoQubitWalshEnv(config)
    env.reset(seed=920000)
    value = env.evaluate_physical_coefficients(env.base_physical_coefficients)["average_gate_fidelity"]
    return config, env, value


@pytest.mark.parametrize("bins", [14, 140])
def test_grape_reference_matches_walsh_simulator(bins):
    config, env, expected = context()
    pulse = grape.PulseMap(env.base_physical_coefficients, config, bins)
    dynamics = grape.Dynamics(config, env.target, bins)
    value, _, _ = dynamics.objective(pulse.physical(np.zeros(pulse.shape)),
                                     env.noise_sample.detuning[None], np.ones(1), gradient=False)
    assert abs(value-expected) < 1e-11


def test_crab_reference_and_adjoint():
    config, env, expected = context()
    pulse = fourier.FourierPulse(env.base_physical_coefficients, config, 2, 17, 920000)
    settings = dict(rtol=1e-10, atol=1e-12, points_per_period=8)
    dynamics = fourier.ContinuousDynamics(config, env.target, pulse, settings)
    zero = np.zeros(pulse.shape)
    value, _ = dynamics.objective(zero, env.noise_sample.detuning[None], np.ones(1))
    assert abs(value-expected) < 1e-8
    coefficients = np.random.default_rng(2).normal(0, 0.001, pulse.shape)
    nodes, weights = np.zeros((1, 2)), np.ones(1)
    adjoint_value, gradient = fidelity_gradient(dynamics, coefficients, nodes, weights, fourier.PTM_CONTROLS)
    direct = dynamics.objective(coefficients, nodes, weights)[0]
    assert abs(adjoint_value-direct) < 1e-8
    direction = np.random.default_rng(3).normal(size=pulse.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-5
    finite_difference = (dynamics.objective(coefficients+eps*direction, nodes, weights)[0]
                         - dynamics.objective(coefficients-eps*direction, nodes, weights)[0])/(2*eps)
    assert abs(finite_difference - np.sum(gradient*direction)) < 2e-6
