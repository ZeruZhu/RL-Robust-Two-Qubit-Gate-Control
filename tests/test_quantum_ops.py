import numpy as np

from walsh_rl import quantum_ops as qops
from walsh_rl.controls import single_hamiltonian_to_polar_controls


def test_no_jump_superoperator_matches_unitary_propagation():
    hamiltonian = 0.2 * qops.Z + 0.7 * qops.X - 0.3 * qops.Y
    total_time = 0.4
    unitary = qops.propagate_unitary([hamiltonian], total_time)
    superoperator = qops.propagate_superoperator([hamiltonian], total_time, [])
    assert np.allclose(superoperator, qops.unitary_to_superoperator(unitary), atol=1e-12)


def test_single_qubit_polar_control_round_trip():
    original = 0.2 * qops.Z + 0.7 * (np.cos(-0.4) * qops.X + np.sin(-0.4) * qops.Y)
    alpha, gamma, phi = single_hamiltonian_to_polar_controls(original)
    reconstructed = alpha * qops.Z + gamma * (np.cos(phi) * qops.X + np.sin(phi) * qops.Y)
    assert np.allclose(reconstructed, original, atol=1e-12)


def test_channel_is_trace_preserving():
    generator = 0.3 * qops.Z1 + 0.7 * qops.EXCHANGE_XX_YY
    jumps = [np.sqrt(0.1) * np.kron(qops.SIGMA_MINUS, qops.I2)]
    channel = qops.propagate_superoperator([generator], 0.4, jumps)
    trace = np.eye(4).reshape(-1, order="F")
    np.testing.assert_allclose(trace @ channel, trace, atol=1e-12)
