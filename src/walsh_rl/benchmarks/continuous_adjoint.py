"""Continuous adjoint copied verbatim from the validated 20261014 optimizer diagnostic.

The projection pullback uses central differences of the algebraic constraint map;
physical propagation and its gradient use continuous DOP853 integration.
"""
import numpy as np
from scipy.integrate import solve_ivp


def fidelity_gradient(dynamics, coefficients, nodes, weights, generators):
    """Continuous Lindblad adjoint; derivative w.r.t. projected coefficients."""
    pulse = dynamics.pulse
    n = len(nodes)
    size = n * 256
    drift = dynamics.segment_time * (
        dynamics.dissipator[None]
        + nodes[:, 0, None, None] * generators[0]
        + nodes[:, 1, None, None] * generators[3])
    state = np.broadcast_to(np.eye(16), (n, 16, 16)).copy()
    forwards = []
    definitions = []
    for segment in range(7):
        indices, controls = pulse.segment_channels(segment)
        omega = pulse.omega[indices]
        scaled_generators = dynamics.segment_time * generators[controls]
        scale = pulse.scales[indices]
        base = pulse.base[indices]
        local_coefficients = coefficients[indices]

        # Capture each segment's parameters for the later backward solve.
        def definition(t, omega=omega, mats=scaled_generators, scale=scale,
                       base=base, coeff=local_coefficients):
            basis = np.concatenate((np.ones((len(omega), 1)),
                                    np.sin(omega*t), np.cos(omega*t)), axis=1)
            amplitude = base + scale * np.sum(coeff*basis, axis=1)
            matrix = drift + np.einsum("c,cab->ab", amplitude, mats)
            return matrix, basis, mats, scale

        max_step = min(1., 2*np.pi/(dynamics.points_per_period*np.max(omega)))
        solved = solve_ivp(lambda t, y: (definition(t)[0] @ y.reshape(n, 16, 16)).ravel(),
                           (0., 1.), state.ravel(), method="DOP853", dense_output=True,
                           rtol=dynamics.rtol, atol=dynamics.atol, max_step=max_step)
        if not solved.success:
            raise RuntimeError(solved.message)
        state = solved.y[:, -1].reshape(n, 16, 16)
        forwards.append(solved.sol)
        definitions.append((definition, max_step, indices))
    values = (np.einsum("ab,nab->n", dynamics.target, state)+4)/20
    costate = weights[:, None, None] * dynamics.target[None]/20
    gradient = np.zeros(pulse.shape)
    for segment in reversed(range(7)):
        definition, max_step, indices = definitions[segment]
        forward = forwards[segment]

        def backward(t, y):
            matrix, basis, mats, scale = definition(t)
            lam = y[:size].reshape(n, 16, 16)
            channel = forward(t).reshape(n, 16, 16)
            overlap = np.sum(lam @ channel.transpose(0, 2, 1), axis=0)
            sensitivity = np.einsum("ab,cab->c", overlap, mats) * scale
            # Integrate from 1 to 0: minus sign yields the positive integral.
            return np.concatenate(((-matrix.transpose(0, 2, 1) @ lam).ravel(),
                                   (-sensitivity[:, None]*basis).ravel()))

        solved = solve_ivp(backward, (1., 0.),
                           np.concatenate((costate.ravel(), np.zeros(len(indices)*pulse.shape[1]))),
                           method="DOP853", rtol=dynamics.rtol, atol=dynamics.atol,
                           max_step=max_step)
        if not solved.success:
            raise RuntimeError(solved.message)
        costate = solved.y[:size, -1].reshape(n, 16, 16)
        gradient[indices] = solved.y[size:, -1].reshape(len(indices), -1)
    return float(weights @ values), gradient


def projection_pullback(pulse, theta, gradient, epsilon=1e-6):
    """Differentiate the original envelope map, not an unconstrained surrogate.

    Only this cheap algebraic map is finite-differenced. At its kinks the
    symmetric directional estimates are not a unique classical derivative.
    """
    result = np.empty(pulse.dimension)
    for j in range(pulse.dimension):
        displacement = np.zeros(pulse.dimension)
        displacement[j] = epsilon
        difference = pulse.project(theta+displacement)-pulse.project(theta-displacement)
        result[j] = np.sum(gradient*difference)/(2*epsilon)
    return result
