"""Fourier pulse and continuous Lindblad propagation for CRAB."""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from .grape import ALPHA, EXCHANGE, GAMMA, PHASE, D_L, SEGMENTS, CONTROL_NAMES, Dynamics, PulseMap, gaussian_ensemble, qops
PAULIS = [np.kron(a, b) for a in (qops.I2, qops.X, qops.Y, qops.Z)
          for b in (qops.I2, qops.X, qops.Y, qops.Z)]
PTM_BASIS = np.column_stack([p.reshape(-1, order="F") / 2 for p in PAULIS])

def real_ptm(matrix):
    transformed = PTM_BASIS.conj().T @ matrix @ PTM_BASIS
    if np.max(np.abs(transformed.imag)) > 1e-7:
        raise ValueError("generator/channel is not real in the Hermitian Pauli basis")
    return np.ascontiguousarray(transformed.real)


PTM_CONTROLS = np.array([real_ptm(generator) for generator in D_L])


class FourierPulse:
    """Analytically bounded Fourier corrections, with no time-point clipping."""

    def __init__(self, base, env_config, basis_size, basis_seed, context_seed):
        self.reference = PulseMap(base, env_config, 140)
        if np.max(np.abs(np.asarray(base)[1])) > 1e-12:
            raise ValueError("this benchmark requires the constant cached KAK initializer")
        self.k = int(basis_size)
        if self.k < 1:
            raise ValueError("basis_size must be positive")
        self.shape = (27, 1 + 2 * self.k)
        self.dimension = int(np.prod(self.shape))
        self.scales = self.reference.scales
        self.base = np.concatenate((base[0, ALPHA], base[0, EXCHANGE],
                                    self.reference.base_x[0], self.reference.base_y[0]))
        # Draw each channel separately so K=2 is nested exactly inside K=4.
        shifts = np.array([np.random.default_rng(np.random.SeedSequence(
            [int(basis_seed), int(context_seed), channel])).uniform(-0.5, 0.5, self.k)
            for channel in range(27)])
        self.omega = 2 * np.pi * np.arange(1, self.k + 1)[None] * (1 + shifts)
        self.cos_half = np.abs(np.cos(self.omega / 4))
        self.sin_half = np.abs(np.sin(self.omega / 4))

    def pair_envelopes(self, coefficients):
        radius = np.hypot(coefficients[:, 1:1+self.k], coefficients[:, 1+self.k:])
        return np.sum(radius * self.cos_half, axis=1), np.sum(radius * self.sin_half, axis=1)

    def project(self, theta):
        coefficients = np.asarray(theta, dtype=float).reshape(self.shape).copy()
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("nonfinite Fourier coefficients")
        even, odd = self.pair_envelopes(coefficients)
        factors = np.maximum(1.0, np.maximum(np.abs(coefficients[:, 0]) + even, odd))
        coefficients /= factors[:, None]
        even, odd = self.pair_envelopes(coefficients)
        x, y = np.arange(11, 19), np.arange(19, 27)
        dc = np.column_stack((coefficients[x, 0], coefficients[y, 0])) * self.scales[x, None]
        oscillation = np.column_stack((even[x], even[y])) * self.scales[x, None]
        baseline = np.column_stack((self.base[x], self.base[y]))
        odd_bound = np.hypot(odd[x], odd[y]) * self.scales[x]
        limits = self.reference.gamma_limits
        upper = np.minimum(1.0, limits[1] / np.maximum(odd_bound, 1e-300))

        def feasible(scale):
            bound = np.linalg.norm(np.abs(baseline + scale[:, None]*dc)
                                   + scale[:, None]*oscillation, axis=1)
            return bound <= limits[0]

        valid = feasible(upper)
        low = np.where(valid, upper, 0.0)
        high = upper.copy()
        # Convex envelope sublevel sets intersect each ray in an interval.
        for _ in range(48):
            mid = (low + high) / 2
            good = feasible(mid)
            low = np.where(good, mid, low)
            high = np.where(good, high, mid)
        scale = np.where(valid, upper, low)
        coefficients[x] *= scale[:, None]
        coefficients[y] *= scale[:, None]
        return coefficients

    def channel_values(self, coefficients, times):
        times = np.asarray(times)
        phases = times[..., None, None] * self.omega
        residual = coefficients[:, 0] + np.sum(
            coefficients[:, 1:1+self.k]*np.sin(phases)
            + coefficients[:, 1+self.k:]*np.cos(phases), axis=-1)
        return self.base + residual*self.scales

    @staticmethod
    def segment_channels(segment):
        if segment % 2:
            return np.array([8 + segment//2]), np.array([6])
        local = segment//2
        indices = np.array([2*local, 11+2*local, 19+2*local,
                            2*local+1, 12+2*local, 20+2*local])
        return indices, np.arange(6)

    def physical(self, coefficients, segment, times):
        values = self.channel_values(coefficients, times)
        output = np.zeros(values.shape[:-1] + (7,))
        channels, controls = self.segment_channels(segment)
        output[..., controls] = values[..., channels]
        return output


class ContinuousDynamics:
    def __init__(self, env_config, target, pulse, settings):
        reference = Dynamics(env_config, target, 140)
        self.pulse = pulse
        self.segment_time = float(env_config["segment_time"])
        self.dissipator = real_ptm(reference.dissipator)
        self.target = real_ptm(reference.target_super)
        self.rtol = float(settings.get("rtol", 1e-8))
        self.atol = float(settings.get("atol", 1e-10))
        self.points_per_period = float(settings.get("points_per_period", 8))
        self.propagations = 0
        self.rhs_calls = 0

    def channels(self, coefficients, detunings):
        detunings = np.asarray(detunings, dtype=float).reshape(-1, 2)
        count = len(detunings)
        state = np.broadcast_to(np.eye(16), (count, 16, 16)).copy()
        drift = self.segment_time * (self.dissipator[None]
            + detunings[:, 0, None, None]*PTM_CONTROLS[0]
            + detunings[:, 1, None, None]*PTM_CONTROLS[3])
        for segment in range(7):
            indices, controls = self.pulse.segment_channels(segment)
            omega = self.pulse.omega[indices]
            k = self.pulse.k
            base = self.pulse.base[indices] + self.pulse.scales[indices]*coefficients[indices, 0]
            sine = self.pulse.scales[indices, None]*coefficients[indices, 1:1+k]
            cosine = self.pulse.scales[indices, None]*coefficients[indices, 1+k:]
            generators = self.segment_time*PTM_CONTROLS[controls]

            def rhs(tau, flat):
                amplitudes = base + np.sum(sine*np.sin(omega*tau) + cosine*np.cos(omega*tau), axis=1)
                generator = drift + np.einsum("c,cab->ab", amplitudes, generators)
                return (generator @ flat.reshape(count, 16, 16)).ravel()

            max_step = min(1.0, 2*np.pi/(self.points_per_period*np.max(omega)))
            solved = solve_ivp(rhs, (0.0, 1.0), state.ravel(), method="DOP853",
                               rtol=self.rtol, atol=self.atol, max_step=max_step,
                               t_eval=[1.0])
            self.rhs_calls += solved.nfev
            if not solved.success or solved.y.shape[1] != 1:
                raise RuntimeError(f"continuous propagation failed: {solved.message}")
            state = solved.y[:, -1].reshape(count, 16, 16)
        self.propagations += count
        if not np.all(np.isfinite(state)) or np.max(np.abs(state[:, 0]-np.eye(16)[0])) > 1e-7:
            raise FloatingPointError("nonfinite or non-trace-preserving continuous channel")
        return state

    def objective(self, coefficients, detunings, weights):
        channels = self.channels(coefficients, detunings)
        values = (np.einsum("ab,nab->n", self.target, channels) + 4)/20
        if np.any(values < 0.2-1e-7) or np.any(values > 1+1e-7):
            raise FloatingPointError("invalid average gate fidelity")
        mean = float(weights @ values)
        std = float(np.sqrt(weights @ (values-mean)**2))
        return mean, std
