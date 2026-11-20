#!/usr/bin/env python
"""Materialize paired Haar/KAK tasks with independent signed, binned detunings."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from walsh_rl.config import load_config
from walsh_rl.envs.two_qubit_env import TwoQubitWalshEnv
from walsh_rl.noise import NoiseSample
from walsh_rl.training.fidelity_only_multi_agent import (
    _environment_signature, save_context_cache,
)


def generate(config, output, reference_cache, *, bins=40, per_bin=400, seed_start=940000):
    if any((output/name).exists() for name in ('test.npz', 'metadata.json', 'binning.json')):
        raise FileExistsError(f'Refusing to overwrite generated context cache: {output}')
    reference = json.loads((reference_cache/'metadata.json').read_text())
    assert _environment_signature(config) == reference['environment_signature']
    assert config['fidelity_only_eight_agent']['residual_projection_observation_scale'] == 5e6
    env = TwoQubitWalshEnv(config)
    rows = []
    width = 5e5
    try:
        for index in range(bins*per_bin):
            seed = seed_start+index
            k = index//per_bin
            env.reset(seed=seed)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 20261014]))
            mags = rng.uniform(k*width, (k+1)*width, 2)
            signs = rng.choice([-1., 1.], 2)
            old = env.noise_sample
            env.noise_sample = NoiseSample(signs*mags, old.relaxation_rates.copy(), old.dephasing_rates.copy())
            analytic = env.base_physical_coefficients.copy()
            info = env.evaluate_physical_coefficients(analytic)
            rows.append(dict(seed=seed, target=env.target.copy(), noise=env.noise_sample.as_vector().copy(),
                             analytic_coefficients=analytic, analytic_agf=float(info['average_gate_fidelity']),
                             alpha_commands=np.zeros((2, 8)), g_commands=np.zeros((2, 3)),
                             x_commands=np.zeros((2, 8)), y_commands=np.zeros((2, 8)),
                             current_coefficients=analytic.copy(), current_superoperator=env.current_operator.copy(),
                             current_agf=float(info['average_gate_fidelity']), cumulative_pulse_evaluations=1))
            if (index+1) % 400 == 0:
                print(f'Generated {index+1}/{bins*per_bin}', flush=True)
    finally:
        env.close()
    save_context_cache(output, config, {'test': rows}, stage_provenance='analytic', source_cache_dir=str(reference_cache))
    np.save(output/'bin_index.npy', np.repeat(np.arange(bins), per_bin))
    (output/'binning.json').write_text(json.dumps(dict(num_bins=bins, tasks_per_bin=per_bin, width_rad_s=width,
        detuning_std=5e6, seed_start=seed_start, seed_stop=seed_start+bins*per_bin-1,
        rng='default_rng(SeedSequence([seed, 20261014]))', draw_order=['two uniform magnitudes', 'two independent signs'],
        intervals='left-closed, right-open'), indent=2)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--reference-cache', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    generate(load_config(args.config), args.output_dir, args.reference_cache)

if __name__ == '__main__':
    main()
