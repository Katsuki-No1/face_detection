"""Opt-in AMP calibration, separate from weight updates and gate verification."""
from contextlib import contextmanager
import math
import random
import time

import numpy as np
import torch


def calibration_policy(config):
    raw = config.get('train', {}).get('amp_calibration', {})
    if not raw.get('enabled', False):
        return None
    if not str(config.get('model', {}).get('type', '')).startswith('dinov3_codino_official'):
        raise ValueError('Experimental AMP calibration is restricted to official-SFP')
    policy = dict(enabled=True, status='experimental',
        initial_scale=float(raw.get('initial_scale', 512)),
        minimum_scale=float(raw.get('minimum_scale', 1)),
        max_windows=int(raw.get('max_windows', 16)),
        required_finite_windows=int(raw.get('required_finite_windows', 2)))
    if (not math.isfinite(policy['initial_scale']) or not math.isfinite(policy['minimum_scale'])
            or not 0 < policy['minimum_scale'] <= policy['initial_scale']
            or not 1 <= policy['required_finite_windows'] <= policy['max_windows'] <= 64):
        raise ValueError('Invalid bounded AMP calibration policy')
    return policy


def gradients_finite(parameters):
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)


@contextmanager
def preserve_rng_and_buffers(model):
    """Calibration must not advance production sampling or normalization state."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)


def calibrate_gradients(*, model, optimizer, scaler, policy, backward_window, report):
    """Never call optimizer/scaler.step. Overflow is allowed only in this phase.

    backward_window(index) must accumulate one real, fixed-size window, reject
    nonfinite forward loss, and use scaler.scale. report is a diagnostic writer,
    never a gate success marker. Strict finite verification follows this call.
    """
    evidence = dict(status='calibrating', policy=policy, optimizer_updates=0,
                    windows=[], elapsed_seconds=0., final_scale=scaler.get_scale())
    start = time.monotonic()
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    consecutive = 0
    try:
        for index in range(policy['max_windows']):
            optimizer.zero_grad(set_to_none=True)
            scale = scaler.get_scale()
            backward_window(index)
            scaler.unscale_(optimizer)
            finite = gradients_finite(parameters)
            # update(new_scale) resets per-optimizer unscale state without any
            # optimizer update, EMA update, LR step or training-success credit.
            next_scale = scale if finite else scale * scaler.get_backoff_factor()
            evidence['windows'].append(dict(index=index, scale=scale, finite=finite))
            if not finite and next_scale < policy['minimum_scale']:
                raise RuntimeError('AMP calibration reached minimum scale without finite gradients')
            scaler.update(new_scale=next_scale)
            consecutive = consecutive + 1 if finite else 0
            evidence['final_scale'] = scaler.get_scale()
            evidence['elapsed_seconds'] = time.monotonic() - start
            report(evidence)
            if consecutive >= policy['required_finite_windows']:
                evidence['status'] = 'calibrated'
                report(evidence)
                return evidence
        raise RuntimeError('AMP calibration exhausted its bounded windows')
    except BaseException as exc:
        evidence.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                        elapsed_seconds=time.monotonic() - start)
        report(evidence)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
