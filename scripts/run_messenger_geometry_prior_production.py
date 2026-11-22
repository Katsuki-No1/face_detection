#!/usr/bin/env python3
"""Provide file-descriptor headroom for the authenticated production launcher.

Eight workers prefetch effective-batch64 windows with many per-target shared
tensors. Linux file_descriptor sharing can exhaust a 10240 descriptor limit
and fail while receiving SCM_RIGHTS ("received 0 items of ancdata"). Raise only
this process's soft resource limit before exec; preserve workers, RNG, batching,
training source identity and the original resume/checkpoint contract.
"""
import os
from pathlib import Path
import resource
import sys


def ensure_descriptor_limit(minimum=65536):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < minimum:
        if hard != resource.RLIM_INFINITY and hard < minimum:
            raise RuntimeError(f'Production DataLoader needs NOFILE >= {minimum}; hard limit is {hard}')
        resource.setrlimit(resource.RLIMIT_NOFILE, (minimum, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)


if __name__=='__main__':
    print(f'geometry-prior production NOFILE={ensure_descriptor_limit()}', flush=True)
    launcher=Path(__file__).with_name('train_messenger_geometry_prior.py')
    os.execv(sys.executable,[sys.executable,str(launcher),*sys.argv[1:]])
