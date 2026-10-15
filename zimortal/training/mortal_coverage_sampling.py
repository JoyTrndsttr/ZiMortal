"""Frozen, bounded exploration among parent-supported alternative actions."""

import numpy as np


def enhance_probability(p, parent, family, config):
    p = np.asarray(p, dtype=np.float64)
    if family not in ("discard", "chi/pass"):
        return p.copy()
    settings = config.get("coverage")
    if settings is None:
        return p.copy()
    mix, minimum, count = settings["mixture"], settings["minimum"], settings["alternatives"]
    if not 0 <= mix <= 1 or not 0 < minimum < 1 or count < 1:
        raise ValueError("invalid coverage configuration")
    candidates = [i for i in np.argsort(-p, kind="stable") if i != parent and p[i] >= minimum]
    candidates = candidates[:count]
    if not candidates:
        return p.copy()
    alternate = np.zeros_like(p)
    alternate[candidates] = 1 / len(candidates)
    return (1 - mix) * p + mix * alternate
