"""Own-decision n-step SARSA links and frozen hard-sync target network."""

import copy
import hashlib
from collections import defaultdict

import numpy as np
import torch

from .online_rl import pack


def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def future_indices(rows, n):
    """t+n within the same complete learner trajectory; -1 if termination precedes it."""
    if n < 1:
        raise ValueError("positive own-decision horizon required")
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(int(row["game_id"]), int(row["seat"]))].append(index)
    links = np.full(len(rows), -1, dtype=np.int64)
    for indices in grouped.values():
        indices.sort(key=lambda i: int(rows[i]["history_step"]))
        steps = [int(rows[i]["history_step"]) for i in indices]
        if len(set(steps)) != len(steps):
            raise ValueError("duplicate own-decision history step")
        cash = float(rows[indices[0]]["terminal_cash"])
        for pos, index in enumerate(indices):
            if (
                int(rows[index]["steps_to_terminal"]) != len(indices) - pos - 1
                or float(rows[index]["terminal_cash"]) != cash
            ):
                raise ValueError("incomplete or inconsistent learner trajectory")
            if pos + n < len(indices):
                links[index] = indices[pos + n]
    return links


class HardTarget:
    def __init__(self, online, sync_updates):
        if sync_updates < 1:
            raise ValueError("positive hard-sync frequency required")
        self.model = copy.deepcopy(online).eval().requires_grad_(False)
        self.interval = sync_updates
        self.updates = 0
        self.sync_history = [0]

    def advance(self, online):
        self.updates += 1
        if self.updates % self.interval == 0:
            self.model.load_state_dict(online.state_dict())
            self.model.eval()
            self.sync_history.append(self.updates)


def sarsa_target(rows, indices, links, target, *, n, device, gamma=1.0):
    """Sparse terminal reward, otherwise gamma^n Q_target(s[t+n], actual a[t+n])."""
    batch = [rows[int(i)] for i in indices]
    result = torch.tensor(
        [float(r["terminal_cash"]) * gamma ** int(r["steps_to_terminal"]) for r in batch],
        device=device,
    )
    positions = [j for j, i in enumerate(indices) if links[int(i)] >= 0]
    if positions:
        future = [rows[int(links[int(indices[j])])] for j in positions]
        x, a, mask = pack([r["features"] for r in future], [r["actions"] for r in future], device)
        chosen = torch.tensor([int(r["chosen"]) for r in future], device=device)
        if not mask.gather(1, chosen[:, None]).all():
            raise ValueError("bootstrap behavior action must be legal")
        with torch.no_grad():
            q = target.forward_q(x, a, mask)[0]
            boot = q.gather(1, chosen[:, None]).squeeze(-1)
        if not torch.isfinite(boot).all():
            raise RuntimeError("nonfinite SARSA bootstrap")
        result[positions] = gamma**n * boot
    return result.detach()
