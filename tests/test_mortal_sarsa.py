import numpy as np
import pytest
import torch

from zimortal.model.encoding import ACTION_DIM
from zimortal.training.mortal_credit import improvement
from zimortal.training.mortal_sarsa import HardTarget, future_indices, sarsa_target


def trajectory(game, seat, length, cash):
    return [
        {
            "game_id": game,
            "seat": seat,
            "history_step": 2 + i * 7,
            "steps_to_terminal": length - i - 1,
            "terminal_cash": cash,
            "features": np.full((55, 20), i, np.float32),
            "actions": np.zeros((2 + (i % 2), ACTION_DIM), np.float32),
            "chosen": 1,
        }
        for i in range(length)
    ]


@pytest.mark.parametrize("n", [3, 5])
def test_own_decision_links_ignore_history_gaps_and_do_not_cross_games_or_seats(n):
    a, b = trajectory(10, 0, 8, 2.0), trajectory(10, 1, 6, -1.0)
    rows = list(reversed(a + b + trajectory(11, 0, 8, 1.0)))
    links = future_indices(rows, n)
    for i, r in enumerate(rows):
        if r["steps_to_terminal"] < n:
            assert links[i] == -1
        else:
            future = rows[links[i]]
            assert (future["game_id"], future["seat"]) == (r["game_id"], r["seat"])
            assert future["history_step"] - r["history_step"] == n * 7
    broken = [dict(r) for r in a]
    broken[0]["steps_to_terminal"] -= 1
    with pytest.raises(ValueError, match="incomplete"):
        future_indices(broken, n)
    broken = [dict(r) for r in a]
    broken[1]["history_step"] = broken[0]["history_step"]
    with pytest.raises(ValueError, match="duplicate"):
        future_indices(broken, n)


def test_bootstrap_actual_behavior_action_detached_and_exact_terminal_boundary():
    class Target(torch.nn.Module):
        def forward_q(self, x, a, mask):
            q = torch.full(mask.shape, 999.0, device=x.device, requires_grad=True)
            chosen = x[:, 0, 0]
            return (
                torch.stack([q[:, 0], chosen], -1).masked_fill(~mask[:, :2], -torch.inf),
                None,
                {},
            )

    rows = trajectory(10, 0, 7, 2.0)
    # Constant two candidates for this stub; production supports ragged actions.
    for r in rows:
        r["actions"] = r["actions"][:2]
    links = future_indices(rows, 3)
    target = sarsa_target(rows, [0, 3, 4, 6], links, Target(), n=3, device="cpu", gamma=0.5)
    assert target.tolist() == pytest.approx([0.5**3 * 3, 0.5**3 * 6, 0.5**2 * 2, 2.0])
    assert not target.requires_grad
    # At remaining==n, bootstrap final recorded action, not terminal reward.
    assert links[3] == 6 and links[4] == -1


def test_hard_target_is_separate_frozen_and_syncs_only_at_update_boundary():
    online = torch.nn.Linear(2, 1)
    target = HardTarget(online, 2)
    before = target.model.weight.clone()
    with torch.no_grad():
        online.weight.add_(3)
    assert torch.equal(target.model.weight, before)
    target.advance(online)
    assert torch.equal(target.model.weight, before)
    target.advance(online)
    assert torch.equal(target.model.weight, online.weight)
    assert target.model.weight.data_ptr() != online.weight.data_ptr()
    assert all(not p.requires_grad for p in target.model.parameters())
    assert not target.model.training and target.sync_history == [0, 2]
    with pytest.raises(ValueError):
        HardTarget(online, 0)


def test_screening_trigger_requires_cash_signal_and_key_families_on_both_splits():
    def result(corr, chi, peng):
        return {
            s: {
                "correlation": corr,
                "rmse": 0.7,
                "per_family": {
                    "chi/pass": {"correlation": chi},
                    "peng/pass": {"correlation": peng},
                },
            }
            for s in ("selected_dev", "heldout")
        }

    mc = result(0.16, 0.11, 0.07)
    assert improvement(result(0.21, 0.14, 0.10), mc)["passed"]
    assert not improvement(result(0.21, 0.04, 0.07), mc)["passed"]
    candidate = result(0.21, 0.14, 0.10)
    candidate["heldout"]["correlation"] = 0.16
    assert not improvement(candidate, mc)["passed"]


def test_training_path_matches_initialization_batches_and_saves_synced_target(tmp_path):
    from zimortal.model.mortal_q import MortalQNet
    from zimortal.training.mortal_offline import fit

    model = MortalQNet(width=8, latent=32, blocks=1)
    rows = trajectory(10, 0, 7, 2.0)
    for r in rows:
        r.update(family="discard", parent_greedy=0)
    mc = fit(model, rows, rows, rows, tmp_path / "mc", epochs=1, batch_size=7)
    td = fit(
        model,
        rows,
        rows,
        rows,
        tmp_path / "td",
        epochs=1,
        batch_size=7,
        n_step=3,
        target_sync_updates=1,
    )
    assert mc["initial_parameter_sha256"] == td["initial_parameter_sha256"]
    assert mc["epochs"][0]["batch_order_sha256"] == td["epochs"][0]["batch_order_sha256"]
    assert td["epochs"][0]["bootstrap_rows"] == 4
    assert td["epochs"][0]["terminal_rows"] == 3
    saved = torch.load(tmp_path / "td/best-optimizer.pt", weights_only=True)
    online = torch.load(tmp_path / "td/best-dev.pt", weights_only=True)
    assert saved["target_updates"] == 1 and saved["target_sync_history"] == [0, 1]
    assert all(torch.equal(v, online["state_dict"][k]) for k, v in saved["target_network"].items())


def test_screening_replay_sink_and_per_step_audit_reject_wrong_cash(tmp_path):
    import gzip
    import json

    from zimortal.model.network import PolicyValueNet
    from zimortal.training.mortal_credit import audit_screening
    from zimortal.training.online_rl import play

    model = PolicyValueNet(width=8, feature_version="huxi").eval()
    with gzip.open(tmp_path / "screen-mc-replays.jsonl.gz", "wt") as file:
        _, games, errors = play(
            model,
            [model],
            [(118, 1, 0, 0), (119, 2, 1, 0)],
            slots=2,
            replay_sink=lambda e: file.write(json.dumps(e) + "\n"),
        )
    assert not errors and len(games) == 2
    path = tmp_path / "screening-outcomes.json"
    path.write_text(json.dumps({"mc": games}))
    report = audit_screening(tmp_path)
    assert report["games"] == 2 and report["steps"] == sum(r["steps"] for r in games)
    assert report["per_step_conservation_verified"]
    games[0]["cash"] += 5
    path.write_text(json.dumps({"mc": games}))
    with pytest.raises(RuntimeError, match="cash mismatch"):
        audit_screening(tmp_path)
