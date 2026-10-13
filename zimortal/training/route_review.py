"""Fresh, replay-verified human route-Q cases; never cash training labels."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import ActionType, RuleClarificationRequired, RuleEngine
from zimortal.engine.tiles import tile_name
from zimortal.model.encoding import encode_action, encode_observation
from zimortal.model.routeq import plan_arrays
from zimortal.web.server import build_review

from .active import atomic_json
from .decision_audit import record
from .online_rl import digest
from .routeq import mine_observation, model_root
from .runtime import load_model, predict


def generate(model_path, directory, *, start=970000000, games=20, limit=6, first_id=101):
    torch.set_num_threads(2)
    model = load_model(model_path)
    if model.architecture != "routeq":
        raise ValueError("route review requires route Q")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    candidates, failures = [], []
    for i in range(games):
        seed = start + i
        try:
            mined = mine_observation(
                seed, model.parent, "cpu", ("chi", "peng", "hu", "discard")[i % 4]
            )
            if mined is None:
                continue
            obs, plans, family = mined
            probabilities, _ = predict(model.parent, obs)
            p = probabilities.softmax(-1).numpy()
            baseline = int(p.argmax())
            groups, contexts = plan_arrays(obs, plans)
            row = {
                "features": encode_observation(obs, "huxi"),
                "actions": np.stack([encode_action(a, obs.player) for a in obs.legal_actions]),
                "plans": groups,
                "contexts": contexts,
            }
            with torch.inference_mode():
                q, value, _ = model_root(model, row, "cpu")
            cash = q[0].numpy() * 100
            action_index, route_index = np.unravel_index(cash.argmax(), cash.shape)
            by_action = cash.max(-1)
            order = np.sort(by_action)
            gap = float(order[-1] - order[-2])
            case = record(
                obs,
                int(action_index),
                baseline,
                p,
                spec=(seed, seed % 3, obs.player, 0),
                step=len(obs.history) + 1,
            )
            hu_action = next((a for a in obs.legal_actions if a.kind == ActionType.HU), None)
            if hu_action is not None:
                engine = RuleEngine()
                state = engine.replay(engine.new_game(seed, seed % 3), obs.history)
                settlement = engine.step(state, hu_action).settlement
                case["hu_now"] = asdict(settlement)
                case["immediate_hu_cash"] = settlement.payments[obs.player]
            case.update(
                {
                    "history_prefix": [asdict(a) for a in obs.history],
                    "model": "trajectory-v4-5000 parent policy (saved prefix)",
                    "opponent_pool": ["trajectory-v4-5000 parent policy (saved prefix)"],
                    "analysis_model": str(model_path),
                    "analysis_model_sha256": digest(model_path),
                    "diagnostic_only": True,
                    "human_feedback_is_cash_label": False,
                    "route_comparison": {
                        "family": family,
                        "parent_action": baseline,
                        "q_action": int(action_index),
                        "q_route": int(route_index),
                        "action_route_q_cash": cash.tolist(),
                        "route_value_cash": (value[0].numpy() * 100).tolist(),
                        "action_gap_cash": gap,
                        "plans": [
                            {
                                "groups": plan["groups"],
                                "huxi_lower": round(float(contexts[r, -8]) * 60),
                                "huxi_upper": round(float(contexts[r, -7]) * 60),
                                "missing": [round(float(v) * 4) for v in contexts[r, 20:40]],
                            }
                            for r, plan in enumerate(plans)
                        ],
                        "meaning": "试验模型对指定路线跟随策略的现金预测；不是最优EV或真实收益标签",
                    },
                }
            )
            candidates.append(case)
        except RuleClarificationRequired as exc:
            failures.append({"seed": seed, "error": str(exc)})
    ranked = sorted(
        candidates,
        key=lambda c: (not c["different_from_parent"], c["route_comparison"]["action_gap_cash"]),
    )
    selected, seen = [], set()
    for family in ("peng", "chi", "hu", "discard"):
        found = next((c for c in ranked if c["route_comparison"]["family"] == family), None)
        if found is not None:
            selected.append(found)
            seen.add(found["seed"])
    for case in ranked:
        if len(selected) >= limit:
            break
        if case["seed"] not in seen:
            selected.append(case)
            seen.add(case["seed"])
    selected = selected[:limit]
    index, lines = (
        [],
        ["# 路线Q人工审核", "", "新审核种子，不来自训练／开发／独立检验集。所有预测仅供诊断。", ""],
    )
    for offset, case in enumerate(selected):
        identifier = f"split-{first_id + offset:03d}"
        if (root / f"{identifier}.json").exists() or (root / f"{identifier}-trace.json").exists():
            raise ValueError(f"review id already exists: {identifier}")
        case["id"] = identifier
        payload = build_review(case)
        atomic_json(root / f"{identifier}.json", case)
        atomic_json(root / f"{identifier}-trace.json", payload)
        url = f"http://127.0.0.1:8765/?review={identifier}"
        item = {
            "id": identifier,
            "seed": case["seed"],
            "player": case["player"] + 1,
            "step": case["step"],
            "url": url,
            "family": case["route_comparison"]["family"],
            "parent": case["parent"],
            "q": case["chosen"],
            "different_from_parent": case["different_from_parent"],
        }
        index.append(item)

        def name(a):
            tile = tile_name(a["tile"]) if a["tile"] is not None else ""
            groups = "".join(tile_name(t) for t in a["chi"])
            bi = "／".join("".join(tile_name(t) for t in g) for g in a["bi"])
            return f"{a['kind']} {tile} {groups}" + (f" 下比{bi}" if bi else "")

        lines += [
            f"## {identifier}：seed {case['seed']}，玩家{case['player'] + 1}，第{case['step']}步",
            "",
            f"[打开审核]({url})",
            "",
            f"父策略：{name(case['parent'])}；路线Q建议：{name(case['chosen'])}。",
            "",
            "手牌：" + " ".join(tile_name(t) for t in case["hand"]),
            "",
        ]
    atomic_json(
        root / "routeq-review-index.json",
        {
            "cases": index,
            "model_sha256": digest(model_path),
            "start": start,
            "games": games,
            "failed_rule_games": failures,
            "diagnostic_only": True,
        },
    )
    (root / "routeq-review.md").write_text("\n".join(lines), encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/generated/routeq-v1/best-dev.pt")
    parser.add_argument("--directory", default="data/generated/human-review-v1")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--start", type=int, default=970000000)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--first-id", type=int, default=101)
    args = parser.parse_args()
    if min(args.games, args.limit) < 1 or not 0 <= args.first_id <= 999 - args.limit + 1:
        parser.error("invalid case budget or id range")
    import json

    print(
        json.dumps(
            generate(
                args.model,
                args.directory,
                start=args.start,
                games=args.games,
                limit=args.limit,
                first_id=args.first_id,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
