"""Export the two existing rare test PASS traces; no games or labels generated."""

import gzip
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import GameState, RuleEngine
from zimortal.model.encoding import encode_action, encode_observation
from zimortal.training.decision_audit import record
from zimortal.training.mortal_data import load_rows
from zimortal.training.online_rl import pack
from zimortal.training.runtime import load_model

from .server import build_review, snapshot


def export():
    torch.set_num_threads(2)
    data = Path("data/generated/mortal-v2/data")
    manifest = json.loads((data / "manifest.json").read_text())
    selected = {(1020001300, 55): "split-901", (1020006920, 45): "split-902"}
    rows = {
        (int(r["seed"]), int(r["history_step"])): r
        for r in load_rows(data, manifest)
        if (int(r["seed"]), int(r["history_step"])) in selected
    }
    model = load_model("data/generated/mortal-faithful-v1/faithful-mc/best-dev.pt")
    engine = RuleEngine()
    output = Path("data/generated/human-review-v1")
    output.mkdir(parents=True, exist_ok=True)
    exported = []
    for shard in manifest["shards"]:
        games = json.loads(gzip.decompress((data / shard["replays"]).read_bytes()))
        for game in games:
            targets = [k for k in selected if k[0] == game["seed"]]
            if not targets:
                continue
            end = GameState.deserialize(game["state"])
            for key in targets:
                row = rows[key]
                step = key[1]
                state = engine.new_game(game["seed"], game["dealer"])
                for action in end.history[:step]:
                    state = engine.step(state, action)
                obs = engine.observation(state, game["seat"])
                actions = obs.legal_actions
                encoded = np.stack([encode_action(a, obs.player) for a in actions])
                if not np.array_equal(
                    encode_observation(obs, "huxi"), row["features"]
                ) or not np.array_equal(encoded, row["actions"]):
                    raise ValueError("saved visible test input differs")
                if (
                    actions[int(row["chosen"])] != end.history[step]
                    or end.history[step].kind.value != "pass"
                ):
                    raise ValueError("saved action is not the expected PASS")
                with torch.inference_mode():
                    q = (
                        model.forward_q(*pack([row["features"]], [encoded], "cpu"))[0][0].numpy()
                        * 100
                    )
                case = record(
                    obs,
                    int(row["chosen"]),
                    int(row["parent_greedy"]),
                    row["behavior_probabilities"],
                    spec=(game["seed"], game["dealer"], game["seat"], 0),
                    step=step + 1,
                )
                hu = next(a for a in actions if a.kind.value == "hu")
                won = engine.step(state.clone(), hu)
                immediate = won.settlement.payments[obs.player]
                terminal = end.settlement.payments[obs.player]
                case.update(
                    id=selected[key],
                    history_prefix=[asdict(a) for a in state.history],
                    model="原10k行为策略（冻结parent按概率采样）",
                    learner_seat=game["seat"],
                    opponent_pool=["冻结baseline stage-02 greedy"],
                    opponent=0,
                    offline_test={
                        "description": "这次过胡是原行为策略2%的探索采样，不是faithful-MC的greedy选择。可向后播放原始完整轨迹。",
                        "terminal_cash": terminal,
                        "immediate_hu_cash": immediate,
                    },
                )
                for option, value in zip(case["options"], q, strict=True):
                    option["cash_q"] = float(value)
                trace = build_review(case)
                trace["start_frame"] = step
                for index, action in enumerate(end.history[step:], step + 1):
                    state = engine.step(state, action)
                    state.validate()
                    shown = asdict(action)
                    if action.kind.value == "draw" and state.pending:
                        shown["tile"] = state.pending.tile
                    trace["frames"].append(
                        snapshot(engine, state, include_huxi=False)
                        | {
                            "index": index,
                            "action": shown,
                            "note": "原10k test完整轨迹",
                            "verified": True,
                        }
                    )
                if state.serialize() != end.serialize():
                    raise ValueError("full test replay differs")
                trace["actions"] = len(end.history)
                trace["frames"][step]["note"] = (
                    f"待查看第{step + 1}步：原行为策略以2%概率过胡。现在胡净收益{immediate}元，原轨迹终局+{terminal}元。"
                )
                path = output / f"{case['id']}-trace.json"
                path.write_text(json.dumps(trace, ensure_ascii=False) + "\n")
                exported.append(
                    {
                        "id": case["id"],
                        "seed": game["seed"],
                        "step": step + 1,
                        "immediate_hu_cash": immediate,
                        "terminal_cash": terminal,
                    }
                )
    if len(exported) != 2:
        raise ValueError("missing test reviews")
    print(json.dumps(exported, ensure_ascii=False))


if __name__ == "__main__":
    export()
