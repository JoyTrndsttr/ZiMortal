"""Mine uncertain chi/peng choices for human review; never invent EV labels."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from zimortal.engine.tiles import tile_name

from .active import atomic_json
from .cashq import digest
from .online_rl import play
from .runtime import load_model


def select_cases(records, limit):
    candidates = [r for r in records if r.get("review_eligible")]
    candidates.sort(
        key=lambda r: (not r["different_from_parent"], r["policy_margin"], r["seed"], r["step"])
    )
    chosen, seen = [], set()
    # Diversify chi and peng, then unique seed/player/opponent histories.
    for category in ("peng_pass", "chi_bi_pass"):
        for row in candidates:
            key = (
                row["seed"],
                row["player"],
                hashlib.sha256(
                    json.dumps(row["history_prefix"], sort_keys=True).encode()
                ).hexdigest(),
            )
            if (
                row["category"] == category
                and key not in seen
                and sum(r["category"] == category for r in chosen) < limit // 2
            ):
                chosen.append(row)
                seen.add(key)
    for row in candidates:
        key = (
            row["seed"],
            row["player"],
            hashlib.sha256(json.dumps(row["history_prefix"], sort_keys=True).encode()).hexdigest(),
        )
        if key not in seen and len(chosen) < limit:
            chosen.append(row)
            seen.add(key)
    return chosen


def run(args):
    torch.set_num_threads(2)
    root, reports = Path(args.directory), Path(args.reports)
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model, args.device)
    parent = load_model(args.parent, args.device)
    opponents = [load_model(p, args.device) for p in args.opponents]
    specs = [
        (s, s % 3, seat, opponent)
        for s in range(args.start, args.start + args.seeds)
        for seat in range(3)
        for opponent in range(len(opponents))
    ]
    records = []
    _, games, questions = play(
        model, opponents, specs, device=args.device, slots=32, audit=records, audit_parent=parent
    )
    selected = select_cases(records, args.limit)
    atomic_json(root / "candidates.json", [r for r in records if r["review_eligible"]])
    from zimortal.web.server import build_review

    index = []
    lines = [
        "# 吃碰拆组人工审核",
        "",
        "概率接近不等于EV接近；静态组合损失不等于动作错误。",
        "本批只用于诊断，不作为训练标签或独立测试。下面第几步指所有动作的编号。",
        "",
    ]
    for i, record in enumerate(selected, 1):
        identifier = f"split-{i:03d}"
        record = {
            **record,
            "id": identifier,
            "model": args.model,
            "model_sha256": digest(args.model),
            "parent_model": args.parent,
            "opponent_pool": args.opponents,
        }
        atomic_json(root / f"{identifier}.json", record)
        payload = build_review(record)
        atomic_json(root / f"{identifier}-trace.json", payload)
        url = f"http://127.0.0.1:8765/?review={identifier}"
        index.append(
            {
                "id": identifier,
                "seed": record["seed"],
                "step": record["step"],
                "player": record["player"],
                "url": url,
                "category": record["category"],
            }
        )
        lines += [
            f"## {identifier}：seed {record['seed']}，玩家{record['player'] + 1}，第{record['step']}步",
            "",
            f"[打开原轨迹]({url})（定位在动作之前）。",
            "",
            "手牌：" + " ".join(tile_name(t) for t in record["hand"]),
            "",
            f"公开胡息{record['exposed_huxi']}；剩余牌堆{record['remaining_tiles']}张。",
            "",
        ]
        for option in sorted(record["options"], key=lambda o: -o["probability"]):
            action, structure = option["action"], option["structure"]
            label = {"chi": "吃", "peng": "碰", "pass": "过"}.get(action["kind"], action["kind"])
            group = "".join(tile_name(t) for t in action["chi"])
            if not group and action["tile"] is not None:
                group = tile_name(action["tile"])
            bi = "／".join("".join(tile_name(t) for t in g) for g in action["bi"])
            broken = "、".join(
                "".join(tile_name(t) for t in g) for g in structure["broken_patterns"]
            )
            lines.append(
                f"- {label}{group} 下比{bi or '无'}：概率{option['probability']:.1%}，"
                f"静态胡息潜力变化{structure['static_huxi_delta']:+d}；拆组{broken or '无'}。"
            )
        lines += [
            "",
            "请判断：更倾向哪一个方案，为什么；这是速度／胡息／名堂取舍，还是规则问题？",
            "",
        ]
    atomic_json(
        root / "index.json",
        {
            "cases": index,
            "completed_games": len(games),
            "excluded_rule_games": len(questions),
            "uncertain_records": sum(r["review_eligible"] for r in records),
            "model_sha256": digest(args.model),
            "start": args.start,
            "seeds": args.seeds,
        },
    )
    if questions:
        atomic_json(root / "rule-questions.json", questions)
    (reports / "human-review.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(reports / "human-review-index.json", index)
    print(
        json.dumps({"completed_games": len(games), "cases": index}, ensure_ascii=False), flush=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/generated/league-v1/stage-03/best-dev.pt")
    parser.add_argument("--parent", default="checkpoints/huxi-resnet.pt")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=[
            "checkpoints/huxi-resnet.pt",
            "checkpoints/scale100k-resnet.pt",
            "data/generated/online-rl-v2/best-dev.pt",
        ],
    )
    parser.add_argument("--directory", default="data/generated/human-review-v1")
    parser.add_argument("--reports", default="docs/training/league-v2")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--start", type=int, default=90000000)
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if args.seeds < 1 or args.limit < 2:
        parser.error("positive seed count and at least two cases required")
    run(args)


if __name__ == "__main__":
    main()
