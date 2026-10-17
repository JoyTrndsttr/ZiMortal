"""Read saved benchmark results only; no simulator, models, or training."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from zimortal.training.active import atomic_json
from zimortal.training.counterfactual_rollout import QUOTAS, SEED, ranking
from zimortal.training.mortal_credit_diagnostics import summary
from zimortal.training.online_rl import digest


def cluster_difference(roots, a, b, field, *, count=10000):
    groups = sorted({r["locator"][4] for r in roots})
    sums, nums = [], []
    for group in groups:
        values = [
            r["ranking"][a][field] - r["ranking"][b][field]
            for r in roots
            if r["locator"][4] == group
            and r["ranking"][a][field] is not None
            and r["ranking"][b][field] is not None
        ]
        sums.append(sum(values))
        nums.append(len(values))
    if not groups or not sum(nums):
        return {"mean": None, "ci95": None}
    sums, nums = np.asarray(sums), np.asarray(nums)
    sample = np.random.default_rng(SEED).integers(len(groups), size=(count, len(groups)))
    denominator = nums[sample].sum(1)
    valid = denominator > 0
    means = sums[sample].sum(1)[valid] / denominator[valid]
    return {
        "mean": float(sums.sum() / nums.sum()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def run(directory, figure=None):
    root = Path(directory)
    result = json.loads((root / "result.json").read_text())
    assert result["status"] == "complete_diagnostic_stop"
    manifest = json.loads((root / "locked-state-manifest.json").read_text())
    roots = []
    outcome_map = {}
    for e in manifest:
        with gzip.open(root / f"{e['id']}-rollouts.json.gz", "rt") as f:
            r = json.load(f)
        if "result" in r:
            with gzip.open(root / e["file"], "rt") as file:
                locked = json.load(file)
            vectors = {"baseline": locked["predictions"]["baseline"]}
            for checkpoint in ("initial", "long10", "A10", "B10"):
                vectors[f"{checkpoint}/raw"] = locked["predictions"][checkpoint]["q"]
                vectors[f"{checkpoint}/anchor"] = locked["predictions"][checkpoint]["anchor"]
            for strategy, scores in vectors.items():
                recalculated = ranking(scores, r["result"]["ev"], r["outcomes"])
                stored = r["result"]["ranking"][strategy]
                assert recalculated.keys() == stored.keys()
                for field, value in recalculated.items():
                    if value is None:
                        assert stored[field] is None
                    else:
                        np.testing.assert_allclose(value, stored[field], atol=1e-12, rtol=1e-10)
            roots.append(r["result"])
            outcome_map[r["result"]["id"]] = np.asarray(r["outcomes"], float)
    extra = {
        "source_sha256": digest(__file__),
        "per_state_rankings_recomputed_from_locked_scores_and_outcomes": True,
        "definition": "paired game-seed bootstrap; conditional root EV, not deployment cash",
        "comparisons": {},
        "uniform_action_reference": {},
    }
    for r in roots:
        evs = [a["mean"] for a in r["ev"]["actions"]]
        r["ranking"]["uniform_reference"] = {
            "best_action_hit_rate": len(r["ev"]["best_set"]) / len(evs),
            "pairwise_accuracy": 0.5 if r["ranking"]["baseline"]["non_tied_pairs"] else None,
            "regret_yuan": max(evs) - float(np.mean(evs)),
            "chosen_EV_yuan": float(np.mean(evs)),
        }
    pairs = [("B10/raw", "A10/raw"), ("B10/anchor", "A10/anchor")]
    pairs += [(f"{cp}/anchor", f"{cp}/raw") for cp in ("initial", "long10", "A10", "B10")]
    pairs += [("baseline", f"{cp}/raw") for cp in ("initial", "long10", "A10", "B10")]
    pairs += [(f"{cp}/raw", "uniform_reference") for cp in ("initial", "long10", "A10", "B10")]
    for a, b in pairs:
        for subset in ("all", "clear"):
            selected = [r for r in roots if subset == "all" or r["ev"]["clear"]]
            extra["comparisons"][f"{a}-minus-{b}:{subset}"] = {
                metric: cluster_difference(selected, a, b, metric)
                for metric in (
                    "best_action_hit_rate",
                    "pairwise_accuracy",
                    "regret_yuan",
                    "chosen_EV_yuan",
                )
            }
    for family in ("all", *QUOTAS):
        for subset in ("all", "clear", "ambiguous"):
            selected = [
                r
                for r in roots
                if (family == "all" or r["family"] == family)
                and (subset == "all" or r["ev"]["clear"] == (subset == "clear"))
            ]
            extra["uniform_action_reference"][f"{family}:{subset}"] = {
                "states": len(selected),
                "expected_strict_top1": float(
                    np.mean([1 / len(r["ev"]["actions"]) for r in selected])
                )
                if selected
                else None,
                "expected_best_hit": float(
                    np.mean([len(r["ev"]["best_set"]) / len(r["ev"]["actions"]) for r in selected])
                )
                if selected
                else None,
                "expected_regret_yuan": float(
                    np.mean(
                        [
                            max(a["mean"] for a in r["ev"]["actions"])
                            - np.mean([a["mean"] for a in r["ev"]["actions"]])
                            for r in selected
                        ]
                    )
                )
                if selected
                else None,
                "expected_pairwise_non_tied": 0.5
                if any(r["ranking"]["baseline"]["non_tied_pairs"] for r in selected)
                else None,
            }
    extra["rollout_precision"] = {}
    for family in ("all", *QUOTAS):
        selected = [r for r in roots if family == "all" or r["family"] == family]
        actions = [a for r in selected for a in r["ev"]["actions"]]
        pairs = [p for r in selected for p in r["ev"]["pairs"]]
        extra["rollout_precision"][family] = {
            "actions": len(actions),
            "action_se_yuan": summary([a["se"] for a in actions]),
            "zero_action_SE_fraction": sum(a["se"] == 0 for a in actions) / len(actions)
            if actions
            else None,
            "top_pair_mean_yuan": summary([r["ev"]["top_second_delta"] for r in selected]),
            "top_pair_ci_width_yuan": summary(
                [r["ev"]["top_second_ci95"][1] - r["ev"]["top_second_ci95"][0] for r in selected]
            ),
            "resolved_pair_fraction": sum(p["ci95"][0] > 0 or p["ci95"][1] < 0 for p in pairs)
            / len(pairs)
            if pairs
            else None,
            "distinct_outcome_vectors": summary(
                [r["ev"]["distinct_outcome_vectors"] for r in selected]
            ),
        }
    extra["policy_overrides"] = {
        strategy: {
            family: {
                "states": sum(family == "all" or r["family"] == family for r in roots),
                "overrides": sum(
                    (family == "all" or r["family"] == family)
                    and r["ranking"][strategy]["chosen"] != r["ranking"]["baseline"]["chosen"]
                    for r in roots
                ),
            }
            for family in ("all", *QUOTAS)
        }
        for strategy in result["ranking"]
        if strategy != "baseline"
    }
    extra["split_half_reliability"] = {}
    for family in ("all", *QUOTAS):
        for subset in ("all", "clear"):
            chosen = [
                r
                for r in roots
                if (family == "all" or r["family"] == family)
                and (subset == "all" or r["ev"]["clear"])
            ]
            agreements, best_hits, pair_agreements = [], [], []
            for r in chosen:
                a, b = [v.mean(0) for v in np.array_split(outcome_map[r["id"]], 2)]
                agreements.append(int(a.argmax() == b.argmax()))
                best_hits.append(int(np.isclose(b[int(a.argmax())], b.max(), atol=1e-10, rtol=0)))
                for i in range(len(a)):
                    for j in range(i + 1, len(a)):
                        da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
                        if da and db:
                            pair_agreements.append(int(da == db))
            extra["split_half_reliability"][f"{family}:{subset}"] = {
                "states": len(chosen),
                "strict_top_agreement": float(np.mean(agreements)) if agreements else None,
                "half1_top_in_half2_best_set": float(np.mean(best_hits)) if best_hits else None,
                "non_tied_both_halves_pairs": len(pair_agreements),
                "pair_sign_agreement": float(np.mean(pair_agreements)) if pair_agreements else None,
            }
    atomic_json(root / "comparison-report.json", extra)
    lines = ["\n### 完整benchmark结果\n"]

    def fmt(v):
        return "—" if v is None else f"{v:.4f}"

    def table(head, rows):
        cell = lambda v: str(v).replace("|", r"\|")
        lines.append("| " + " | ".join(map(cell, head)) + " |")
        lines.append("|" + "|".join("---" for _ in head) + "|")
        lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
        lines.append("")

    lines.append(
        f"有效状态{result['states']}/500，失败{len(result['failures'])}；clear={result['clear_states']}、ambiguous={result['states'] - result['clear_states']}，经验并列最优状态{result['ties']}。原始结果不进入训练。\n"
    )
    table(
        [
            "族",
            "有效状态",
            "clear",
            "ambiguous",
            "有非并列EV pair的状态",
            "uniform best hit(all/clear)",
        ],
        [
            [
                f,
                sum(r["family"] == f for r in roots),
                result["clear_by_family"][f],
                sum(r["family"] == f and not r["ev"]["clear"] for r in roots),
                sum(
                    r["family"] == f and r["ranking"]["baseline"]["non_tied_pairs"] > 0
                    for r in roots
                ),
                fmt(extra["uniform_action_reference"][f"{f}:all"]["expected_best_hit"])
                + "/"
                + fmt(extra["uniform_action_reference"][f"{f}:clear"]["expected_best_hit"]),
            ]
            for f in QUOTAS
        ],
    )
    for subset in ("all", "clear", "ambiguous"):
        lines.append(f"#### {subset}状态：总体排序\n")
        table(
            [
                "策略",
                "N",
                "strict Top1",
                "best-set hit",
                "pairwise macro/pooled",
                "Spearman",
                "Kendall τb",
                "经验regret元",
                "所选经验EV元",
            ],
            [
                [
                    s,
                    d[f"all:{subset}"]["states"],
                    fmt(d[f"all:{subset}"]["top1_accuracy"]),
                    fmt(d[f"all:{subset}"]["best_action_hit_rate"]),
                    fmt(d[f"all:{subset}"]["pairwise_accuracy"])
                    + "/"
                    + fmt(d[f"all:{subset}"]["pooled_pairwise_accuracy"]),
                    fmt(d[f"all:{subset}"]["spearman"]),
                    fmt(d[f"all:{subset}"]["kendall_tau_b"]),
                    fmt(d[f"all:{subset}"]["regret_yuan"]),
                    fmt(d[f"all:{subset}"]["chosen_EV_yuan"]),
                ]
                for s, d in result["ranking"].items()
            ],
        )
    table(
        ["策略", "偏离baseline根数/总数", "discard/chi/peng/hu偏离数"],
        [
            [
                strategy,
                str(d["all"]["overrides"]) + "/" + str(d["all"]["states"]),
                "/".join(str(d[f]["overrides"]) for f in QUOTAS),
            ]
            for strategy, d in extra["policy_overrides"].items()
        ],
    )
    table(
        [
            "族",
            "动作数",
            "action SE P25/median/P75元",
            "top pair CI宽P25/median/P75元",
            "零action SE比例",
            "CI可分辨pair比例",
        ],
        [
            [
                f,
                d["actions"],
                "/".join(fmt(d["action_se_yuan"].get("p" + str(k))) for k in (25, 50, 75)),
                "/".join(fmt(d["top_pair_ci_width_yuan"].get("p" + str(k))) for k in (25, 50, 75)),
                fmt(d["zero_action_SE_fraction"]),
                fmt(d["resolved_pair_fraction"]),
            ]
            for f, d in extra["rollout_precision"].items()
        ],
    )
    lines.append(
        "零SE可能来自相同结果、立即胡或无余牌；它不证明信息集下没有隐藏信息不确定性。每个action的mean/std/SE/CI/median/P25/P75及所有pair的配对差与CI完整保存在逐root结果，不人工筛case展示。\n"
    )
    lines.append("#### 32-world排序自身的split-half稳定性\n")
    table(
        [
            "族/子集",
            "N",
            "两半strict top相同",
            "前半top命中后半best-set",
            "两半皆非并列pair数",
            "pair符号一致率",
        ],
        [
            [
                k,
                v["states"],
                fmt(v["strict_top_agreement"]),
                fmt(v["half1_top_in_half2_best_set"]),
                v["non_tied_both_halves_pairs"],
                fmt(v["pair_sign_agreement"]),
            ]
            for k, v in extra["split_half_reliability"].items()
        ],
    )
    lines.append(
        "前16/后16只拆分已有32个world，不追加rollout；这是参照标签可靠性诊断，不是另一份训练/选型数据。并列会抬高best-set稳定性，pair符号一致率仅针对两半都非零的pair。\n"
    )
    lines.append("#### 分动作族：all与clear分别报告\n")
    table(
        [
            "族/子集/策略",
            "N",
            "strict Top1",
            "best-set hit",
            "pairwise macro",
            "Spearman",
            "Kendall τb",
            "经验regret元",
        ],
        [
            [
                f + "/" + subset + "/" + s,
                d[f"{f}:{subset}"]["states"],
                fmt(d[f"{f}:{subset}"]["top1_accuracy"]),
                fmt(d[f"{f}:{subset}"]["best_action_hit_rate"]),
                fmt(d[f"{f}:{subset}"]["pairwise_accuracy"]),
                fmt(d[f"{f}:{subset}"]["spearman"]),
                fmt(d[f"{f}:{subset}"]["kendall_tau_b"]),
                fmt(d[f"{f}:{subset}"]["regret_yuan"]),
            ]
            for f in QUOTAS
            for subset in ("all", "clear")
            for s, d in result["ranking"].items()
        ],
    )
    table(
        [
            "族/子集",
            "uniform strict Top1",
            "uniform best hit",
            "uniform pairwise",
            "uniform经验regret元",
        ],
        [
            [
                k,
                fmt(v["expected_strict_top1"]),
                fmt(v["expected_best_hit"]),
                fmt(v["expected_pairwise_non_tied"]),
                fmt(v["expected_regret_yuan"]),
            ]
            for k, v in extra["uniform_action_reference"].items()
        ],
    )
    lines.append(
        "Uniform reference仅解析计算“随机选合法候选”的期望指标，没有新增random actor rollout。并列经验EV较多时best hit的随机参考也会很高，不能直接拿50%作全部Top1的随机线。macro pairwise按有可比pair的状态等权，pooled按pair等权；—为不可估。\n"
    )
    table(
        ["相对initial/指标", "差值", "seed-cluster95%CI"],
        [
            [
                name + "/" + metric,
                fmt(v["delta"]),
                "—"
                if v["seed_cluster_ci95"] is None
                else "/".join(map(fmt, v["seed_cluster_ci95"])),
            ]
            for name, d in result["paired_checkpoint_comparisons"].items()
            for metric, v in d.items()
        ],
    )
    table(
        ["额外配对/子集/指标", "差值", "seed-cluster95%CI"],
        [
            [
                name + "/" + metric,
                fmt(v["mean"]),
                "—" if v["ci95"] is None else "/".join(map(fmt, v["ci95"])),
            ]
            for name, d in extra["comparisons"].items()
            for metric, v in d.items()
        ],
    )
    lines.append(
        "这些区间仅覆盖来源game簇抽样差异，没有嵌套重抽world，也未校正多重比较；不能当正式棋力晋级证据。regret负差表示改进，hit/pairwise正差表示改进。\n"
    )
    table(
        ["策略", "model top1-top2 margin与真实配对差Pearson/Spearman", "可分辨pair上的macro准确率"],
        [
            [
                s,
                fmt(d["all:all"]["margin_pearson"]) + "/" + fmt(d["all:all"]["margin_spearman"]),
                fmt(d["all:all"]["resolved_pairwise_accuracy"]),
            ]
            for s, d in result["ranking"].items()
        ],
    )
    table(
        ["族/策略", "自身top1-top2 margin vs rollout配对差Pearson/Spearman"],
        [
            [
                f + "/" + strategy,
                fmt(d[f + ":all"]["margin_pearson"]) + "/" + fmt(d[f + ":all"]["margin_spearman"]),
            ]
            for f in QUOTAS
            for strategy, d in result["ranking"].items()
        ],
    )
    lines.append(
        "跨动作族的总体margin相关可能混入HU/PENG与discard的尺度差异；应同时看族内相关，不能只凭总体Pearson宣称margin可用于confidence gate。\n"
    )
    lines.append("#### Margin分桶：相对差值校准，不是绝对return校准\n")
    rows = []
    for strategy, d in result["ranking"].items():
        if strategy == "baseline":
            continue
        for bucket, v in d["calibration"].items():
            lo, hi = map(float, bucket.split(":"))
            selected = [
                r["ranking"][strategy]
                for r in roots
                if lo <= 100 * r["ranking"][strategy]["model_margin_native"] < hi
            ]
            rows.append(
                [
                    strategy + "/" + bucket,
                    len(selected),
                    fmt(float(np.mean([r["model_margin_native"] * 100 for r in selected])))
                    if selected
                    else "—",
                    fmt(float(np.mean([r["rollout_margin_yuan"] for r in selected])))
                    if selected
                    else "—",
                    fmt(v["pairwise_accuracy"]),
                    fmt(v["regret_yuan"]),
                ]
            )
    table(
        [
            "策略/桶",
            "N",
            "模型margin均值（raw元/anchor评分）",
            "其top1-top2真实配对差均值元",
            "pairwise",
            "经验regret元",
        ],
        rows,
    )
    lines.append(
        "anchor margin包含log-prior，不能把评分数值和现金EV不等直接叫Q错误；观察其与真实配对差的关系即可。raw差值也对应训练continuation之外的共同π0和fixed hands，不能由绝对scale偏差直接判断Q训练错误。\n"
    )
    (root / "report.md").write_text("\n".join(lines))
    if figure is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matplotlib.rcParams["svg.hashsalt"] = "counterfactual-ranking-v1"
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.5), constrained_layout=True)
        checkpoint_names = ["initial", "long10", "A10", "B10"]
        positions = np.arange(4)
        for row, subset in enumerate(("all", "clear")):
            for col, metric in enumerate(
                ("best_action_hit_rate", "pairwise_accuracy", "regret_yuan")
            ):
                ax = axes[row, col]
                for offset, mode, color in ((-0.17, "raw", "#3878b8"), (0.17, "anchor", "#df8a35")):
                    vals = [
                        result["ranking"][f"{cp}/{mode}"][f"all:{subset}"][metric]
                        for cp in checkpoint_names
                    ]
                    ax.bar(
                        positions + offset,
                        [np.nan if v is None else v for v in vals],
                        width=0.32,
                        color=color,
                        label=mode,
                    )
                baseline = result["ranking"]["baseline"][f"all:{subset}"][metric]
                reference = extra["uniform_action_reference"][f"all:{subset}"]
                uniform_key = {
                    "best_action_hit_rate": "expected_best_hit",
                    "pairwise_accuracy": "expected_pairwise_non_tied",
                    "regret_yuan": "expected_regret_yuan",
                }[metric]
                if baseline is not None:
                    ax.axhline(
                        baseline, color="#222222", linestyle="--", linewidth=1.1, label="baseline"
                    )
                if reference[uniform_key] is not None:
                    ax.axhline(
                        reference[uniform_key],
                        color="#888888",
                        linestyle=":",
                        linewidth=1.2,
                        label="uniform (analytic)",
                    )
                ax.set_xticks(positions, checkpoint_names, fontsize=8)
                ax.tick_params(axis="y", labelsize=8)
                if col < 2:
                    ax.set_ylim(0, 1.02)
                else:
                    ax.set_ylim(bottom=0)
                title = {
                    "best_action_hit_rate": "Best-set hit rate",
                    "pairwise_accuracy": "Pairwise accuracy (macro)",
                    "regret_yuan": "Empirical regret (yuan)",
                }[metric]
                n = result["ranking"]["baseline"][f"all:{subset}"]["states"]
                ax.set_title(f"{subset}: {title} (n={n})", fontsize=9)
                ax.grid(axis="y", alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
        fig.suptitle(
            "Fixed hidden hands; common initial-anchor continuation\nPoint estimates only; paired CIs and family breakdown in report",
            fontsize=10,
        )
        destination = Path(figure)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, metadata={"Date": None})
        destination.write_text(
            "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n"
        )
        fig.savefig("/tmp/counterfactual-ranking-v1.png", dpi=150)
        plt.close(fig)
    print("Read-only ranking report generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/generated/counterfactual-ranking-v1")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    run(args.directory, args.figure)
