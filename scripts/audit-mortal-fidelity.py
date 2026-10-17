"""Read-only fixed-commit source/config/old-log audit; no games or model loads."""

import argparse
import gzip
import hashlib
import json
import math
import urllib.request
from collections import defaultdict
from pathlib import Path

COMMIT = "0cff2b52982be5b1163aa9a62fb01f03ce91e0d2"
PRIOR = Path("data/generated/mortal-continuation-v1/official-source")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ZiMortal-read-only-fidelity-audit"})
    with urllib.request.urlopen(req, timeout=40) as response:
        return response.read()


def source_audit(root, refresh):
    out = root / "official-source"
    out.mkdir(exist_ok=True)
    tree_path = out / "tree.json"
    if refresh:
        tree_path.write_bytes(
            download(
                f"https://api.github.com/repos/Equim-chan/Mortal/git/trees/{COMMIT}?recursive=1"
            )
        )
    tree = json.loads(tree_path.read_text())
    blobs = {e["path"]: e["sha"] for e in tree["tree"] if e["type"] == "blob"}
    prior = json.loads((PRIOR / "manifest.json").read_text())
    assert prior["commit"] == COMMIT
    paths = sorted(set(prior["files"]) | {"mortal/train_grp.py", "libriichi/src/dataset/grp.rs"})
    records = {}
    for name in paths:
        target = out / name.replace("/", "__")
        if refresh:
            target.write_bytes(
                download(f"https://raw.githubusercontent.com/Equim-chan/Mortal/{COMMIT}/{name}")
            )
        data = target.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == blobs[name], name
        if name in prior["files"]:
            assert hashlib.sha256(data).hexdigest() == prior["files"][name]["sha256"]
        records[name] = {
            "sha256": sha(target),
            "git_blob_sha1": blob,
            "lines": len(data.splitlines()),
            "url": f"https://github.com/Equim-chan/Mortal/blob/{COMMIT}/{name}",
        }
    result = {
        "commit": COMMIT,
        "tree_sha256": sha(tree_path),
        "files": records,
        "production_config": "UNKNOWN; config.example.toml explicitly says dummy placeholders",
    }
    save(out / "manifest.json", result)
    return result


def percentile(values, p):
    values = sorted(values)
    return values[round((len(values) - 1) * p)] if values else None


def distribution(values):
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        **{f"p{p}": percentile(values, p / 100) for p in (5, 25, 50, 75, 95)},
    }


def old_logs(root):
    base = Path("data/generated/mortal-online-long-v1")
    groups = defaultdict(
        lambda: {
            "count": 0,
            "vs_parent": 0,
            "vs_anchor": 0,
            "chosen_raw": 0,
            "low_support": 0,
            "low_support_overrides": 0,
            "p_chosen": [],
            "actor_chosen": [],
        }
    )
    inputs, generations = {}, []
    reconstructed_center_disagreements = 0
    for generation in range(1, 11):
        directory = base / f"gen-{generation:02d}"
        result_path = directory / "training-result.json"
        training = json.loads(result_path.read_text())
        inputs[str(result_path)] = sha(result_path)
        generations.append(
            {
                "generation": generation,
                "games": training["audit"]["games"],
                "decisions": training["decisions"],
                "updates": training["updates_this_generation"],
            }
        )
        files = sorted(directory.glob("evidence-*/decisions-*.jsonl.gz"))
        assert len(files) == 17
        gen_count = 0
        for path in files:
            inputs[str(path)] = sha(path)
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    r = json.loads(line)
                    q, p = r["q"], r["probabilities"]
                    n, chosen = len(p), r["chosen"]
                    assert n == len(q) and n > 1
                    center = max(range(n), key=lambda i: q[i] + 0.3 * math.log(p[i] + 1e-8))
                    assert len(r["scores"]) == n
                    recorded_center = max(range(n), key=r["scores"].__getitem__)
                    reconstructed_center_disagreements += center != recorded_center
                    center = recorded_center  # Actual saved float32 actor scores are authoritative.
                    raw = max(range(n), key=q.__getitem__)
                    for family in ("all", r["family"]):
                        g = groups[family]
                        g["count"] += 1
                        g["vs_parent"] += chosen != r["parent"]
                        g["vs_anchor"] += chosen != center
                        g["chosen_raw"] += chosen == raw
                        g["low_support"] += p[chosen] <= 0.1
                        g["low_support_overrides"] += p[chosen] <= 0.1 and chosen != r["parent"]
                        g["p_chosen"].append(p[chosen])
                        g["actor_chosen"].append(r["actual_behavior_probability"])
                    gen_count += 1
        assert gen_count == training["decisions"]
    result = {}
    for family, g in groups.items():
        result[family] = {k: v for k, v in g.items() if not isinstance(v, list)}
        result[family]["deviation_from_parent_rate"] = g["vs_parent"] / g["count"]
        result[family]["deviation_from_anchor_rate"] = g["vs_anchor"] / g["count"]
        result[family]["low_support_rate"] = g["low_support"] / g["count"]
        result[family]["chosen_p_reference"] = distribution(g["p_chosen"])
        result[family]["executed_actor_probability"] = distribution(g["actor_chosen"])
    rows = sum(g["decisions"] for g in generations)
    updates = sum(g["updates"] for g in generations)
    save(
        root / "old-log-audit.json",
        {
            "families": result,
            "generations": generations,
            "games": 50000,
            "reconstructed_center_disagreements": reconstructed_center_disagreements,
            "decisions": rows,
            "updates": updates,
            "decisions_per_game": rows / 50000,
            "updates_per_game": updates / 50000,
            "rows_per_update": rows / updates,
            "inputs": inputs,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-official", action="store_true")
    parser.add_argument(
        "--directory", type=Path, default=Path("data/generated/mortal-fidelity-audit-v1")
    )
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path("zimortal").glob("**/*.py"))
    checkpoints = [
        Path("data/generated/league-v2/stage-02/best-dev.pt"),
        Path("data/generated/mortal-coverage-v1/B-coverage/fit/selected-common.pt"),
    ]
    before = {str(p): sha(p) for p in paths + checkpoints}
    official = source_audit(args.directory, args.refresh_official)
    behavior = old_logs(args.directory)
    assert before == {str(p): sha(p) for p in paths + checkpoints}
    save(
        args.directory / "result.json",
        {
            "official": official,
            "behavior": behavior,
            "source_before_after": before,
            "training_updates_created": 0,
            "games_created": 0,
            "rollouts_created": 0,
            "audit_script_sha256": sha(__file__),
        },
    )
    print(
        "source files",
        len(official["files"]),
        "old decisions",
        behavior["all"]["count"],
        "training/games/rollouts created: 0",
    )


if __name__ == "__main__":
    main()
