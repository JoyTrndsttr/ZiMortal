"""Fixed independent evaluation for all three saved training rounds."""

import json
from pathlib import Path

import torch

from zimortal.training.runtime import load_model, tournament


def main():
    torch.set_num_threads(2)
    seeds = list(range(10000, 10050))
    result = {"seeds": seeds, "evaluation_only": True, "seat_rotation": True}
    models = {f"round{i}": load_model(f"checkpoints/round{i}-resnet.pt") for i in (1, 2, 3)}
    for policy in ["random", "teacher"]:
        result[policy] = tournament(None, seeds, model_policy=policy)
        print(policy, result[policy], flush=True)
    for name, model in models.items():
        result[name] = tournament(model, seeds)
        print(name, result[name], flush=True)
    for name in ("round2", "round3"):
        result[name + "_vs_teacher"] = tournament(
            models[name], list(range(12000, 12020)), opponent="teacher"
        )
        print(name + "_vs_teacher", result[name + "_vs_teacher"], flush=True)
    result["round3_vs_round2"] = tournament(
        models["round3"],
        list(range(11000, 11020)),
        opponent="model",
        opponent_model=models["round2"],
    )
    result["round2_vs_round3"] = tournament(
        models["round2"],
        list(range(11000, 11020)),
        opponent="model",
        opponent_model=models["round3"],
    )
    a = models["round2"].state_dict()
    b = models["round3"].state_dict()
    result["policy_encoder_parameter_delta_l2"] = (
        sum(float(((a[k] - b[k]) ** 2).sum()) for k in a if not k.startswith(("value.", "wait.")))
        ** 0.5
    )
    Path("docs/training/blind-evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["round3_vs_round2"], result["round2_vs_round3"], flush=True)


if __name__ == "__main__":
    main()
