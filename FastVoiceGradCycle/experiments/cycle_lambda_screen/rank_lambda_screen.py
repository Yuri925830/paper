#!/usr/bin/env python3
"""Rank completed lambda candidates from fixed development results only."""

import argparse
import json
import math
import os
from pathlib import Path


PRIMARY = {
    "utmos": "max",
    "dnsmos_p808": "max",
    "corpus_cer_percent": "min",
    "ecapa_cosine": "max",
}


def atomic_write(path, content):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def label(value):
    return format(value, ".8g").replace(".", "p")


def arm_for(value):
    return "base_post" if value == 0 else "cycle_post"


def load_candidate(root, value, start_pass, end_pass):
    arm = arm_for(value)
    directory = root / f"{arm}_seed_1234_lambda_{label(value)}_screen_{end_pass - start_pass}pass"
    paths = {
        "start": directory / f"development_metrics_pass_{start_pass:04d}.json",
        "end": directory / f"development_metrics_pass_{end_pass:04d}.json",
        "completion": directory / "completion.json",
        "contract": directory / "run_contract.json",
    }
    item = {"lambda": value, "arm": arm, "directory": str(directory), "eligible": False, "errors": []}
    for name, path in paths.items():
        if not path.is_file():
            item["errors"].append(f"missing {path.name}")
        else:
            item[name] = json.loads(path.read_text(encoding="utf-8"))
    if item["errors"]:
        return item
    contract = item["contract"]
    if float(contract.get("lambda_cycle", -1)) != value or contract.get("arm") != arm:
        item["errors"].append("run contract lambda or arm mismatch")
    if int(contract.get("screen_passes", -1)) != end_pass - start_pass or int(contract.get("seed", -1)) != 1234:
        item["errors"].append("run contract budget or seed mismatch")
    expected_step = end_pass * 1229
    for point, expected_pass in (("start", start_pass), ("end", end_pass)):
        summary = item[point]
        if summary.get("status") != "complete" or summary.get("split") != "development":
            item["errors"].append(f"{point} summary is not complete development")
        if int(summary.get("count", 0)) != 90 or int(summary.get("seed", -1)) != 1234:
            item["errors"].append(f"{point} summary count or seed mismatch")
        if int(summary.get("global_step", -1)) != expected_pass * 1229:
            item["errors"].append(f"{point} summary global step mismatch")
    if int(item["completion"].get("target_global_step", -1)) != expected_step:
        item["errors"].append("completion target step mismatch")
    if item["completion"].get("screening_uses_test") is not False:
        item["errors"].append("completion does not attest test isolation")
    for metric in PRIMARY:
        for point in ("start", "end"):
            measured = item[point].get(metric)
            if measured is None or not math.isfinite(float(measured)):
                item["errors"].append(f"non-finite {point}.{metric}")
    if item["errors"]:
        return item
    item["eligible"] = True
    item["endpoint"] = {metric: float(item["end"][metric]) for metric in PRIMARY}
    item["delta_from_shared_parent"] = {
        metric: float(item["end"][metric]) - float(item["start"][metric]) for metric in PRIMARY
    }
    item["diagnostics_not_ranked"] = {
        key: item["end"].get(key)
        for key in ("utterance_cer_percent", "sva_percent", "cycle_mae", "mel_rtf", "mel_vocoder_rtf")
    }
    return item


def competition_ranks(pairs, maximize):
    ordered = sorted(pairs, key=lambda pair: pair[1], reverse=maximize)
    result = {}
    previous = None
    rank = 0
    for index, (candidate, measured) in enumerate(ordered, 1):
        if previous is None or not math.isclose(measured, previous, rel_tol=1e-12, abs_tol=1e-12):
            rank = index
            previous = measured
        result[candidate] = rank
    return result


def dominates(left, right):
    no_worse = True
    strictly_better = False
    for metric, direction in PRIMARY.items():
        lvalue, rvalue = left["endpoint"][metric], right["endpoint"][metric]
        if direction == "max":
            no_worse &= lvalue >= rvalue
            strictly_better |= lvalue > rvalue
        else:
            no_worse &= lvalue <= rvalue
            strictly_better |= lvalue < rvalue
    return no_worse and strictly_better


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lambdas", type=float, nargs="+", required=True)
    parser.add_argument("--start-pass", type=int, default=100)
    parser.add_argument("--end-pass", type=int, default=105)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [load_candidate(args.run_root, value, args.start_pass, args.end_pass) for value in args.lambdas]
    eligible = [item for item in candidates if item["eligible"]]
    if len(eligible) != len(candidates):
        payload = {
            "status": "incomplete_candidate_set",
            "reason": "all requested lambda candidates must finish and pass comparability checks before selection",
            "requested_lambdas": args.lambdas,
            "eligible_lambdas": [item["lambda"] for item in eligible],
            "selection_uses_test": False,
            "candidates": candidates,
        }
        atomic_write(args.output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        raise SystemExit(2)
    parent_hashes = {item["contract"].get("parent_sha256") for item in eligible}
    development_hashes = {item["contract"].get("development_contract_sha256") for item in eligible}
    start_vectors = {
        tuple(round(float(item["start"][metric]), 10) for metric in PRIMARY) for item in eligible
    }
    if len(parent_hashes) != 1 or len(development_hashes) != 1 or len(start_vectors) != 1:
        payload = {
            "status": "incomparable_candidates",
            "reason": "parent, development contract, or shared-start metrics differ",
            "selection_uses_test": False,
            "candidates": candidates,
        }
        atomic_write(args.output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        raise SystemExit(3)
    for metric, direction in PRIMARY.items():
        endpoint = competition_ranks(
            [(item["lambda"], item["endpoint"][metric]) for item in eligible], direction == "max"
        )
        improvement = competition_ranks(
            [(item["lambda"], item["delta_from_shared_parent"][metric]) for item in eligible], direction == "max"
        )
        for item in eligible:
            item.setdefault("metric_ranks", {})[metric] = {
                "endpoint": endpoint[item["lambda"]],
                "improvement": improvement[item["lambda"]],
            }
    for item in eligible:
        values = [rank[kind] for rank in item["metric_ranks"].values() for kind in ("endpoint", "improvement")]
        item["mean_rank"] = sum(values) / len(values)
        item["pareto_dominated_by"] = [
            other["lambda"] for other in eligible if other is not item and dominates(other, item)
        ]
    eligible.sort(key=lambda item: (bool(item["pareto_dominated_by"]), item["mean_rank"], item["lambda"]))
    best = eligible[0]
    control_available = any(item["lambda"] == 0 for item in eligible)
    payload = {
        "status": "complete",
        "best_lambda": best["lambda"],
        "best_arm": best["arm"],
        "cycle_loss_helped": (best["lambda"] != 0) if control_available else None,
        "lambda_zero_control_available": control_available,
        "selection_scope": f"seed-1234, {args.end_pass - args.start_pass}-pass development-only engineering screen",
        "selection_uses_test": False,
        "primary_rule_frozen_before_results": "prefer the Pareto frontier, then equal-weight mean rank over endpoint and shared-parent improvement for UTMOS, DNSMOS-P808, corpus CER, and ECAPA cosine; final tie chooses the smaller lambda",
        "no_fake_uncertainty": "Only aggregate summaries are ranked; no confidence interval or independent-pair bootstrap is claimed.",
        "replication_notice": "Confirm the selected lambda with additional seeds before a final scientific claim.",
        "parent_sha256": next(iter(parent_hashes)),
        "development_contract_sha256": next(iter(development_hashes)),
        "ranking": eligible,
        "all_candidates": candidates,
    }
    atomic_write(args.output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    atomic_write(args.output.with_name("BEST_LAMBDA.txt"), f"{best['lambda']}\n")
    rows = [
        "# FastVoiceGrad Cycle-loss lambda screen",
        "",
        f"Best development-screening lambda: **{best['lambda']}**",
        "",
        "| rank | lambda | arm | mean rank | Pareto dominated by | UTMOS | P808 | corpus CER % | ECAPA |",
        "|---:|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for index, item in enumerate(eligible, 1):
        endpoint = item["endpoint"]
        rows.append(
            f"| {index} | {item['lambda']} | {item['arm']} | {item['mean_rank']:.3f} | "
            f"{item['pareto_dominated_by'] or '-'} | {endpoint['utmos']:.6f} | "
            f"{endpoint['dnsmos_p808']:.6f} | {endpoint['corpus_cer_percent']:.6f} | "
            f"{endpoint['ecapa_cosine']:.6f} |"
        )
    rows += ["", "The selection used development data only; the test split was not read.", ""]
    atomic_write(args.output.with_name("lambda_ranking.md"), "\n".join(rows))
    print(json.dumps({"best_lambda": best["lambda"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
