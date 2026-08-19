#!/usr/bin/env python3
"""Analyse a paired multi-lambda lm-eval comparison.

All arms evaluate the same documents, so the useful uncertainty test is paired:
McNemar's exact test over questions that changed correctness.  The
report also exposes empty model outputs, which otherwise look like ordinary
wrong answers in lm-eval's aggregate score.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [centre - radius, centre + radius]


def _exact_mcnemar(left_only: int, right_only: int) -> float:
    """Two-sided exact binomial p-value conditional on discordant pairs."""
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(left_only, right_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _paired_delta_ci(deltas: list[int], z: float = 1.96) -> list[float]:
    n = len(deltas)
    if n < 2:
        return [math.nan, math.nan]
    mean = sum(deltas) / n
    variance = sum((value - mean) ** 2 for value in deltas) / (n - 1)
    radius = z * math.sqrt(variance / n)
    return [mean - radius, mean + radius]


def _sample_files(arm_dir: Path, task: str) -> list[Path]:
    return sorted((arm_dir / task).glob("*/samples_*.jsonl"))


def _load_task(arm_dir: Path, task: str) -> dict[str, dict[str, Any]]:
    wanted_filter = "flexible-extract" if task == "gsm8k" else "strict_match"
    samples: dict[str, dict[str, Any]] = {}
    for path in _sample_files(arm_dir, task):
        for row in _read_jsonl(path):
            if row.get("filter") != wanted_filter:
                continue
            key = str(row["doc_hash"])
            if key in samples:
                raise RuntimeError(f"documento duplicado en {arm_dir.name}/{task}: {key}")
            response = ""
            resps = row.get("resps") or []
            if resps and resps[0]:
                response = str(resps[0][0] or "")
            samples[key] = {
                "correct": bool(row.get("exact_match")),
                "empty": not response.strip(),
                "doc_id": row.get("doc_id"),
                "category": (row.get("doc") or {}).get("category"),
            }
    if not samples:
        raise RuntimeError(f"no hay samples {wanted_filter!r} para {arm_dir.name}/{task}")
    return samples


def _pair(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if left.keys() != right.keys():
        missing_left = sorted(right.keys() - left.keys())
        missing_right = sorted(left.keys() - right.keys())
        raise RuntimeError(
            f"los brazos no contienen las mismas preguntas: "
            f"faltan_izquierda={missing_left[:3]} faltan_derecha={missing_right[:3]}")

    keys = sorted(left)
    left_correct = sum(left[key]["correct"] for key in keys)
    right_correct = sum(right[key]["correct"] for key in keys)
    left_only = sum(left[key]["correct"] and not right[key]["correct"] for key in keys)
    right_only = sum(right[key]["correct"] and not left[key]["correct"] for key in keys)
    deltas = [int(right[key]["correct"]) - int(left[key]["correct"]) for key in keys]
    left_empty = sum(left[key]["empty"] for key in keys)
    right_empty = sum(right[key]["empty"] for key in keys)
    left_empty_only = sum(left[key]["empty"] and not right[key]["empty"] for key in keys)
    right_empty_only = sum(right[key]["empty"] and not left[key]["empty"] for key in keys)
    n = len(keys)
    return {
        "n": n,
        "left": {
            "correct": left_correct,
            "score": left_correct / n,
            "score_ci95_wilson": _wilson(left_correct, n),
            "empty": left_empty,
            "empty_rate": left_empty / n,
        },
        "right": {
            "correct": right_correct,
            "score": right_correct / n,
            "score_ci95_wilson": _wilson(right_correct, n),
            "empty": right_empty,
            "empty_rate": right_empty / n,
        },
        "delta_right_minus_left": (right_correct - left_correct) / n,
        "delta_ci95_paired_normal": _paired_delta_ci(deltas),
        "correctness_discordant": {
            "left_only": left_only,
            "right_only": right_only,
            "mcnemar_exact_p_two_sided": _exact_mcnemar(left_only, right_only),
        },
        "empty_delta_right_minus_left": (right_empty - left_empty) / n,
        "empty_discordant": {
            "left_only": left_empty_only,
            "right_only": right_empty_only,
            "mcnemar_exact_p_two_sided": _exact_mcnemar(left_empty_only, right_empty_only),
        },
    }


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _arm_summary(samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = len(samples)
    correct = sum(row["correct"] for row in samples.values())
    empty = sum(row["empty"] for row in samples.values())
    return {
        "correct": correct,
        "score": correct / n,
        "score_ci95_wilson": _wilson(correct, n),
        "empty": empty,
        "empty_rate": empty / n,
    }


def _comparison(samples_by_lambda: dict[str, dict[str, dict[str, Any]]],
                lambdas: list[float]) -> dict[str, Any]:
    first = samples_by_lambda[str(lambdas[0])]
    result: dict[str, Any] = {
        "n": len(first),
        "arms": {
            str(lam): _arm_summary(samples_by_lambda[str(lam)]) for lam in lambdas
        },
        "pairwise": {},
    }
    for left_index, left_lambda in enumerate(lambdas):
        for right_lambda in lambdas[left_index + 1:]:
            key = f"{left_lambda}_vs_{right_lambda}"
            result["pairwise"][key] = {
                "left_lambda": left_lambda,
                "right_lambda": right_lambda,
                **_pair(samples_by_lambda[str(left_lambda)], samples_by_lambda[str(right_lambda)]),
            }
    return result


def _markdown(report: dict[str, Any]) -> str:
    lambdas = report["lambdas"]
    arm_headers = " | ".join(f"lambda={lam}" for lam in lambdas)
    lines = [
        "# Lambda quality comparison",
        "",
        "Comparacion pareada sobre las mismas preguntas.",
        "",
        f"| Suite | n | {arm_headers} |",
        "|---|---:" + "|---:" * len(lambdas) + "|",
    ]
    for task, result in report["tasks"].items():
        arm_cells = " | ".join(
            f"{result['arms'][str(lam)]['correct']}/{result['n']} "
            f"({_fmt_pct(result['arms'][str(lam)]['score'])}; "
            f"vacias {result['arms'][str(lam)]['empty']})"
            for lam in lambdas
        )
        lines.append(f"| {task} | {result['n']} | {arm_cells} |")

    lines += [
        "",
        "## Comparaciones pareadas",
        "",
        "| Suite | izquierda | derecha | delta | IC95 pareado | McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, result in report["tasks"].items():
        for pair in result["pairwise"].values():
            low, high = pair["delta_ci95_paired_normal"]
            lines.append(
                f"| {task} | {pair['left_lambda']} | {pair['right_lambda']} | "
                f"{_fmt_pct(pair['delta_right_minus_left'])} | "
                f"{_fmt_pct(low)} a {_fmt_pct(high)} | "
                f"{pair['correctness_discordant']['mcnemar_exact_p_two_sided']:.4f} |")

    subjects = report.get("mmlu_pro_subjects") or {}
    if subjects:
        lines += [
            "",
            "## MMLU-Pro por materia",
            "",
            f"| Materia | n | {arm_headers} |",
            "|---|---:" + "|---:" * len(lambdas) + "|",
        ]
        for subject, result in sorted(subjects.items()):
            arm_cells = " | ".join(
                _fmt_pct(result["arms"][str(lam)]["score"]) for lam in lambdas)
            lines.append(f"| {subject} | {result['n']} | {arm_cells} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--lambdas",
        help="Subconjunto de brazos completos, separado por comas",
    )
    args = parser.parse_args()

    metadata = _read_json(args.results_dir / "metadata.json")
    metadata_lambdas = [float(value) for value in metadata["lambdas"]]
    lambdas = (
        [float(value) for value in args.lambdas.split(",")]
        if args.lambdas else metadata_lambdas
    )
    if any(value not in metadata_lambdas for value in lambdas):
        raise RuntimeError(
            f"--lambdas contiene brazos fuera de metadata: "
            f"{lambdas} no es subconjunto de {metadata_lambdas}")
    if len(lambdas) < 2:
        raise RuntimeError(f"se esperaban al menos dos brazos: {lambdas}")
    tasks = [suite["task"] for suite in metadata["suites"]]

    report: dict[str, Any] = {
        "lambdas": lambdas,
        "tasks": {},
        "mmlu_pro_subjects": {},
    }
    for task in tasks:
        samples_by_lambda = {
            str(lam): _load_task(args.results_dir / f"lambda_{lam}", task)
            for lam in lambdas
        }
        report["tasks"][task] = _comparison(samples_by_lambda, lambdas)
        if task == "mmlu_pro_llama":
            categories = sorted({
                str(row["category"])
                for row in samples_by_lambda[str(lambdas[0])].values()
            })
            for category in categories:
                subject_samples = {
                    str(lam): {
                        key: row for key, row in samples_by_lambda[str(lam)].items()
                        if row["category"] == category
                    }
                    for lam in lambdas
                }
                report["mmlu_pro_subjects"][category] = _comparison(
                    subject_samples, lambdas)

    out_path = args.out or args.results_dir / "paired_analysis.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path = out_path.with_suffix(".md")
    markdown_path.write_text(_markdown(report))
    print(markdown_path.read_text(), end="")
    print(f"JSON: {out_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
