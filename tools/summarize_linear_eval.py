#!/usr/bin/env python
"""Summarize linear-evaluation runs by following weights back to pretraining.

The script scans work_dir/linear_eval/**/config.yaml, reads the evaluated
checkpoint from the weights field, opens the referenced pretraining config, and
classifies the row as C/H/PL/HPL from clustering, hierarchy, and pseudo-label
settings. Metrics are taken from the same log epoch that achieved the best
Top-1 value.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WEIGHTS_RE = re.compile(
    r"(?m)^weights:\s*(?P<weights>.+?)\s*$|--weights\s+(?P<cli>\S+)"
)
EVAL_EPOCH_RE = re.compile(r"Eval epoch:\s*(\d+)")
TOP1_RE = re.compile(r"\bTop1:\s*([0-9.]+)%")
TOP5_RE = re.compile(r"\bTop5:\s*([0-9.]+)%")
LAMBDA_HIER_RE = re.compile(r"(?m)^lambda_hier:\s*([^\r\n#]+)")
NUM_CLUSTERS_RE = re.compile(r"(?m)^\s*num_clusters:\s*(\d+)")


@dataclass(frozen=True)
class BestEval:
    epoch: int | None
    top1: float
    top5: float | None


@dataclass(frozen=True)
class EvalSummary:
    split: str
    stream: str
    variant: str
    sinkhorn: bool
    hierarchy: bool
    pseudo: bool
    num_clusters: int | None
    epoch: int | None
    top1: float
    top5: float | None
    eval_run: str
    pretrain_run: str
    weights: str


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def normalize_slashes(value: str) -> str:
    return value.strip().strip("'\"").replace("\\", "/")


def rel_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def find_weights(config_text: str) -> str | None:
    for match in WEIGHTS_RE.finditer(config_text):
        value = match.group("weights") or match.group("cli")
        if value:
            return normalize_slashes(value)
    return None


def pretrain_run_from_weights(weights: str) -> str | None:
    weights = normalize_slashes(weights)
    match = re.search(
        r"(?P<run>work_dir/(?:skeletonclr|skeletonclr_3views)/.+?/runs/[^/\s]+)/"
        r"epoch\d+_model\.pt",
        weights,
    )
    if match:
        return match.group("run")
    return None


def parse_float(text: str) -> float | None:
    value = text.strip()
    if value.lower() in {"none", "null", "~"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_lambda_hier(config_text: str) -> float | None:
    match = LAMBDA_HIER_RE.search(config_text)
    if not match:
        return None
    return parse_float(match.group(1))


def parse_num_clusters(config_text: str, pretrain_run: str) -> int | None:
    match = NUM_CLUSTERS_RE.search(config_text)
    if match:
        return int(match.group(1))
    match = re.search(r"clust(\d+)", pretrain_run)
    if match:
        return int(match.group(1))
    return None


def bool_from_config(config_text: str, key: str) -> bool | None:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^\r\n#]+)", config_text)
    if not match:
        return None
    value = match.group(1).strip().lower()
    if value in {"true", "yes", "1"}:
        return True
    if value in {"false", "no", "0"}:
        return False
    return None


def infer_split(pretrain_run: str) -> str:
    if "xsubject" in pretrain_run or "xsub" in pretrain_run:
        return "cross-subject"
    if "xview" in pretrain_run:
        return "cross-view"
    return "unknown"


def infer_stream(pretrain_run: str) -> str:
    return "3s" if "skeletonclr_3views" in pretrain_run else "joint"


def infer_pseudo(config_text: str, pretrain_run: str) -> bool:
    lowered = config_text.lower()
    return (
        "pseudo" in pretrain_run.lower()
        or "mode: pseudo" in lowered
        or "pseudo_soft" in lowered
    )


def infer_sinkhorn(config_text: str, pretrain_run: str) -> bool:
    cluster_enabled = bool_from_config(config_text, "cluster_enabled")
    if cluster_enabled is not None:
        return cluster_enabled
    return "clust" in pretrain_run


def classify_variant(stream: str, sinkhorn: bool, hierarchy: bool, pseudo: bool) -> str:
    base = "3s-HypSkeletonCLR" if stream == "3s" else "HypSkeletonCLR"
    if not sinkhorn and not hierarchy and not pseudo:
        return f"{base} baseline"
    suffix = ""
    if sinkhorn and not hierarchy and not pseudo:
        suffix = "C"
    elif sinkhorn and hierarchy and not pseudo:
        suffix = "H"
    elif sinkhorn and not hierarchy and pseudo:
        suffix = "PL"
    elif sinkhorn and hierarchy and pseudo:
        suffix = "HPL"
    else:
        bits = []
        if sinkhorn:
            bits.append("C")
        if hierarchy:
            bits.append("H")
        if pseudo:
            bits.append("PL")
        suffix = "+".join(bits) if bits else "unknown"
    return f"{base}-{suffix}"


def best_eval_from_log(log_path: Path) -> BestEval | None:
    if not log_path.exists():
        return None

    records: list[BestEval] = []
    epoch: int | None = None
    current_top1: float | None = None

    for line in read_text(log_path).splitlines():
        epoch_match = EVAL_EPOCH_RE.search(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            current_top1 = None
            continue

        top1_match = TOP1_RE.search(line)
        if top1_match:
            current_top1 = float(top1_match.group(1))
            continue

        top5_match = TOP5_RE.search(line)
        if top5_match and current_top1 is not None:
            records.append(
                BestEval(
                    epoch=epoch,
                    top1=current_top1,
                    top5=float(top5_match.group(1)),
                )
            )

    if not records:
        return None
    return max(records, key=lambda item: (item.top1, item.epoch or -1))


def iter_eval_summaries(root: Path, work_dir: Path) -> Iterable[EvalSummary]:
    linear_eval_root = work_dir / "linear_eval"
    if not linear_eval_root.exists():
        return

    for eval_config in linear_eval_root.rglob("config.yaml"):
        eval_text = read_text(eval_config)
        weights = find_weights(eval_text)
        if not weights:
            continue

        pretrain_run = pretrain_run_from_weights(weights)
        if not pretrain_run:
            continue

        pretrain_config = root / pretrain_run / "config.yaml"
        if not pretrain_config.exists():
            continue

        best_eval = best_eval_from_log(eval_config.parent / "log.txt")
        if best_eval is None:
            continue

        pretrain_text = read_text(pretrain_config)
        split = infer_split(pretrain_run)
        stream = infer_stream(pretrain_run)
        sinkhorn = infer_sinkhorn(pretrain_text, pretrain_run)
        lambda_hier = parse_lambda_hier(pretrain_text)
        hierarchy = bool(lambda_hier and lambda_hier > 0.0)
        pseudo = infer_pseudo(pretrain_text, pretrain_run)
        variant = classify_variant(stream, sinkhorn, hierarchy, pseudo)

        yield EvalSummary(
            split=split,
            stream=stream,
            variant=variant,
            sinkhorn=sinkhorn,
            hierarchy=hierarchy,
            pseudo=pseudo,
            num_clusters=parse_num_clusters(pretrain_text, pretrain_run),
            epoch=best_eval.epoch,
            top1=best_eval.top1,
            top5=best_eval.top5,
            eval_run=rel_to_root(eval_config.parent, root),
            pretrain_run=pretrain_run,
            weights=weights,
        )


def filtered_rows(rows: Iterable[EvalSummary], args: argparse.Namespace) -> list[EvalSummary]:
    result = list(rows)
    if args.split:
        result = [row for row in result if row.split == args.split]
    if args.stream:
        result = [row for row in result if row.stream == args.stream]
    if args.variant:
        wanted = {item.strip() for item in args.variant.split(",")}
        result = [row for row in result if row.variant in wanted]
    if args.k is not None:
        result = [row for row in result if row.num_clusters == args.k]
    return result


def best_per_group(rows: Iterable[EvalSummary]) -> list[EvalSummary]:
    best: dict[tuple[str, str, str, int | None], EvalSummary] = {}
    for row in rows:
        key = (row.split, row.stream, row.variant, row.num_clusters)
        old = best.get(key)
        if old is None or (row.top1, row.epoch or -1) > (old.top1, old.epoch or -1):
            best[key] = row
    return list(best.values())


def fmt_bool(value: bool, latex: bool = False) -> str:
    if latex:
        return r"\(\checkmark\)" if value else "--"
    return "yes" if value else "no"


def fmt_float(value: float | None, bold: bool = False, latex: bool = False) -> str:
    if value is None:
        return ""
    text = f"{value:.2f}"
    if bold and latex:
        return rf"\textbf{{{text}}}"
    if bold:
        return f"**{text}**"
    return text


def write_csv(rows: list[EvalSummary]) -> None:
    fieldnames = [
        "split",
        "stream",
        "variant",
        "k",
        "sinkhorn",
        "hierarchy",
        "pseudo",
        "best_epoch",
        "top1",
        "top5",
        "eval_run",
        "pretrain_run",
        "weights",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "split": row.split,
                "stream": row.stream,
                "variant": row.variant,
                "k": row.num_clusters,
                "sinkhorn": row.sinkhorn,
                "hierarchy": row.hierarchy,
                "pseudo": row.pseudo,
                "best_epoch": row.epoch,
                "top1": f"{row.top1:.2f}",
                "top5": "" if row.top5 is None else f"{row.top5:.2f}",
                "eval_run": row.eval_run,
                "pretrain_run": row.pretrain_run,
                "weights": row.weights,
            }
        )


def rows_with_bold_flags(rows: list[EvalSummary]) -> dict[tuple[str, str], tuple[float, float]]:
    maxima: dict[tuple[str, str], tuple[float, float]] = {}
    for row in rows:
        key = (row.split, row.stream)
        top5 = row.top5 if row.top5 is not None else float("-inf")
        old_top1, old_top5 = maxima.get(key, (float("-inf"), float("-inf")))
        maxima[key] = (max(old_top1, row.top1), max(old_top5, top5))
    return maxima


def write_markdown(rows: list[EvalSummary], bold_best: bool) -> None:
    maxima = rows_with_bold_flags(rows) if bold_best else {}
    print("| Split | Stream | Variant | K | Sink. | Hier. | Pseudo | Epoch | Top-1 | Top-5 | Eval run |")
    print("|---|---|---|---:|---|---|---|---:|---:|---:|---|")
    for row in rows:
        max_top1, max_top5 = maxima.get((row.split, row.stream), (None, None))
        bold_top1 = bold_best and row.top1 == max_top1
        bold_top5 = bold_best and row.top5 is not None and row.top5 == max_top5
        print(
            "| {split} | {stream} | {variant} | {k} | {sink} | {hier} | {pseudo} | "
            "{epoch} | {top1} | {top5} | `{eval_run}` |".format(
                split=row.split,
                stream=row.stream,
                variant=row.variant,
                k="" if row.num_clusters is None else row.num_clusters,
                sink=fmt_bool(row.sinkhorn),
                hier=fmt_bool(row.hierarchy),
                pseudo=fmt_bool(row.pseudo),
                epoch="" if row.epoch is None else row.epoch,
                top1=fmt_float(row.top1, bold_top1),
                top5=fmt_float(row.top5, bold_top5),
                eval_run=row.eval_run,
            )
        )


def write_latex(rows: list[EvalSummary], bold_best: bool) -> None:
    maxima = rows_with_bold_flags(rows) if bold_best else {}
    for row in rows:
        max_top1, max_top5 = maxima.get((row.split, row.stream), (None, None))
        bold_top1 = bold_best and row.top1 == max_top1
        bold_top5 = bold_best and row.top5 is not None and row.top5 == max_top5
        print(
            "{variant}\n"
            "    & {split} & {sink} & {hier} & {pseudo} & {top1} & {top5} \\\\".format(
                variant=row.variant,
                split=row.split,
                sink=fmt_bool(row.sinkhorn, latex=True),
                hier=fmt_bool(row.hierarchy, latex=True),
                pseudo=fmt_bool(row.pseudo, latex=True),
                top1=fmt_float(row.top1, bold_top1, latex=True),
                top5=fmt_float(row.top5, bold_top5, latex=True),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map linear-eval logs to pretraining configs and summarize metrics."
    )
    parser.add_argument(
        "--root",
        default=".",
        type=Path,
        help="repository root; defaults to the current directory",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        type=Path,
        help="work_dir path; defaults to <root>/work_dir",
    )
    parser.add_argument("--split", choices=["cross-view", "cross-subject"])
    parser.add_argument("--stream", choices=["joint", "3s"])
    parser.add_argument("--variant", help="comma-separated variant names to include")
    parser.add_argument("--k", type=int, help="only include this cluster count")
    parser.add_argument(
        "--best-per-group",
        action="store_true",
        help="keep only best Top-1 run per split/stream/variant/K",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "markdown", "latex"],
        default="csv",
        help="output format",
    )
    parser.add_argument(
        "--bold-best",
        action="store_true",
        help="bold per-split/per-stream maxima in markdown or latex output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    work_dir = args.work_dir.resolve() if args.work_dir else root / "work_dir"

    rows = filtered_rows(iter_eval_summaries(root, work_dir), args)
    if args.best_per_group:
        rows = best_per_group(rows)

    rows.sort(
        key=lambda row: (
            row.stream,
            row.split,
            row.variant,
            row.num_clusters if row.num_clusters is not None else -1,
            -row.top1,
            row.eval_run,
        )
    )

    if args.format == "csv":
        write_csv(rows)
    elif args.format == "markdown":
        write_markdown(rows, args.bold_best)
    else:
        write_latex(rows, args.bold_best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
