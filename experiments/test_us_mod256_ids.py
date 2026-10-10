#!/usr/bin/env python3
"""Test configurable transform/RS/indexed oligos under an independent IDS channel."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import reedsolo

from encode_us_gf256_second_rs_fixed130 import (
    apply_transform,
    build_rs_grid,
    inverse_transform,
    required_index_symbols,
)


BASES = "ACGT"
_W = {}


def decode_linear_index(symbols) -> int:
    values = [int(value) for value in symbols]
    if not values:
        raise ValueError("the linear index must contain at least one symbol")
    position = 0
    for value in values:
        if not 0 <= value <= 255:
            raise ValueError("linear index symbols must be in GF(256)")
        position = (position << 8) | value
    return int(position)


def entropy_u8(values: np.ndarray) -> float:
    counts = np.bincount(values.reshape(-1), minlength=256).astype(np.float64)
    p = counts[counts > 0] / values.size
    return float(-np.sum(p * np.log2(p)))


def mod256_second(image: np.ndarray) -> np.ndarray:
    """Compatibility alias for the current row-second transform."""
    return apply_transform(image, "row_second")[0]


def ids_channel(dna: str, p_i: float, p_d: float, p_s: float, rng):
    """Independent deletion/substitution per source base and insertion after it."""
    output = []
    n_i = n_d = n_s = 0
    for base in dna:
        deleted = rng.random() < p_d
        if deleted:
            n_d += 1
        else:
            emitted = base
            if rng.random() < p_s:
                emitted = BASES[int(rng.integers(0, 4))]
                while emitted == base:
                    emitted = BASES[int(rng.integers(0, 4))]
                n_s += 1
            output.append(emitted)
        if rng.random() < p_i:
            output.append(BASES[int(rng.integers(0, 4))])
            n_i += 1
    return "".join(output), {"insertions": n_i, "deletions": n_d, "substitutions": n_s}


def _decode_case(case):
    index, expected_start, expected_payload, noisy, meta, events = case
    started = time.perf_counter()
    try:
        decoded, details = _W["decoder"](
            noisy, _W["dfa"], meta, global_table=_W["global_table"],
            beam_width=_W["beam"], edit_penalty=4,
            relative_cost_window=_W["relative_cost_window"],
            reserve_edited_fraction=0.5,
            short_macro_lookahead=_W["short_lookahead"],
            lookahead_pool_factor=_W["lookahead_pool_factor"],
            share_macro_transitions=_W["share_macro_transitions"],
            share_alignment_results=_W["share_alignment_results"],
            use_numpy_batch=_W["use_numpy_batch"],
            return_details=True,
        )
        index_symbols = int(_W["index_symbols"])
        if decoded.size < index_symbols:
            raise ValueError("decoded inner sequence is shorter than its index")
        decoded_start = decode_linear_index(decoded[-index_symbols:][::-1])
        decoded_payload = np.asarray(decoded[:-index_symbols], dtype=np.int64)
        decoded_length = int(decoded_payload.size)
        if not 0 <= decoded_start < _W["stream_length"]:
            raise ValueError(f"decoded index k={decoded_start} is out of range")
        if decoded_start + decoded_length > _W["stream_length"]:
            raise ValueError(
                f"decoded interval [{decoded_start}, "
                f"{decoded_start + decoded_length}) is out of range"
            )
        exact = bool(
            decoded_start == expected_start
            and np.array_equal(decoded_payload, expected_payload)
        )
        status = "exact" if exact else "wrong"
        detail = details
    except Exception as exc:
        status = "erasure"
        decoded_start = None
        decoded_length = None
        decoded_payload = None
        detail = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "oligo_index": int(index), "status": status,
        "events": events, "event_count": int(sum(events.values())),
        "received_length": len(noisy),
        "decoded_start": decoded_start,
        "decoded_payload_length": decoded_length,
        "decoded_payload": (
            decoded_payload.tolist() if decoded_payload is not None else None
        ),
        "seconds": float(time.perf_counter() - started),
        "details": detail,
    }


def _row_cost(row) -> int:
    return int(row.get("details", {}).get("cost", 1 << 30))


def _row_interval(row):
    start = int(row["decoded_start"])
    length = int(row["decoded_payload_length"])
    return start, start + length


def reconstruct_by_decoded_indices(
    results, stream_length: int, *, cost_threshold=None, excluded_oligos=None,
    forced_oligos=None,
):
    """Conservatively arbitrate decoded index intervals.

    A positive gap is never evidence of an index error: it may represent a
    missing oligo.  Duplicate starts and overlapping decoded intervals are
    resolved only by a >=4 cost advantage or by unique two-sided boundary
    support.  Unresolved candidates are withheld as erasures for outer RS.
    """
    excluded = {int(value) for value in (excluded_oligos or ())}
    forced = {int(value) for value in (forced_oligos or ())} - excluded
    valid = []
    invalid_count = cost_erased = feedback_erased = 0
    for row in results:
        if row.get("decoded_start") is None or row.get("decoded_payload") is None:
            continue
        start, end = _row_interval(row)
        if start < 0 or end <= start or end > int(stream_length):
            invalid_count += 1
            continue
        if int(row["oligo_index"]) in excluded:
            feedback_erased += 1
            continue
        if cost_threshold is not None and _row_cost(row) > int(cost_threshold):
            cost_erased += 1
            continue
        valid.append(row)

    groups = {}
    for row in valid:
        groups.setdefault(int(row["decoded_start"]), []).append(row)
    starts = sorted(groups)
    duplicate_count = sum(max(0, len(rows) - 1) for rows in groups.values())

    # A next index is trusted when it is unique, or has a uniquely lower-cost
    # candidate.  Its start is useful as a tie-break boundary, never as a rule
    # for truncating the preceding payload.
    trusted_starts = []
    for start in starts:
        costs = sorted(_row_cost(row) for row in groups[start])
        if len(costs) == 1 or costs[0] + 4 <= costs[1]:
            trusted_starts.append(start)

    accepted = []
    held = []
    ambiguous_held = []
    decision_log = []
    for start in starts:
        candidates = sorted(
            groups[start], key=lambda row: (_row_cost(row), int(row["oligo_index"])),
        )
        forced_here = [
            row for row in candidates if int(row["oligo_index"]) in forced
        ]
        trusted_offset = bisect.bisect_right(trusted_starts, start)
        next_trusted = (
            trusted_starts[trusted_offset]
            if trusted_offset < len(trusted_starts) else None
        )
        if len(forced_here) == 1:
            chosen = forced_here[0]
            reason = "rs_feedback_forced_candidate"
            held.extend(row for row in candidates if row is not chosen)
        elif len(candidates) == 1:
            chosen = candidates[0]
            reason = "unique_index"
        elif _row_cost(candidates[0]) + 4 <= _row_cost(candidates[1]):
            chosen = candidates[0]
            reason = "duplicate_lower_cost"
            held.extend(candidates[1:])
        else:
            best_cost = _row_cost(candidates[0])
            close = [row for row in candidates if _row_cost(row) < best_cost + 4]
            boundary_matches = [
                row for row in close
                if next_trusted is not None and _row_interval(row)[1] == next_trusted
            ]
            if len(boundary_matches) == 1:
                chosen = boundary_matches[0]
                reason = "duplicate_next_trusted_boundary"
                held.extend(row for row in candidates if row is not chosen)
            else:
                chosen = None
                reason = "duplicate_ambiguous_erasure"
                held.extend(candidates)
                ambiguous_held.extend(candidates)
        if chosen is not None:
            accepted.append(chosen)
        decision_log.append({
            "start": int(start), "candidate_count": int(len(candidates)),
            "next_trusted_start": (
                None if next_trusted is None else int(next_trusted)
            ),
            "chosen_oligo_index": (
                None if chosen is None else int(chosen["oligo_index"])
            ),
            "reason": reason,
        })

    def two_sided_support(candidate, pool, conflict_ids):
        start, end = _row_interval(candidate)
        outside = [
            row for row in pool
            if int(row["oligo_index"]) not in conflict_ids
        ]
        left = [row for row in outside if _row_interval(row)[1] <= start]
        right = [row for row in outside if _row_interval(row)[0] >= end]
        if not left or not right:
            return False
        left_row = max(left, key=lambda row: _row_interval(row)[1])
        right_row = min(right, key=lambda row: _row_interval(row)[0])
        return bool(
            _row_interval(left_row)[1] == start
            and end == _row_interval(right_row)[0]
            and _row_cost(left_row) <= _row_cost(candidate)
            and _row_cost(right_row) <= _row_cost(candidate)
        )

    # Resolve true interval overlap.  Gaps are deliberately ignored.
    overlap_conflicts = overlap_cost_resolved = overlap_boundary_resolved = 0
    while True:
        accepted.sort(key=lambda row: (_row_interval(row)[0], _row_interval(row)[1]))
        pair = None
        for left, right in zip(accepted, accepted[1:]):
            if _row_interval(left)[1] > _row_interval(right)[0]:
                pair = (left, right)
                break
        if pair is None:
            break
        overlap_conflicts += 1
        left, right = pair
        left_cost, right_cost = _row_cost(left), _row_cost(right)
        left_forced = int(left["oligo_index"]) in forced
        right_forced = int(right["oligo_index"]) in forced
        if left_forced != right_forced:
            loser = right if left_forced else left
            accepted.remove(loser)
            held.append(loser)
            continue
        if abs(left_cost - right_cost) >= 4:
            loser = right if right_cost > left_cost else left
            accepted.remove(loser)
            held.append(loser)
            overlap_cost_resolved += 1
            continue
        conflict_ids = {int(left["oligo_index"]), int(right["oligo_index"])}
        support = [
            row for row in pair
            if two_sided_support(row, accepted, conflict_ids)
        ]
        if len(support) == 1:
            loser = right if support[0] is left else left
            accepted.remove(loser)
            held.append(loser)
            overlap_boundary_resolved += 1
        else:
            accepted.remove(left)
            accepted.remove(right)
            held.extend((left, right))
            ambiguous_held.extend((left, right))

    reconstructed = np.zeros(int(stream_length), dtype=np.uint8)
    known = np.zeros(int(stream_length), dtype=bool)
    active_intervals = []
    for row in accepted:
        start, end = _row_interval(row)
        payload = np.asarray(row["decoded_payload"], dtype=np.uint8)
        reconstructed[start:end] = payload
        known[start:end] = True
        active_intervals.append({
            "oligo_index": int(row["oligo_index"]), "start": int(start),
            "end": int(end), "cost": int(_row_cost(row)),
        })

    held_ids = sorted({int(row["oligo_index"]) for row in held})
    ambiguous_held_ids = sorted({
        int(row["oligo_index"]) for row in ambiguous_held
    })
    return reconstructed, known, {
        "decoded_valid_intervals": int(len(valid)),
        "invalid_intervals_erased": int(invalid_count),
        "cost_threshold": None if cost_threshold is None else int(cost_threshold),
        "cost_threshold_erased": int(cost_erased),
        "rs_feedback_erased": int(feedback_erased),
        "distinct_decoded_indices": int(len(starts)),
        "duplicate_index_candidates": int(duplicate_count),
        "overlap_conflicts": int(overlap_conflicts),
        "overlap_resolved_by_cost": int(overlap_cost_resolved),
        "overlap_resolved_by_two_sided_boundary": int(overlap_boundary_resolved),
        "held_candidate_erasures": int(len(held_ids)),
        "ambiguous_candidate_erasures": int(len(ambiguous_held_ids)),
        "held_candidate_oligo_indices": held_ids,
        "ambiguous_candidate_oligo_indices": ambiguous_held_ids,
        "positive_gaps_are_not_error_evidence": True,
        "known_symbols_after_index_reassembly": int(known.sum()),
        "missing_symbols_after_index_reassembly": int((~known).sum()),
        "decision_count": int(len(decision_log)),
        "first_anomalous_decisions": [
            row for row in decision_log if row["candidate_count"] > 1
        ][:20],
        "active_intervals": active_intervals,
    }


def evaluate_rs_recovery(reconstructed, known, expected_grid, expected_systematic):
    n, columns = expected_grid.shape
    k = int(expected_systematic.shape[0])
    received_grid = np.asarray(reconstructed, dtype=np.uint8).reshape(n, columns)
    known_grid = np.asarray(known, dtype=bool).reshape(n, columns)
    codec = reedsolo.RSCodec(n - k, nsize=n)
    exact_columns = wrong_columns = failed_columns = 0
    failed_column_indices = []
    wrong_column_indices = []
    erasures_per_column = []
    recovered = np.zeros_like(expected_systematic)
    for column in range(columns):
        erasures = np.flatnonzero(~known_grid[:, column]).astype(int).tolist()
        erasures_per_column.append(len(erasures))
        try:
            decoded = codec.decode(
                received_grid[:, column].tobytes(), erase_pos=erasures,
            )[0]
            values = np.frombuffer(decoded, dtype=np.uint8)
            recovered[:, column] = values
            if np.array_equal(values, expected_systematic[:, column]):
                exact_columns += 1
            else:
                wrong_columns += 1
                wrong_column_indices.append(int(column))
        except Exception:
            failed_columns += 1
            failed_column_indices.append(int(column))
    return {
        "columns": int(columns),
        "exact_columns": int(exact_columns),
        "wrong_columns": int(wrong_columns),
        "failed_columns": int(failed_columns),
        "failed_column_indices": failed_column_indices,
        "wrong_column_indices": wrong_column_indices,
        "all_columns_exact": bool(exact_columns == columns),
        "mismatched_bytes": int((recovered != expected_systematic).sum()),
        "total_matrix_bytes": int(expected_systematic.size),
        "erasure_symbols_per_column": {
            "min": int(min(erasures_per_column)),
            "max": int(max(erasures_per_column)),
            "mean": float(np.mean(erasures_per_column)),
        },
        "recovered_systematic": recovered,
    }


def reconstruct_with_rs_feedback(
    results, stream_length, expected_grid, expected_systematic, *,
    cost_threshold, max_iterations=32,
):
    """Reassemble and iteratively demote high-risk failed-column intervals.

    Only an RS *decoding failure* is used as feedback.  Ground-truth column
    equality is reported by the experiment but is never used to choose an
    interval, so this procedure remains usable without the source data.
    """
    excluded = set()
    forced = set()
    history = []
    reconstructed = known = reassembly = rs = None
    columns = int(expected_grid.shape[1])
    for iteration in range(int(max_iterations) + 1):
        reconstructed, known, reassembly = reconstruct_by_decoded_indices(
            results, int(stream_length), cost_threshold=cost_threshold,
            excluded_oligos=excluded, forced_oligos=forced,
        )
        rs = evaluate_rs_recovery(
            reconstructed, known, expected_grid, expected_systematic,
        )
        failed = set(rs["failed_column_indices"])
        history.append({
            "iteration": int(iteration),
            "failed_columns": int(len(failed)),
            "newly_erased_oligo_indices": [],
            "newly_forced_oligo_indices": [],
        })
        if not failed or iteration >= int(max_iterations):
            break

        # First let RS adjudicate candidates withheld only because of a
        # structural ambiguity.  A candidate is promoted only when trying it
        # strictly reduces the number of RS decode failures.
        rows_by_id = {int(row["oligo_index"]): row for row in results}
        promotable = []
        for oligo_index in reassembly["ambiguous_candidate_oligo_indices"]:
            if oligo_index in forced or oligo_index in excluded:
                continue
            row = rows_by_id.get(int(oligo_index))
            if row is None:
                continue
            start, end = _row_interval(row)
            touched = {
                int(position % columns) for position in range(start, end)
            }
            if touched & failed:
                promotable.append(int(oligo_index))
        best_trial = None
        for oligo_index in promotable:
            trial_forced = forced | {int(oligo_index)}
            trial_reconstructed, trial_known, trial_reassembly = (
                reconstruct_by_decoded_indices(
                    results, int(stream_length), cost_threshold=cost_threshold,
                    excluded_oligos=excluded, forced_oligos=trial_forced,
                )
            )
            trial_rs = evaluate_rs_recovery(
                trial_reconstructed, trial_known,
                expected_grid, expected_systematic,
            )
            trial_key = (
                int(trial_rs["failed_columns"]), int(oligo_index),
            )
            if best_trial is None or trial_key < best_trial[0]:
                best_trial = (
                    trial_key, oligo_index, trial_reconstructed, trial_known,
                    trial_reassembly, trial_rs,
                )
        if best_trial is not None and best_trial[0][0] < len(failed):
            _, oligo_index, reconstructed, known, reassembly, rs = best_trial
            forced.add(int(oligo_index))
            history[-1]["newly_forced_oligo_indices"] = [int(oligo_index)]
            continue

        # If no held candidate improves RS, nominate the highest-cost active
        # interval touching every failed column and demote it to erasure.
        # For every failed column, nominate the highest-cost active interval
        # touching it.  One interval can cover several failed columns.  Cost
        # dominates; failed-column coverage and oligo id are tie-breakers.
        active = reassembly["active_intervals"]
        coverage = {}
        for interval in active:
            touched = {
                int(position % columns)
                for position in range(interval["start"], interval["end"])
            } & failed
            if touched:
                coverage[int(interval["oligo_index"])] = (
                    interval, touched,
                )
        if not coverage:
            break
        nominees = set()
        for column in failed:
            choices = [
                value for value in coverage.values() if column in value[1]
            ]
            if not choices:
                continue
            interval, _ = max(
                choices,
                key=lambda value: (
                    int(value[0]["cost"]), len(value[1]),
                    -int(value[0]["oligo_index"]),
                ),
            )
            nominees.add(int(interval["oligo_index"]))
        if not nominees:
            break
        excluded.update(nominees)
        history[-1]["newly_erased_oligo_indices"] = sorted(nominees)

    rs["feedback_iterations"] = int(len(history) - 1)
    rs["feedback_erased_oligo_indices"] = sorted(excluded)
    rs["feedback_forced_oligo_indices"] = sorted(forced)
    rs["feedback_history"] = history
    reassembly["accepted_oligo_indices"] = sorted(
        int(interval["oligo_index"])
        for interval in reassembly["active_intervals"]
    )
    # Active intervals are an internal feedback data structure; retaining all
    # of them would inflate each result JSON by tens of thousands of lines.
    reassembly.pop("active_intervals", None)
    return reconstructed, known, reassembly, rs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/us_full.bin")
    parser.add_argument(
        "--oligos",
        default="outputs/us_mod256_second_rs255_k200_fixed130_indexed/oligos.csv",
    )
    parser.add_argument("--code-dir", default="codec")
    parser.add_argument("--height", type=int, default=471)
    parser.add_argument("--width", type=int, default=562)
    parser.add_argument(
        "--transform",
        choices=("raw", "row_first", "row_second", "column_first"),
        default="row_second",
    )
    parser.add_argument("--N", type=int, default=255)
    parser.add_argument("--K", type=int, default=200)
    parser.add_argument("--dna-length", type=int, default=130)
    parser.add_argument("--pb", type=int, default=3)
    parser.add_argument("--gtable-log", type=int, default=14)
    parser.add_argument(
        "--shared-table", default=None,
        help="数据集共用表 json;给定则编解码都用其 step3_final_shared_table,"
             "不从本图现算、不走 header 往返。",
    )
    parser.add_argument(
        "--signature-mode", choices=("mixed_v2", "adaptive_v1"),
        default="mixed_v2",
    )
    parser.add_argument(
        "--signature-count", type=int, choices=(0, 1, 2), default=None,
        help=(
            "fixed path-verification bases per internal symbol; must match "
            "the encoding run"
        ),
    )
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument(
        "--all", action="store_true",
        help="decode every oligo in the indexed complete-data encoding",
    )
    parser.add_argument(
        "--indices", default="",
        help="optional comma-separated actual oligo indices; overrides --cases sampling",
    )
    parser.add_argument(
        "--retry-clean-nonexact-from", default="",
        help=(
            "select rows with event_count=0 and status!=exact from a prior "
            "result.json; useful for clean beam-width recovery audits"
        ),
    )
    parser.add_argument(
        "--retry-ids-nonexact-from", default="",
        help=(
            "select rows with event_count>0 and status!=exact from a prior "
            "result.json, regenerating the same deterministic IDS reads"
        ),
    )
    parser.add_argument(
        "--retry-status", choices=("nonexact", "wrong", "erasure"),
        default="nonexact",
        help="optional status filter for --retry-*-nonexact-from",
    )
    parser.add_argument(
        "--keep-decoded-payload", action="store_true",
        help="retain decoded payload arrays in JSON for external reassembly",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--beam", type=int, default=256)
    parser.add_argument(
        "--cost-window", type=int, default=4,
        help="relative augmented-cost window; not a cumulative edit limit",
    )
    parser.add_argument(
        "--short-lookahead", type=int, default=4,
        help="apply one-symbol lookahead when the current macro is this many nt or shorter; 0 disables it",
    )
    parser.add_argument("--lookahead-pool", type=float, default=1.5)
    parser.add_argument("--enable-macro-memo", action="store_true")
    parser.add_argument("--disable-alignment-memo", action="store_true")
    parser.add_argument("--disable-numpy-batch", action="store_true")
    parser.add_argument("--pi", type=float, default=0.003)
    parser.add_argument("--pd", type=float, default=0.003)
    parser.add_argument("--ps", type=float, default=0.003)
    parser.add_argument(
        "--cost-threshold", type=int, default=None,
        help=(
            "erase decoded intervals above this cost; default is "
            "2*ceil(expected IDS events per oligo)*4"
        ),
    )
    parser.add_argument("--rs-feedback-iterations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="oligo 丢失率（整条链缺失，交由列向 RS erasure 恢复）")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--homopolymer-h", type=int, default=5,
        help="DFA h 参数：禁止长度>=h 的同碱基连跑（最大均聚物长度=h-1）。",
    )
    parser.add_argument("--gc-window", type=int, default=6)
    parser.add_argument("--gc-min", type=int, default=None)
    parser.add_argument("--gc-max", type=int, default=None)
    parser.add_argument(
        "--forbidden-motifs", default=None,
        help="逗号分隔的用户自定义禁用 motif 列表，如 AGCT,GACGC",
    )
    args = parser.parse_args()

    if not 1 <= int(args.K) < int(args.N) <= 255:
        raise ValueError("GF(256) RS parameters must satisfy 1 <= K < N <= 255")
    raw = np.fromfile(args.data, dtype=np.uint8)
    if raw.size != int(args.height) * int(args.width):
        raise ValueError(
            f"input has {raw.size} bytes, expected {args.height}x{args.width}"
        )
    image = raw.reshape(args.height, args.width)
    transformed, restored, transform_description = apply_transform(
        image, args.transform,
    )
    if not np.array_equal(restored, image):
        raise AssertionError(f"{args.transform} round trip failed")
    residual = transformed.reshape(-1)
    k, n = int(args.K), int(args.N)
    padded, grid, source_padding = build_rs_grid(residual, n, k)
    columns = int(padded.shape[1])
    stream = grid.reshape(-1).astype(np.int64)

    sys.path.insert(0, os.path.abspath(args.code_dir))
    from stage3_tans import (
        SIGNATURE_MODE_ADAPTIVE_V1,
        SIGNATURE_MODE_FIXED_0,
        SIGNATURE_MODE_FIXED_2,
        SIGNATURE_MODE_MIXED_V2, TERMINAL_MODE_JOINT_STATE_V1,
        build_constraint_dfa, build_global_table, tans_encode_fixed_length_dna,
    )
    if args.signature_count is not None:
        if args.signature_mode == "adaptive_v1":
            raise ValueError(
                "--signature-count cannot be combined with "
                "--signature-mode adaptive_v1"
            )
        signature_mode = {
            0: SIGNATURE_MODE_FIXED_0,
            1: SIGNATURE_MODE_MIXED_V2,
            2: SIGNATURE_MODE_FIXED_2,
        }[int(args.signature_count)]
    else:
        signature_mode = (
            SIGNATURE_MODE_ADAPTIVE_V1
            if args.signature_mode == "adaptive_v1"
            else SIGNATURE_MODE_MIXED_V2
        )
    from stage3_tans_decoder_macro import decode_fixed_length_macro

    forbidden_motifs = (
        [m.strip() for m in args.forbidden_motifs.split(",") if m.strip()]
        if args.forbidden_motifs else None
    )
    dfa = build_constraint_dfa(
        h=args.homopolymer_h, gc_window=args.gc_window,
        gc_min=args.gc_min, gc_max=args.gc_max,
        forbidden_motifs=forbidden_motifs,
    )
    if args.shared_table:
        # 路线A：编解码都用数据集【共用表】(残差域),不现算、不走 header。
        import json as _json
        with open(args.shared_table, encoding="utf-8") as _h:
            _st = _json.load(_h)["step3_final_shared_table"]
        global_table = (
            np.asarray(_st["symbols"], dtype=np.int64),
            np.asarray(_st["freqs"], dtype=np.int64),
            np.asarray(_st["starts"], dtype=np.int64),
            int(_st["table_log"]),
        )
        args.gtable_log = int(_st["table_log"])
        decode_global_table = global_table
    else:
        global_table = build_global_table(
            grid, table_log=args.gtable_log, alphabet_size=256,
        )
        # ── 方案1：解码器从 header DNA 还原全局表,而不是从原图重算 ──
        from table_header import encode_table_header, decode_table_header
        header_dna, header_info = encode_table_header(
            global_table, dfa, dna_length=args.dna_length,
            parity_bits=args.pb, signature_mode=signature_mode,
        )
        decode_global_table = decode_table_header(
            header_dna, dfa,
            header_stream_length=header_info["header_stream_length"],
            dna_length=args.dna_length, parity_bits=args.pb,
            signature_mode=signature_mode,
        )
        if not (
            np.array_equal(decode_global_table[0], global_table[0])
            and np.array_equal(decode_global_table[1], global_table[1])
            and np.array_equal(decode_global_table[2], global_table[2])
            and int(decode_global_table[3]) == int(global_table[3])
        ):
            raise AssertionError("header 还原的全局表与编码表不一致")
    with open(args.oligos, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    global_max_inner_symbols = max(
        int(row["inner_symbol_count"]) for row in rows
    )
    index_symbol_count = required_index_symbols(stream.size)
    index_fields = [
        f"index_symbol_{offset}" for offset in range(index_symbol_count)
    ]
    if not rows or any(field not in rows[0] for field in index_fields):
        raise ValueError(
            "oligos.csv index width does not match the selected data and RS parameters"
        )
    retry_result = (
        args.retry_clean_nonexact_from or args.retry_ids_nonexact_from
    )
    if retry_result:
        with open(retry_result, encoding="utf-8") as handle:
            prior = json.load(handle)
        retry_clean = bool(args.retry_clean_nonexact_from)
        selected = np.asarray(sorted({
            int(row["oligo_index"])
            for row in prior.get("rows", [])
            if (
                int(row.get("event_count", -1)) == 0
                if retry_clean else int(row.get("event_count", 0)) > 0
            )
            and row.get("status") != "exact"
            and (
                args.retry_status == "nonexact"
                or row.get("status") == args.retry_status
            )
        }), dtype=np.int64)
        if selected.size == 0 or selected[0] < 0 or selected[-1] >= len(rows):
            raise ValueError(
                "--retry-clean-nonexact-from selected no valid oligos"
            )
        args.cases = int(selected.size)
        selection_description = (
            ("clean" if retry_clean else "IDS-affected")
            + " non-exact oligos selected from "
            + os.path.abspath(retry_result)
        )
    elif args.all:
        selected = np.arange(len(rows), dtype=np.int64)
        args.cases = int(selected.size)
        selection_description = "all indexed oligos"
    elif args.indices.strip():
        selected = np.asarray(sorted({
            int(value) for value in args.indices.split(",") if value.strip()
        }), dtype=np.int64)
        if selected.size == 0 or selected[0] < 0 or selected[-1] >= len(rows):
            raise ValueError("--indices values must identify actual oligos")
        args.cases = int(selected.size)
        selection_description = "explicit actual oligo indices"
    else:
        if args.cases > len(rows):
            raise ValueError("--cases exceeds available oligos")
        select_rng = np.random.default_rng(args.seed)
        selected = np.sort(select_rng.choice(len(rows), args.cases, replace=False))
        selection_description = (
            f"uniform without replacement from {len(rows)} actual oligos"
        )

    cases = []
    event_totals = {"insertions": 0, "deletions": 0, "substitutions": 0}
    for index in selected.tolist():
        row = rows[index]
        start, end = int(row["start_symbol"]), int(row["end_symbol_exclusive"])
        expected = stream[start:end]
        index_symbols = np.asarray(
            [int(row[field]) for field in index_fields], dtype=np.int64,
        )
        expected_inner = np.concatenate((expected, index_symbols[::-1]))
        dna, meta = tans_encode_fixed_length_dna(
            expected_inner, dfa, target_dna_length=args.dna_length,
            parity_bits=args.pb,
            global_table=global_table, include_model_in_meta=False,
            signature_mode=signature_mode,
            terminal_mode=TERMINAL_MODE_JOINT_STATE_V1,
        )
        if (
            dna != row["dna"]
            or int(meta["n_payload_symbols"]) != expected_inner.size
        ):
            raise AssertionError(f"oligo {index} could not be reproduced")
        # Per-oligo symbol count is deliberately withheld.  The decoder must
        # discover the terminal depth and then parse the fixed-width index.
        meta.pop("n_payload_symbols", None)
        meta.pop("sequence_length", None)
        meta["min_payload_symbols"] = index_symbol_count
        meta["max_payload_symbols"] = global_max_inner_symbols
        meta["linear_index_symbols"] = index_symbol_count
        meta["source_stream_length"] = int(stream.size)
        noisy, events = ids_channel(
            dna, args.pi, args.pd, args.ps,
            np.random.default_rng(args.seed + 1000003 * (index + 1)),
        )
        for name in event_totals:
            event_totals[name] += int(events[name])
        cases.append((index, start, expected, noisy, meta, events))

    # Deliberately destroy synthesis order.  Reconstruction below may only use
    # decoded linear indices, never the arrival/CSV order.
    np.random.default_rng(args.seed ^ 0x5A17).shuffle(cases)

    # ── Oligo dropout（整条链丢失，模拟测序/合成漏读）──
    # 被丢弃的 oligo 根本不进入解码；其符号在栅格里保持未知，交由列向 RS erasure 恢复。
    n_dropped = 0
    if args.dropout > 0:
        drop_rng = np.random.default_rng(args.seed ^ 0xD30F)
        keep_mask = drop_rng.random(len(cases)) >= args.dropout
        n_dropped = int((~keep_mask).sum())
        cases = [case for case, keep in zip(cases, keep_mask) if keep]
        print(f"oligo dropout: rate={args.dropout} dropped={n_dropped}/{len(keep_mask)} "
              f"decoding={len(cases)}", flush=True)

    _W.update({
        "decoder": decode_fixed_length_macro, "dfa": dfa,
        "global_table": decode_global_table, "beam": int(args.beam),
        "relative_cost_window": int(args.cost_window),
        "stream_length": int(stream.size),
        "short_lookahead": int(args.short_lookahead),
        "lookahead_pool_factor": float(args.lookahead_pool),
        "share_macro_transitions": bool(args.enable_macro_memo),
        "share_alignment_results": not args.disable_alignment_memo,
        "use_numpy_batch": not args.disable_numpy_batch,
        "index_symbols": int(index_symbol_count),
    })
    started = time.perf_counter()
    if args.workers <= 1:
        results = []
        for completed, case in enumerate(cases, 1):
            results.append(_decode_case(case))
            if completed % 100 == 0 or completed == len(cases):
                print(
                    f"decoded_oligos={completed}/{len(cases)} "
                    f"elapsed_seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    else:
        context = mp.get_context("fork")
        with context.Pool(args.workers) as pool:
            results = []
            iterator = pool.imap_unordered(_decode_case, cases, chunksize=1)
            for completed, result in enumerate(iterator, 1):
                results.append(result)
                if completed % 100 == 0 or completed == len(cases):
                    print(
                        f"decoded_oligos={completed}/{len(cases)} "
                        f"elapsed_seconds={time.perf_counter() - started:.1f}",
                        flush=True,
                    )

    expected_channel_events = float(args.dna_length) * (
        args.pd + (1.0 - args.pd) * args.ps + args.pi
    )
    cost_threshold = (
        int(args.cost_threshold) if args.cost_threshold is not None
        else int(2 * math.ceil(expected_channel_events) * 4)
    )
    feedback_iterations = (
        int(args.rs_feedback_iterations)
        if args.all and int(args.cases) == len(rows) else 0
    )
    _rs_t0 = time.perf_counter()
    reconstructed, known, reassembly, rs_recovery = reconstruct_with_rs_feedback(
        results, int(stream.size), grid, padded,
        cost_threshold=cost_threshold,
        max_iterations=feedback_iterations,
    )
    _rs_seconds = float(time.perf_counter() - _rs_t0)
    recovered_systematic = rs_recovery.pop("recovered_systematic")
    rs_recovery["systematic_matrix_exact"] = bool(
        np.array_equal(recovered_systematic, padded)
    )
    # ── 端到端逐字节校验(路线A 成功判据):RS 恢复的 systematic → 去padding →
    #    逆变换 → 逐字节 == 原图?这是真·端到端无损判据(比 RS 矩阵更严格)。
    try:
        rec_flat = np.asarray(recovered_systematic, dtype=np.uint8).reshape(-1)
        rec_residual = rec_flat[:rec_flat.size - int(source_padding)]
        if rec_residual.size == int(args.height) * int(args.width):
            rec_image = inverse_transform(
                rec_residual.reshape(int(args.height), int(args.width)),
                args.transform,
            )
            end_to_end_exact = bool(np.array_equal(
                np.asarray(rec_image, dtype=np.uint8), image))
        else:
            end_to_end_exact = False
    except Exception:
        end_to_end_exact = False
    rs_recovery["end_to_end_image_exact"] = end_to_end_exact
    # Keep the JSON bounded; payload values have already been consumed by the
    # index-only reassembler and are not side information for evaluation.
    if not args.keep_decoded_payload:
        for row in results:
            row.pop("decoded_payload", None)

    counts = {
        status: sum(row["status"] == status for row in results)
        for status in ("exact", "wrong", "erasure")
    }
    event_histogram = {}
    for row in results:
        key = str(row["event_count"])
        bucket = event_histogram.setdefault(
            key, {"trials": 0, "exact": 0, "wrong": 0, "erasure": 0},
        )
        bucket["trials"] += 1
        bucket[row["status"]] += 1
    summary = {
        "pipeline": (
            f"{args.transform} -> RS({n},{k}) -> indexed "
            f"{signature_mode}/joint_state_v1 fixed{args.dna_length}"
        ),
        "input_shape": [int(args.height), int(args.width)],
        "transform_name": args.transform,
        "transform": transform_description,
        "N": n, "K": k, "P": n - k,
        "pre_rs_matrix_shape": [k, columns],
        "post_rs_matrix_shape": [n, columns],
        "source_padding_bytes": int(source_padding),
        "signature_mode": signature_mode,
        "signature_bases_per_symbol": (
            None if signature_mode == SIGNATURE_MODE_ADAPTIVE_V1
            else {SIGNATURE_MODE_FIXED_0: 0, SIGNATURE_MODE_MIXED_V2: 1,
                  SIGNATURE_MODE_FIXED_2: 2}[signature_mode]
        ),
        "sampled_oligos": int(args.cases),
        "selection": selection_description,
        "selected_indices": selected.tolist(),
        "pi": args.pi, "pd": args.pd, "ps": args.ps,
        "oligo_dropout_rate": float(args.dropout),
        "oligos_dropped": int(n_dropped),
        "channel_definition": "per original base: deletion Bernoulli(pd); if retained substitution Bernoulli(ps); insertion Bernoulli(pi) after the position",
        "decoder_max_edits": None,
        "decoder_cumulative_error_budget": "unbounded; beam-ranked accumulated cost",
        "decoder_relative_cost_window": int(args.cost_window),
        "beam_width": int(args.beam),
        "short_macro_lookahead_nt": int(args.short_lookahead),
        "lookahead_pool_factor": float(args.lookahead_pool),
        "share_macro_transitions": bool(args.enable_macro_memo),
        "share_alignment_results": not args.disable_alignment_memo,
        "use_numpy_batch": not args.disable_numpy_batch,
        "index_symbols_per_oligo": int(index_symbol_count),
        "index_layout": (
            "payload followed by reversed big-endian index symbols; beam "
            "decodes the big-endian index first"
        ),
        "global_max_inner_symbols": int(global_max_inner_symbols),
        "input_order_shuffled_before_decode": True,
        "event_totals": event_totals,
        "mean_events_per_oligo": float(sum(event_totals.values()) / args.cases),
        "expected_channel_events_per_oligo": float(expected_channel_events),
        "cost_threshold_formula": "2*ceil(expected_channel_events_per_oligo)*4",
        "cost_threshold": int(cost_threshold),
        "rs_feedback_requested_iterations": int(args.rs_feedback_iterations),
        "rs_feedback_enabled_for_complete_oligo_set": bool(
            feedback_iterations > 0
        ),
        "counts": counts,
        "exact_rate": counts["exact"] / max(1, len(results)),
        "wrong_rate": counts["wrong"] / max(1, len(results)),
        "erasure_rate": counts["erasure"] / max(1, len(results)),
        "event_count_results": event_histogram,
        "index_reassembly": reassembly,
        "rs_recovery": rs_recovery,
        "elapsed_seconds": float(time.perf_counter() - started),
        "rs_recovery_seconds": _rs_seconds,
        "msva_decode_seconds": float(time.perf_counter() - started - _rs_seconds),
        "mean_decode_seconds": float(np.mean([row["seconds"] for row in results])),
        "rows": sorted(results, key=lambda row: row["oligo_index"]),
    }
    if args.out is None:
        args.out = "outputs/us_mod256_ids_p003_unbounded/result.json"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"output={os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
