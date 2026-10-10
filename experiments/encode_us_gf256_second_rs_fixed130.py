#!/usr/bin/env python3
"""Configurable mod-256 predictor -> RS(N,K) -> indexed fixed-length oligos."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import reedsolo


def required_index_symbols(stream_length: int) -> int:
    """Minimum fixed base-256 width covering positions in a symbol stream."""
    maximum = max(0, int(stream_length) - 1)
    return max(1, (maximum.bit_length() + 7) // 8)


def linear_index_symbols(position: int, symbol_count: int) -> np.ndarray:
    """Encode a linear RS-stream position as fixed-width big-endian bytes."""
    position = int(position)
    symbol_count = int(symbol_count)
    if symbol_count < 1 or not 0 <= position < (1 << (8 * symbol_count)):
        raise ValueError("linear position does not fit the selected index width")
    return np.asarray(
        [
            (position >> (8 * (symbol_count - 1 - offset))) & 0xFF
            for offset in range(symbol_count)
        ],
        dtype=np.int64,
    )


def entropy_u8(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    counts = np.bincount(flat, minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / flat.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def mod256_first(values: np.ndarray, axis: int) -> np.ndarray:
    """First difference along one axis, resetting at every row/column."""
    work = np.asarray(values, dtype=np.int16)
    out = work.copy()
    if axis == 1 and work.shape[1] > 1:
        out[:, 1:] = work[:, 1:] - work[:, :-1]
    elif axis == 0 and work.shape[0] > 1:
        out[1:, :] = work[1:, :] - work[:-1, :]
    elif axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    return (out % 256).astype(np.uint8)


def inverse_mod256_first(residual: np.ndarray, axis: int) -> np.ndarray:
    return (np.cumsum(residual.astype(np.int64), axis=axis) % 256).astype(np.uint8)


def mod256_row_second(image: np.ndarray) -> np.ndarray:
    """Second difference within each row; no state crosses a row boundary."""
    work = np.asarray(image, dtype=np.int16)
    out = work.copy()
    if work.shape[1] > 1:
        out[:, 1] = work[:, 1] - work[:, 0]
    if work.shape[1] > 2:
        out[:, 2:] = work[:, 2:] - 2 * work[:, 1:-1] + work[:, :-2]
    return (out % 256).astype(np.uint8)


def inverse_mod256_row_second(residual: np.ndarray) -> np.ndarray:
    work = np.asarray(residual, dtype=np.int64)
    out = np.empty_like(work)
    out[:, 0] = work[:, 0]
    if work.shape[1] > 1:
        out[:, 1] = work[:, 1] + out[:, 0]
    for column in range(2, work.shape[1]):
        out[:, column] = (
            work[:, column] + 2 * out[:, column - 1] - out[:, column - 2]
        )
    return (out % 256).astype(np.uint8)


def apply_transform(image: np.ndarray, mode: str):
    if mode == "raw":
        residual = image.copy()
        restored = residual
        description = "none; raw row-major bytes"
    elif mode == "row_first":
        residual = mod256_first(image, axis=1)
        restored = inverse_mod256_first(residual, axis=1)
        description = "Z/256Z first difference within each row"
    elif mode == "column_first":
        residual = mod256_first(image, axis=0)
        restored = inverse_mod256_first(residual, axis=0)
        description = "Z/256Z first difference within each column"
    elif mode == "row_second":
        residual = mod256_row_second(image)
        restored = inverse_mod256_row_second(residual)
        description = "Z/256Z second difference within each row"
    else:
        raise ValueError(f"unknown transform: {mode}")
    return residual, restored, description


def inverse_transform(residual: np.ndarray, mode: str) -> np.ndarray:
    """把任意残差矩阵逆变换回图像(端到端逐字节校验用,与 apply_transform 对应)。"""
    if mode == "raw":
        return np.asarray(residual, dtype=np.uint8).copy()
    if mode == "row_first":
        return inverse_mod256_first(residual, axis=1)
    if mode == "column_first":
        return inverse_mod256_first(residual, axis=0)
    if mode == "row_second":
        return inverse_mod256_row_second(residual)
    raise ValueError(f"unknown transform: {mode}")


def build_rs_grid(source_symbols: np.ndarray, n: int, k: int):
    """Pad to K rows and encode every matrix column as RS(N,K)."""
    n, k = int(n), int(k)
    if not 1 <= k < n <= 255:
        raise ValueError("GF(256) RS parameters must satisfy 1 <= K < N <= 255")
    source = np.asarray(source_symbols, dtype=np.uint8).reshape(-1)
    columns = int(math.ceil(source.size / k))
    source_padding = k * columns - source.size
    systematic = np.pad(source, (0, source_padding)).reshape(k, columns)
    codec = reedsolo.RSCodec(n - k, nsize=n)
    grid = np.empty((n, columns), dtype=np.uint8)
    grid[:k] = systematic
    for column in range(columns):
        encoded_column = np.frombuffer(
            codec.encode(systematic[:, column].tobytes()), dtype=np.uint8,
        )
        if encoded_column.size != n:
            raise AssertionError("RS encoder did not produce exactly N symbols")
        grid[:, column] = encoded_column
    return systematic, grid, int(source_padding)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/us_full.bin")
    parser.add_argument("--code-dir", default="codec")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--transform",
        choices=("raw", "row_first", "row_second", "column_first"),
        default="row_second",
    )
    parser.add_argument("--height", type=int, default=471)
    parser.add_argument("--width", type=int, default=562)
    parser.add_argument("--N", type=int, default=255)
    parser.add_argument("--K", type=int, default=200)
    parser.add_argument("--dna-length", type=int, default=130)
    parser.add_argument("--pb", type=int, default=3)
    parser.add_argument("--gtable-log", type=int, default=14)
    parser.add_argument(
        "--shared-table", default=None,
        help="数据集共用表 json(dataset_shared_table.py 产出)。给定则用其 "
             "step3_final_shared_table 作为全局表(残差域),不再从本图现算。"
             "此时 --gtable-log 被表内的 table_log 覆盖。",
    )
    parser.add_argument("--candidate-symbols", type=int, default=128)
    parser.add_argument(
        "--signature-mode", choices=("mixed_v2", "adaptive_v1"),
        default="mixed_v2",
    )
    parser.add_argument(
        "--signature-count", type=int, choices=(0, 1, 2), default=None,
        help=(
            "fixed path-verification bases per internal symbol; 0 and 2 are "
            "ablation modes, 1 is bit-identical to mixed_v2"
        ),
    )
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
    if raw.size != args.height * args.width:
        raise ValueError(
            f"input has {raw.size} bytes, expected {args.height}x{args.width}"
        )
    image = raw.reshape(args.height, args.width)
    residual, restored, transform_description = apply_transform(
        image, args.transform,
    )
    roundtrip = np.array_equal(restored, image)
    if not roundtrip:
        raise AssertionError(f"{args.transform} round trip failed")
    if args.out_dir is None:
        args.out_dir = (
            f"outputs/us_{args.transform}_rs{args.N}_k{args.K}_"
            f"fixed{args.dna_length}_"
            f"sig{args.signature_count if args.signature_count is not None else args.signature_mode}"
        )

    source_symbols = residual.reshape(-1)
    padded, grid, source_padding = build_rs_grid(
        source_symbols, args.N, args.K,
    )
    columns = int(padded.shape[1])
    parity_symbols = args.N - args.K

    code_dir = os.path.abspath(args.code_dir)
    sys.path.insert(0, code_dir)
    from stage3_tans import (
        SIGNATURE_MODE_ADAPTIVE_V1,
        SIGNATURE_MODE_FIXED_0,
        SIGNATURE_MODE_FIXED_2,
        SIGNATURE_MODE_MIXED_V2,
        TERMINAL_MODE_JOINT_STATE_V1,
        build_constraint_dfa,
        build_global_table,
        tans_encode_fixed_length_dna,
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
        # 路线A：用数据集【共用表】(残差域)编整个 post-RS grid(含校验行)。
        # 共用表是数据集常量,不随本图现算;table_log 以表内为准。
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
    else:
        global_table = build_global_table(
            grid, table_log=args.gtable_log, alphabet_size=256,
        )
    symbol_stream = grid.reshape(-1).astype(np.int64)
    index_symbol_count = required_index_symbols(symbol_stream.size)

    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    cursor = 0
    started = time.perf_counter()
    while cursor < symbol_stream.size:
        start = int(cursor)
        max_count = min(
            int(args.candidate_symbols), int(symbol_stream.size) - start,
        )
        index_symbols = linear_index_symbols(start, index_symbol_count)

        def encode_count(count: int):
            inner = np.concatenate((
                symbol_stream[start:start + int(count)], index_symbols[::-1],
            ))
            encoded_dna, encoded_meta = tans_encode_fixed_length_dna(
                inner, dfa, target_dna_length=args.dna_length,
                parity_bits=args.pb, global_table=global_table,
                include_model_in_meta=False,
                signature_mode=signature_mode,
                terminal_mode=TERMINAL_MODE_JOINT_STATE_V1,
            )
            fits = int(encoded_meta["n_payload_symbols"]) == int(inner.size)
            return encoded_dna, encoded_meta, fits

        # The encoder takes an input suffix.  Appending the index guarantees
        # that all index symbols are attempted before any payload symbol.
        _probe_dna, probe_meta, _probe_fits = encode_count(max_count)
        count = min(
            max_count,
            max(1, int(probe_meta["n_payload_symbols"]) - index_symbol_count),
        )
        dna, meta, fits = encode_count(count)
        while not fits and count > 1:
            count -= 1
            dna, meta, fits = encode_count(count)
        if not fits:
            raise RuntimeError(f"indexed fixed encoder made no progress at {start}")
        # Changing the payload also changes the terminal state; reclaim any
        # immediately available room that the long suffix probe underestimated.
        while count < max_count:
            next_dna, next_meta, next_fits = encode_count(count + 1)
            if not next_fits:
                break
            count += 1
            dna, meta = next_dna, next_meta

        end = start + count
        records.append({
            "start_symbol": start,
            "end_symbol_exclusive": end,
            "symbol_count": count,
            "inner_symbol_count": count + index_symbol_count,
            "start_rs_row": start // columns,
            "start_rs_col": start % columns,
            "end_rs_row": (end - 1) // columns,
            "end_rs_col": (end - 1) % columns,
            "payload_nt": int(meta["payload_emissions"]),
            "renorm_nt": int(sum(meta["payload_renorm_counts"])),
            "signature_nt": int(meta["signature_emissions"]),
            "trailer_nt": int(meta["trailer_emissions"]),
            "tail_length_nt": int(meta["trailer_emissions"]),
            "dna": dna,
        })
        records[-1].update({
            f"index_symbol_{offset}": int(value)
            for offset, value in enumerate(index_symbols)
        })
        cursor = end
        if len(records) % 1000 == 0:
            print(
                f"encoded_oligos={len(records)} remaining_symbols="
                f"{symbol_stream.size - cursor}", flush=True,
            )

    cursor = 0
    for index, record in enumerate(records):
        if record["start_symbol"] != cursor:
            raise AssertionError("RS symbol stream coverage is not contiguous")
        if len(record["dna"]) != args.dna_length:
            raise AssertionError("DNA sequence is not fixed length")
        record["oligo_index"] = index
        cursor = record["end_symbol_exclusive"]
    if cursor != symbol_stream.size:
        raise AssertionError("RS symbol stream coverage is incomplete")

    csv_path = os.path.join(args.out_dir, "oligos.csv")
    index_fields = [
        f"index_symbol_{offset}" for offset in range(index_symbol_count)
    ]
    fields = [
        "oligo_index", "start_symbol", "end_symbol_exclusive", "symbol_count",
        "inner_symbol_count", *index_fields,
        "start_rs_row", "start_rs_col", "end_rs_row", "end_rs_col",
        "payload_nt", "renorm_nt", "signature_nt",
        "tail_length_nt", "trailer_nt", "dna",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    counts = np.asarray([record["symbol_count"] for record in records])
    inner_counts = np.asarray([record["inner_symbol_count"] for record in records])
    payload_nt = np.asarray([record["payload_nt"] for record in records])
    trailer_nt = np.asarray([record["trailer_nt"] for record in records])

    # ── 全局表 side information ──
    # 逐图表:把表编成自解 header oligo,计入 total_nt(数据依赖 side info)。
    # 共用表(路线A):表是数据集常量,不逐图存 → header_nt=0,净密度分母=纯payload。
    from table_header import encode_table_header
    if args.shared_table:
        header_dna, header_info = [], {
            "header_nt": 0, "header_bytes": 0, "n_header_oligos": 0,
            "boot_model": "dataset_shared_table_not_stored_per_image",
        }
    else:
        header_dna, header_info = encode_table_header(
            global_table, dfa, dna_length=args.dna_length,
            parity_bits=args.pb, signature_mode=signature_mode,
        )
    header_csv_path = os.path.join(args.out_dir, "header_oligos.csv")
    with open(header_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["header_index", "dna"])
        writer.writeheader()
        for header_index, header_seq in enumerate(header_dna):
            writer.writerow({"header_index": header_index, "dna": header_seq})

    payload_nt_total = len(records) * args.dna_length
    header_nt_total = int(header_info["header_nt"])
    total_nt = payload_nt_total + header_nt_total
    systematic_end = args.K * columns
    region_records = {
        "systematic": [r for r in records if r["end_symbol_exclusive"] <= systematic_end],
        "crossing": [r for r in records if r["start_symbol"] < systematic_end < r["end_symbol_exclusive"]],
        "parity": [r for r in records if r["start_symbol"] >= systematic_end],
    }
    region_summary = {}
    for name, subset in region_records.items():
        region_counts = np.asarray([r["symbol_count"] for r in subset], dtype=np.int64)
        region_summary[name] = {
            "oligo_count": len(subset),
            "symbol_count": int(region_counts.sum()) if len(subset) else 0,
            "symbols_per_oligo_mean": float(region_counts.mean()) if len(subset) else None,
            "symbols_per_oligo_median": float(np.median(region_counts)) if len(subset) else None,
        }
    summary = {
        "input_file": os.path.abspath(args.data),
        "input_shape": [args.height, args.width],
        "input_bytes": int(raw.size),
        "transform_name": args.transform,
        "transform": transform_description,
        "transform_roundtrip_verified": bool(roundtrip),
        "raw_entropy_bits_per_symbol": entropy_u8(image),
        "coded_transform_entropy_bits_per_symbol": entropy_u8(residual),
        "N": args.N,
        "K": args.K,
        "P": parity_symbols,
        "pre_rs_matrix_shape": [args.K, columns],
        "source_padding_bytes": int(source_padding),
        "post_rs_matrix_shape": [args.N, columns],
        "post_rs_symbols": int(grid.size),
        "systematic_entropy_bits_per_symbol": entropy_u8(grid[:args.K]),
        "parity_entropy_bits_per_symbol": entropy_u8(grid[args.K:]),
        "post_rs_global_entropy_bits_per_symbol": entropy_u8(grid),
        "global_table_scope": f"entire post-RS {args.N}x{columns} grid",
        "global_table_is_data_dependent_side_information": True,
        "global_table_header": {
            "stored_in_dna": True,
            "boot_model": header_info["boot_model"],
            "header_bytes": int(header_info["header_bytes"]),
            "n_header_oligos": int(header_info["n_header_oligos"]),
            "header_nt": header_nt_total,
            "header_csv": os.path.abspath(header_csv_path),
            "note": (
                "全局频率表是数据依赖 side information，已用零-SI 均匀模型编成 "
                "header oligo 存入 DNA 并计入 total_dna_nt；解码器从 header 还原，"
                "不依赖原图。"
            ),
        },
        "signature_mode": signature_mode,
        "signature_bases_per_symbol": (
            None if signature_mode == SIGNATURE_MODE_ADAPTIVE_V1
            else {SIGNATURE_MODE_FIXED_0: 0, SIGNATURE_MODE_MIXED_V2: 1,
                  SIGNATURE_MODE_FIXED_2: 2}[signature_mode]
        ),
        "terminal_mode": TERMINAL_MODE_JOINT_STATE_V1,
        "parity_bits": args.pb,
        "dna_length_nt": args.dna_length,
        "oligo_count": len(records),
        "payload_dna_nt": int(payload_nt_total),
        "parity_dna_nt": int(sum(r["signature_nt"] for r in records)),
        "parity_ratio_R_par": float(sum(r["signature_nt"] for r in records) / total_nt),
        "renorm_dna_nt": int(sum(r["renorm_nt"] for r in records)),
        "trailer_dna_nt": int(sum(r["trailer_nt"] for r in records)),
        "header_dna_nt": int(header_nt_total),
        "total_dna_nt": int(total_nt),
        "side_information_overhead_ratio": float(header_nt_total / total_nt),
        "source_bits_per_dna_nt": float(raw.size * 8 / total_nt),
        "source_bits_per_dna_nt_payload_only": float(raw.size * 8 / payload_nt_total),
        "post_rs_symbols_per_oligo": {
            "min": int(counts.min()), "max": int(counts.max()),
            "mean": float(counts.mean()), "median": float(np.median(counts)),
        },
        "inner_symbols_per_oligo_including_index": {
            "min": int(inner_counts.min()), "max": int(inner_counts.max()),
            "mean": float(inner_counts.mean()),
            "median": float(np.median(inner_counts)),
        },
        "regions": region_summary,
        "payload_nt": {
            "min": int(payload_nt.min()), "max": int(payload_nt.max()),
            "mean": float(payload_nt.mean()), "median": float(np.median(payload_nt)),
        },
        "trailer_nt": {
            "min": int(trailer_nt.min()), "max": int(trailer_nt.max()),
            "mean": float(trailer_nt.mean()), "median": float(np.median(trailer_nt)),
        },
        "index_encoded_in_dna": True,
        "index_layout": (
            "payload followed by reversed big-endian index symbols; reverse "
            "rANS exposes the big-endian index first"
        ),
        "index_symbols_per_oligo": int(index_symbol_count),
        "index_representation": (
            f"fixed-width big-endian base-256 integer with {index_symbol_count} symbols"
        ),
        "coverage_verified": [0, int(grid.size)],
        "elapsed_seconds": float(time.perf_counter() - started),
        "oligos_csv": os.path.abspath(csv_path),
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()
