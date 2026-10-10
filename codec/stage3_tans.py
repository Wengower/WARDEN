from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from yaogan_common import ensure_dir

DNA_BASES: Tuple[str, ...] = ("A", "C", "G", "T")
GC_BASES = {"G", "C"}


@dataclass(frozen=True)
class ConstraintDFA:
    states: Tuple[str, ...]
    start_state: str
    transitions: Dict[str, Dict[str, str]]
    max_hist: int
    homopolymer_h: int
    gc_window: int
    gc_min: int
    gc_max: int
    forbidden_motifs: Tuple[str, ...] = ()


def _to_tuple_text(seq: np.ndarray) -> str:
    arr = np.asarray(seq, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return "()"
    return "(" + ",".join(str(int(v)) for v in arr.tolist()) + ")"


def _normalize_counts(counts: np.ndarray, table_size: int) -> np.ndarray:
    raw = np.asarray(counts, dtype=np.float64)
    s = float(raw.sum())
    if s <= 0:
        raise ValueError("count 总和必须大于 0")

    scaled = raw / s * float(table_size)
    norm = np.floor(scaled).astype(np.int64)
    norm[norm < 1] = 1
    diff = int(table_size - norm.sum())

    frac = scaled - np.floor(scaled)
    order_inc = np.argsort(frac)[::-1]
    order_dec = np.argsort(frac)

    i = 0
    while diff > 0:
        norm[order_inc[i % len(norm)]] += 1
        diff -= 1
        i += 1
    i = 0
    while diff < 0:
        idx = order_dec[i % len(norm)]
        if norm[idx] > 1:
            norm[idx] -= 1
            diff += 1
        i += 1
        if i > 10 * len(norm):
            break
    return norm


## ── Preset canonical table (Laplace model, zero SI) ────────────────

# 算法规范常量 — 编码器与解码器预先约定，不传输
PRESET_DELTA_MIN = -255
PRESET_DELTA_MAX = 255
PRESET_TABLE_LOG = 12
PRESET_PARITY_BITS = 4
PRESET_ANS_MEMORY_K = 16
PRESET_INITIAL_ANS = 1 << PRESET_TABLE_LOG
PRESET_TERMINAL_ANS = 0          # tail flush 终止常量（编解码器共知）：整除最终归零
PRESET_LAPLACE_B = 21.0          # Laplace scale (typical for 8-bit delta)
PRESET_SYMBOLS_PER_OLIGO = 20    # 解码搜索起点（实际编码数由 DNA 预算自适应决定，可多于/少于此值）
PRESET_N_FLUSH = 1               # parity flush 固定 1 个符号（大字母表保证）
PRESET_DNA_LENGTH = 125          # 固定 DNA 长度（碱基数），编码器自动调整符号数以填满
PRESET_MAX_TAIL_BASES = 14        # tail flush 理论最大保留的碱基数（确保预算充足）
PRESET_N_FLUSH_SYMBOLS = 1       # parity flush 固定符号数（编解码器共知）

# Current per-symbol signature modes.  ``fixed_0`` is the no-signature
# ablation, ``mixed_v2`` is the released one-signature codec, and ``fixed_2``
# repeats the same deterministic signature construction twice.  ``adaptive_v1``
# adds signatures when rANS emits too few bases.  The selected mode is codec
# metadata, so no per-oligo side information is needed.
SIGNATURE_MODE_FIXED_0 = "fixed_0"
SIGNATURE_MODE_MIXED_V2 = "mixed_v2"
SIGNATURE_MODE_FIXED_2 = "fixed_2"
SIGNATURE_MODE_ADAPTIVE_V1 = "adaptive_v1"
VALID_SIGNATURE_MODES = (
    SIGNATURE_MODE_FIXED_0,
    SIGNATURE_MODE_MIXED_V2,
    SIGNATURE_MODE_FIXED_2,
    SIGNATURE_MODE_ADAPTIVE_V1,
)
TERMINAL_MODE_JOINT_STATE_V1 = "joint_state_v1"
SIGNATURE_HISTORY_INIT = 0x6A09E667F3BCC909
_U64_MASK = (1 << 64) - 1


def _mix64(x: int) -> int:
    """Stable SplitMix64 finalizer (unlike Python ``hash``, reproducible)."""
    x = int(x) & _U64_MASK
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _U64_MASK
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _U64_MASK
    x ^= x >> 31
    return x & _U64_MASK


def update_signature_history(history: int, sid: int, symbol_step: int) -> int:
    """Update the rolling state used only to decorrelate signature checks."""
    x = int(history)
    x ^= (int(sid) + 0x9E3779B97F4A7C15) & _U64_MASK
    x ^= ((int(symbol_step) + 1) * 0xD6E8FEB86659FD93) & _U64_MASK
    return _mix64(x)


def adaptive_signature_count(renorm_emissions: int) -> int:
    """Safe no-anchor v1 allocation from the symbol's rANS output.

    Every symbol retains at least one nonlinear signature because a run of
    zero-signature macros can let the correct forward-beam path be pruned
    before joint-trailer verification.  Extra signatures make every short
    symbol macro at least three bases long.  The deterministic rule requires
    no side information.
    """
    count = int(renorm_emissions)
    return int(max(1, 3 - count))


def signature_count_for_mode(mode: str, renorm_emissions: int) -> int:
    """Return the number of path-verification bases for one symbol."""
    if mode == SIGNATURE_MODE_FIXED_0:
        return 0
    if mode == SIGNATURE_MODE_MIXED_V2:
        return 1
    if mode == SIGNATURE_MODE_FIXED_2:
        return 2
    if mode == SIGNATURE_MODE_ADAPTIVE_V1:
        return adaptive_signature_count(renorm_emissions)
    raise ValueError(f"unknown signature mode: {mode}")


def signature_index(
    ans_state: int,
    parity: int,
    sid: int,
    symbol_step: int,
    dna_position: int,
    history: int,
    radix: int,
    mode: str = SIGNATURE_MODE_MIXED_V2,
    signature_ordinal: int = 0,
    renorm_emissions: int = 0,
) -> Tuple[int, int]:
    """Return ``(allowed-base rank, updated history)`` for one signature."""
    radix = int(radix)
    if radix <= 0:
        raise ValueError("signature radix must be positive")
    if mode not in VALID_SIGNATURE_MODES:
        raise ValueError(f"unknown signature mode: {mode}")
    new_history = update_signature_history(history, sid, symbol_step)
    x = int(ans_state) & _U64_MASK
    x ^= ((int(parity) + 1) * 0xA24BAED4963EE407) & _U64_MASK
    x ^= ((int(sid) + 1) * 0x9FB21C651E98DF25) & _U64_MASK
    x ^= ((int(symbol_step) + 1) * 0xC13FA9A902A6328F) & _U64_MASK
    x ^= ((int(dna_position) + 1) * 0x91E10DA5C79E7B1D) & _U64_MASK
    if mode == SIGNATURE_MODE_ADAPTIVE_V1:
        x ^= ((int(signature_ordinal) + 1) * 0xDB4F0B9175AE2165) & _U64_MASK
        x ^= ((int(renorm_emissions) + 1) * 0xBBE0563303A4615F) & _U64_MASK
    x ^= new_history
    u = _mix64(x)
    # Multiply-high reduction avoids modulo bias for radix 2/3/4.
    idx = (u * radix) >> 64
    return int(idx), int(new_history)


def build_preset_table(
    table_log: int = PRESET_TABLE_LOG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """构建 Laplace 预设频率表 — 字母表 [-255, 255], 共 511 个符号.

    返回 (symbols, freqs, starts, table_log)，与 _build_shared_symbol_table
    格式完全相同，但不依赖任何数据。编码器和解码器各自调用此函数即可
    重建完全相同的表，无需传输 side information。
    """
    symbols = np.arange(PRESET_DELTA_MIN, PRESET_DELTA_MAX + 1, dtype=np.int64)
    alpha = int(symbols.size)  # 511

    # Laplace(μ=0, b=PRESET_LAPLACE_B) 离散化
    raw_probs = np.exp(-np.abs(symbols.astype(np.float64)) / PRESET_LAPLACE_B)
    raw_probs /= raw_probs.sum()

    # 归一化到 table_size
    table_log = int(max(4, min(int(table_log), 15)))
    min_log_by_alpha = int(np.ceil(np.log2(max(2, alpha))))
    table_log = int(max(table_log, min_log_by_alpha))
    table_size = 1 << table_log

    # 将概率转换为整数计数
    counts = np.maximum(np.round(raw_probs * table_size).astype(np.int64), 1)
    freqs = _normalize_counts(counts, table_size)
    starts = np.concatenate(([0], np.cumsum(freqs[:-1], dtype=np.int64)))
    return symbols, freqs.astype(np.int64), starts.astype(np.int64), int(table_log)


def build_global_table(
    data_bytes,
    table_log: int = 12,
    alphabet_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """从整个符号集的字节直方图构建**全局** tANS 频率表。

    返回 (symbols, freqs, starts, table_log)，symbols = arange(alphabet_size)（全字母表，
    未出现的符号计数记 1 以保证可解）。这样每条 oligo 编码时共用此表，概率表只作为
    **一次性 side information**（约 alphabet_size×table_log 比特，摊到整源可忽略），
    而不再随每条 oligo 携带。

    ``data_bytes`` 可以是 bytes，也可以是 uint8/整数 ndarray。因此可以在
    RS 编码后直接对整个 ``grid`` 建表。
    """
    table_log = int(max(4, min(int(table_log), 15)))
    min_log = int(math.ceil(math.log2(max(2, alphabet_size))))
    table_log = int(max(table_log, min_log))
    table_size = 1 << table_log
    if isinstance(data_bytes, np.ndarray):
        arr = np.asarray(data_bytes, dtype=np.uint8).reshape(-1)
    else:
        arr = np.frombuffer(bytes(data_bytes), np.uint8)
    counts = np.bincount(arr, minlength=alphabet_size).astype(np.int64)[:alphabet_size]
    counts = np.maximum(counts, 1)               # 未出现符号也留 1，保证全字母表可编
    freqs = _normalize_counts(counts, table_size)
    starts = np.concatenate(([0], np.cumsum(freqs[:-1], dtype=np.int64)))
    symbols = np.arange(alphabet_size, dtype=np.int64)
    return symbols, freqs.astype(np.int64), starts.astype(np.int64), int(table_log)


def _default_gc_bounds(gc_window: int) -> Tuple[int, int]:
    if gc_window <= 0:
        return 0, 0
    gc_min = int(math.floor(0.40 * gc_window))
    gc_max = int(math.ceil(0.60 * gc_window))
    return gc_min, gc_max


def build_constraint_dfa(
    h: int,
    gc_window: int,
    gc_min: int | None = None,
    gc_max: int | None = None,
    forbidden_motifs: List[str] | None = None,
) -> ConstraintDFA:
    h = int(max(2, h))
    gc_window = int(max(0, gc_window))
    if gc_window > 0:
        if gc_min is None or gc_max is None:
            _min, _max = _default_gc_bounds(gc_window)
            gc_min = _min if gc_min is None else int(gc_min)
            gc_max = _max if gc_max is None else int(gc_max)
        else:
            gc_min = int(gc_min)
            gc_max = int(gc_max)
    else:
        gc_min, gc_max = 0, 0

    # 规范化 forbidden motifs：大写，去重，过滤空串
    if forbidden_motifs:
        _fm = tuple(sorted(set(m.upper() for m in forbidden_motifs if m)))
    else:
        _fm = ()
    # 检查 motif 只含合法碱基
    for m in _fm:
        if not all(ch in "ACGT" for ch in m):
            raise ValueError(f"forbidden motif '{m}' 含非法字符，仅允许 A/C/G/T")
        if len(m) < 2:
            raise ValueError(f"forbidden motif '{m}' 长度需 >= 2")

    max_motif_len = max((len(m) for m in _fm), default=0)
    max_hist = max(h - 1, gc_window - 1, max_motif_len - 1, 0)
    start = ""
    seen = {start}
    q = deque([start])
    transitions: Dict[str, Dict[str, str]] = {}

    def violates(suffix: str, b: str) -> bool:
        # --- 同聚物约束 ---
        if h > 1 and len(suffix) >= h - 1:
            if all(ch == b for ch in suffix[-(h - 1) :]):
                return True
        # --- GC 含量约束 ---
        if gc_window > 0:
            need = gc_window - 1
            if need <= 0:
                gc_count = 1 if b in GC_BASES else 0
                if gc_count < gc_min or gc_count > gc_max:
                    return True
            elif len(suffix) >= need:
                win = suffix[-need:] + b
                gc_count = sum(1 for ch in win if ch in GC_BASES)
                if gc_count < gc_min or gc_count > gc_max:
                    return True
        # --- 禁止 motif 约束 ---
        if _fm:
            candidate = suffix + b
            for motif in _fm:
                mlen = len(motif)
                # 只需检查以新碱基结尾的子串是否匹配
                if len(candidate) >= mlen and candidate[-mlen:] == motif:
                    return True
        return False

    while q:
        st = q.popleft()
        nxt_map: Dict[str, str] = {}
        for b in DNA_BASES:
            if violates(st, b):
                continue
            nst = (st + b)[-max_hist:] if max_hist > 0 else ""
            nxt_map[b] = nst
            if nst not in seen:
                seen.add(nst)
                q.append(nst)
        transitions[st] = nxt_map

    # 剪枝：移除没有出边或只能到达死状态的节点，避免后续编码卡死。
    alive = set(seen)
    while True:
        new_alive = set()
        for st in alive:
            nxt_map = transitions.get(st, {})
            filtered = {b: nst for b, nst in nxt_map.items() if nst in alive}
            transitions[st] = filtered
            if filtered:
                new_alive.add(st)
        if new_alive == alive:
            break
        alive = new_alive
    if start not in alive:
        raise ValueError("给定约束下起始状态不可持续，请放宽 h 或 GC 约束")

    for st in list(transitions.keys()):
        if st not in alive:
            transitions.pop(st, None)
    states = tuple(sorted(alive))
    return ConstraintDFA(
        states=states,
        start_state=start,
        transitions=transitions,
        max_hist=max_hist,
        homopolymer_h=h,
        gc_window=gc_window,
        gc_min=int(gc_min),
        gc_max=int(gc_max),
        forbidden_motifs=_fm,
    )


def _precompute_dfa_cache(
    dfa: ConstraintDFA,
) -> Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], int]]:
    """预计算每个 DFA 状态的 sorted choices/nexts/radix，消除热循环中的
    dict.get + sorted() + dict[b] 开销。"""
    cache: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], int]] = {}
    for st, nxt_map in dfa.transitions.items():
        if not nxt_map:
            continue
        choices = tuple(sorted(nxt_map.keys()))
        nexts = tuple(nxt_map[b] for b in choices)
        cache[st] = (choices, nexts, len(choices))
    return cache


def _find_parity_flush_sids(
    parity_current: int, alpha: int, parity_bits: int, max_depth: int = 8
) -> List[int]:
    """BFS 找最短 flush 符号序列使 parity 归零。
    对 8-bit parity + alpha>=4 的典型场景通常只需 1-2 步。"""
    target = 0
    if parity_current == target:
        return []
    q = deque([(parity_current, [])])
    visited = {parity_current}
    while q:
        par, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for sid in range(alpha):
            new_par = update_parity(par, sid, parity_bits)
            if new_par == target:
                return path + [sid]
            if new_par not in visited:
                visited.add(new_par)
                q.append((new_par, path + [sid]))
    return []  # 极端情况下无法归零，返回空列表


def _find_single_flush_sid(
    parity_current: int, alpha: int, parity_bits: int
) -> int:
    """对大字母表（alpha >= 2^parity_bits），保证找到恰好 1 个符号使 parity 归零。
    如果已经为零，返回 -1（不需要 flush）。
    如果字母表太小找不到单步解，返回 -2（调用者需使用多步 BFS）。"""
    if parity_current == 0:
        return -1
    mask = (1 << parity_bits) - 1
    # 目标: (parity_current * 5 + hash(sid)) & mask == 0
    target_hash = (-(parity_current * 5)) & mask
    for sid in range(alpha):
        h = _hash_from_symbol(sid) & mask
        if h == target_hash:
            return sid
    # 小字母表无法单步归零
    return -2


def _tail_flush_to_dna(
    ans_state: int,
    dfa_state: str,
    dfa_cache: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], int]],
    terminal_ans: int = PRESET_TERMINAL_ANS,
) -> Tuple[List[str], str]:
    """将 ANS 状态逐 digit 编码到 DNA 碱基中，直到状态降至 terminal_ans。

    返回 (tail_bases, final_dfa_state)。
    """
    tail: List[str] = []
    while ans_state > terminal_ans:
        choices, nexts, radix = dfa_cache[dfa_state]
        digit = ans_state % radix
        tail.append(choices[digit])
        dfa_state = nexts[digit]
        ans_state //= radix
    return tail, dfa_state


def _joint_state_trailer_to_dna(
    ans_state: int,
    parity: int,
    parity_bits: int,
    dfa_state: str,
    dfa_cache: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], int]],
    trailer_length: int,
) -> Tuple[List[str], str] | None:
    """Encode ``(ANS state, parity)`` into exactly ``trailer_length`` bases.

    The packed integer is expanded least-significant mixed-radix digit first.
    Once it reaches zero, the remaining high digits are deterministically zero.
    ``None`` means that the available suffix has insufficient capacity.
    """
    trailer_length = int(trailer_length)
    if trailer_length < 0:
        return None
    packed = (int(ans_state) << int(parity_bits)) | int(parity)
    st = dfa_state
    out: List[str] = []
    for _ in range(trailer_length):
        choices, nexts, radix = dfa_cache[st]
        digit = packed % radix
        out.append(choices[digit])
        st = nexts[digit]
        packed //= radix
    if packed != 0:
        return None
    return out, st


def _hash_from_symbol(u: int) -> int:
    """Full 32-bit integer hash (Murmur3 mix) to prevent symbol collisions."""
    x = int(u) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85ebca6b) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xc2b2ae35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def update_parity(p: int, u: int, r: int) -> int:
    """Multi-bit parity mixing: p' = (p * 5 + hash(u)) & mask.

    Unlike the old 1-bit shift-xor, this spreads symbol information across
    all *r* parity bits so that different symbols produce distinguishable
    parity values (and therefore different renormalization base sequences).
    """
    r = int(max(1, r))
    mask = (1 << r) - 1
    h = _hash_from_symbol(int(u)) & mask
    return ((int(p) * 5 + h) & mask)


def _emit_from_ans_state(
    dfa_cache: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], int]],
    dfa_state: str,
    ans_state: int,
    parity: int,
) -> Tuple[str, str, int]:
    """直接从 ANS 状态抽取一个"受约束碱基"（使用预计算 DFA 缓存）。"""
    entry = dfa_cache.get(dfa_state)
    if entry is None:
        raise ValueError(f"DFA state '{dfa_state}' 无可行转移，无法从 ANS 状态吐出碱基")
    choices, nexts, radix = entry
    idx = (int(ans_state) + int(parity)) % radix
    return choices[idx], nexts[idx], radix


def tans_encode_constrained_dna_direct(
    sequence: np.ndarray,
    dfa: ConstraintDFA,
    *,
    table_log: int = 12,
    parity_bits: int = 8,
    ans_memory_k: int = 16,
) -> Tuple[str, Dict]:
    """Backward-compatible single-segment encoder."""
    return tans_encode_constrained_dna_direct_with_stats(
        sequence,
        dfa,
        table_log=table_log,
        parity_bits=parity_bits,
        ans_memory_k=ans_memory_k,
    )


def tans_encode_constrained_dna_direct_with_stats(
    sequence: np.ndarray,
    dfa: ConstraintDFA,
    *,
    table_log: int = 12,
    parity_bits: int = 4,
    ans_memory_k: int = 16,
    global_table=None,
    include_model_in_meta: bool = True,
) -> Tuple[str, Dict]:
    """Single-segment encoder with detailed emission accounting.

    `payload_emissions` counts bases emitted while consuming source symbols.
    `parity_flush_emissions` counts extra bases emitted by parity flush symbols.
    `tail_flush_emissions` counts final ANS state drain emissions.

    `global_table`: 若给 (symbols, freqs, starts, table_log)（来自 build_global_table），
    则所有 oligo 共用此**全局**概率表，不再 per-oligo 从数据现算——概率表只作一次性
    side information，不随每条 oligo 携带。默认 None = 原局部自适应行为（每 oligo 自建表）。
    `include_model_in_meta=False` 时，oligo meta 不包含 symbols/freqs/starts；
    解码器必须从全局 side information 注入这些字段。
    """
    seq = np.asarray(sequence, dtype=np.int64).reshape(-1)
    if seq.size == 0:
        return "", {
            "emissions": 0,
            "payload_emissions": 0,
            "parity_flush_emissions": 0,
            "tail_flush_emissions": 0,
            "alphabet_size": 0,
            "n_flush_symbols": 0,
        }

    if global_table is not None:
        # 全局表：symbols/freqs/starts 由调用者一次性从整源直方图算好（build_global_table），
        # 所有 oligo 共用 → 概率表不再随每条 oligo 携带（side info 摊到整源可忽略）。
        g_symbols, g_freqs, g_starts, g_table_log = global_table
        symbols = np.asarray(g_symbols, dtype=np.int64)
        freqs = np.asarray(g_freqs, dtype=np.int64)
        starts = np.asarray(g_starts, dtype=np.int64)
        table_log = int(g_table_log)
        table_size = 1 << table_log
        alpha = int(symbols.size)
        inv = np.searchsorted(symbols, seq).astype(np.int64)   # 符号值 → 全局表索引
        if inv.size and (inv.max() >= alpha or not np.array_equal(symbols[inv], seq)):
            raise ValueError("global_table 未覆盖 seq 中的某些符号值")
    else:
        symbols, inv = np.unique(seq, return_inverse=True)
        alpha = int(symbols.size)

        table_log = int(max(4, min(table_log, 15)))
        min_log_by_alpha = int(math.ceil(math.log2(max(2, alpha))))
        table_log = int(max(table_log, min_log_by_alpha))
        table_size = 1 << table_log

        counts = np.bincount(inv, minlength=alpha).astype(np.int64)
        freqs = _normalize_counts(counts, table_size)
        cdf = np.cumsum(freqs, dtype=np.int64)
        starts = np.concatenate(([0], cdf[:-1]))
    # 注意：阈值不���预先按固定 max_radix=4 计算，
    # 而是在归一化循环中根据当前 DFA 状态的实际 radix 动态判断。
    # 这样避免了 radix=2 或 3 时的状���范围膨胀问题。

    ans_state = 1 << 16
    dfa_state = dfa.start_state
    dna_rev: List[str] = []
    parity = 0
    parity_bits = int(max(1, parity_bits))
    ans_memory_k = int(max(1, ans_memory_k))
    emissions = 0
    payload_emissions = 0
    parity_flush_emissions = 0
    tail_flush_emissions = 0

    dfa_cache = _precompute_dfa_cache(dfa)

    def _encode_one_symbol(sid: int, ans_st: int, dfa_st: str, par: int) -> Tuple[int, str, int, int]:
        f = int(freqs[sid])
        c = int(starts[sid])
        par = update_parity(par, sid, parity_bits)
        n_bases = 0
        # 动态阈值归一化：根据当前 DFA 状态的实际 radix 决定是否需要输出
        while True:
            _choices, _nexts, current_radix = dfa_cache[dfa_st]
            if ans_st < f * current_radix:
                break
            b, dfa_st, radix = _emit_from_ans_state(dfa_cache, dfa_st, ans_st, par)
            dna_rev.append(b)
            ans_st //= radix
            n_bases += 1
        ans_st = ((ans_st // f) << table_log) + (ans_st % f) + c
        sig_choices, sig_nexts, sig_radix = dfa_cache[dfa_st]
        sig_idx = (int(ans_st) + int(par)) % sig_radix
        dna_rev.append(sig_choices[sig_idx])
        dfa_st = sig_nexts[sig_idx]
        n_bases += 1
        return ans_st, dfa_st, par, n_bases

    for sid in inv[::-1]:
        ans_state, dfa_state, parity, nb = _encode_one_symbol(int(sid), ans_state, dfa_state, parity)
        emissions += nb
        payload_emissions += nb

    # ── Parity flush: 尝试单符号归零，小字母表回退到 BFS 多符号 ──
    flush_sid = _find_single_flush_sid(parity, alpha, parity_bits)
    if flush_sid >= 0:
        # 单符号归零成功
        ans_state, dfa_state, parity, nb = _encode_one_symbol(int(flush_sid), ans_state, dfa_state, parity)
        emissions += nb
        parity_flush_emissions += nb
        n_flush_symbols = 1
    elif flush_sid == -2:
        # 小字母表：需要多步 flush
        flush_sids = _find_parity_flush_sids(parity, alpha, parity_bits)
        n_flush_symbols = len(flush_sids)
        for sid in flush_sids:
            ans_state, dfa_state, parity, nb = _encode_one_symbol(int(sid), ans_state, dfa_state, parity)
            emissions += nb
            parity_flush_emissions += nb
    else:
        # parity 已经为 0
        n_flush_symbols = 0

    ans_state_before_tail = int(ans_state)

    # ── Tail flush: 将 ANS 状态编入 DNA ──
    tail_bases, dfa_state = _tail_flush_to_dna(ans_state, dfa_state, dfa_cache, PRESET_TERMINAL_ANS)
    dna_rev.extend(tail_bases)
    tail_flush_emissions = len(tail_bases)
    emissions += tail_flush_emissions

    dna = "".join(dna_rev)
    meta = {
        "emissions": int(emissions),
        "payload_emissions": int(payload_emissions),
        "parity_flush_emissions": int(parity_flush_emissions),
        "tail_flush_emissions": int(tail_flush_emissions),
        "alphabet_size": alpha,
        "table_log": int(table_log),
        "table_size": int(table_size),
        "sequence_length": int(seq.size),
        "n_flush_symbols": int(n_flush_symbols),
        "initial_ans_state": 1 << 16,
        "ans_state_before_tail": int(ans_state_before_tail),
        "final_ans_state": int(PRESET_TERMINAL_ANS),
        "final_parity": int(parity),
        "ans_memory_k": int(ans_memory_k),
        "parity_bits": int(parity_bits),
    }
    if include_model_in_meta:
        meta.update({
            "symbol_values": [int(v) for v in symbols.tolist()],
            "freqs": [int(v) for v in freqs.tolist()],
            "starts": [int(v) for v in starts.tolist()],
        })
    return dna, meta


def tans_encode_fixed_length_dna(
    sequence: np.ndarray,
    dfa: ConstraintDFA,
    *,
    target_dna_length: int = PRESET_DNA_LENGTH,
    table_log: int = PRESET_TABLE_LOG,
    parity_bits: int = PRESET_PARITY_BITS,
    ans_memory_k: int = PRESET_ANS_MEMORY_K,
    global_table=None,
    include_model_in_meta: bool = True,
    signature_mode: str = SIGNATURE_MODE_MIXED_V2,
    terminal_mode: str = TERMINAL_MODE_JOINT_STATE_V1,
) -> Tuple[str, Dict]:
    """固定长度 DNA 编码器：自适应符号数 + joint-state trailer。

    从输入序列末尾向前贪心编码，直到 DNA 预算用尽或输入耗尽。
    - 溢出：如果编码下一个符号将超出预算，回退并停止（编码少于 len(seq) 个符号）
    - 不足：joint-state trailer 编码当前 ANS/parity 状态并填满余下 DNA

    调用者应传入尽可能长的符号序列（比如 30-50 个），编码器自动决定
    实际编入多少个（可能 < 或 = len(seq)），不限制为固定 20 个。

    结构: [payload renormalization+signature] [joint-state trailer]

    零 per-oligo side information:
    - 解码器通过 joint trailer 验证自动检测实际符号数
    - 固定长度保证每条 oligo 合成/测序一致

    `global_table`: 若给 (symbols, freqs, starts, table_log)（来自 build_global_table），
    所有 oligo 共用该表，不再 per-oligo 现算——诚实码率、真正零 side info。

    返回 (dna, meta) 其中 dna 长度恒为 target_dna_length。
    meta['n_payload_symbols'] 记录实际编码的 payload 符号数。
    """
    if signature_mode not in VALID_SIGNATURE_MODES:
        raise ValueError(
            f"signature_mode must be one of {VALID_SIGNATURE_MODES}"
        )
    if terminal_mode != TERMINAL_MODE_JOINT_STATE_V1:
        raise ValueError("only terminal_mode='joint_state_v1' is supported")
    seq = np.asarray(sequence, dtype=np.int64).reshape(-1)
    if seq.size == 0:
        raise ValueError("Empty sequence for fixed-length encoding")

    # ── 构建频率表 ──
    if global_table is not None:
        g_symbols, g_freqs, g_starts, g_table_log = global_table
        symbols = np.asarray(g_symbols, dtype=np.int64)
        freqs = np.asarray(g_freqs, dtype=np.int64)
        starts = np.asarray(g_starts, dtype=np.int64)
        table_log = int(g_table_log)
        table_size = 1 << table_log
        alpha = int(symbols.size)
        inv = np.searchsorted(symbols, seq).astype(np.int64)   # 符号值 → 全局表索引
        if inv.size and (inv.max() >= alpha or not np.array_equal(symbols[inv], seq)):
            raise ValueError("global_table 未覆盖 seq 中的某些符号值")
    else:
        symbols, inv = np.unique(seq, return_inverse=True)
        alpha = int(symbols.size)
        table_log = int(max(4, min(table_log, 15)))
        min_log_by_alpha = int(math.ceil(math.log2(max(2, alpha))))
        table_log = int(max(table_log, min_log_by_alpha))
        table_size = 1 << table_log

        counts = np.bincount(inv, minlength=alpha).astype(np.int64)
        freqs = _normalize_counts(counts, table_size)
        cdf = np.cumsum(freqs, dtype=np.int64)
        starts = np.concatenate(([0], cdf[:-1]))

    ans_state = table_size  # initial ANS = table_size
    dfa_state = dfa.start_state
    dna_rev: List[str] = []
    parity = 0
    signature_history = int(SIGNATURE_HISTORY_INIT)
    signature_step = 0
    parity_bits = int(max(1, parity_bits))

    dfa_cache = _precompute_dfa_cache(dfa)

    def _encode_one_sym(
        sid: int, ans_st: int, dfa_st: str, par: int,
        sig_hist: int, sig_step: int,
    ) -> Tuple[int, str, int, int, int, int]:
        f = int(freqs[sid])
        c = int(starts[sid])
        par = update_parity(par, sid, parity_bits)
        n_bases = 0
        while True:
            _ch, _nx, current_radix = dfa_cache[dfa_st]
            if ans_st < f * current_radix:
                break
            b, dfa_st, radix = _emit_from_ans_state(dfa_cache, dfa_st, ans_st, par)
            dna_rev.append(b)
            ans_st //= radix
            n_bases += 1
        ans_st = ((ans_st // f) << table_log) + (ans_st % f) + c
        n_renorm = int(n_bases)
        n_signatures = signature_count_for_mode(signature_mode, n_renorm)
        for signature_ordinal in range(n_signatures):
            sig_choices, sig_nexts, sig_radix = dfa_cache[dfa_st]
            sig_idx, sig_hist = signature_index(
                ans_st, par, sid, sig_step, len(dna_rev), sig_hist,
                sig_radix, signature_mode,
                signature_ordinal=signature_ordinal,
                renorm_emissions=n_renorm,
            )
            dna_rev.append(sig_choices[sig_idx])
            dfa_st = sig_nexts[sig_idx]
            n_bases += 1
        return ans_st, dfa_st, par, n_bases, sig_hist, n_renorm

    # ── 逐符号编码（反向），同时监控 DNA 预算 ──
    # 策略：先编码所有符号，如果溢出则回退；如果不足则从序列继续取更多符号
    # 实际实现：贪心编码直到剩余空间不足容纳 flush + tail
    # No fixed reservation: after each candidate symbol, directly test whether
    # all remaining bases can carry the packed (ANS, parity) state.
    max_payload_budget = target_dna_length

    payload_symbols_encoded = 0
    payload_renorm_counts: List[int] = []
    payload_signature_counts: List[int] = []
    total_available = int(seq.size)

    # 反向编码：从序列末尾向前
    encode_order = inv[::-1].tolist()

    for idx, sid in enumerate(encode_order):
        # 保存状态快照以便回退
        snapshot = (
            int(ans_state), str(dfa_state), int(parity), len(dna_rev),
            int(signature_history), int(signature_step),
        )

        ans_state, dfa_state, parity, nb, signature_history, n_renorm = _encode_one_sym(
            int(sid), ans_state, dfa_state, parity,
            signature_history, signature_step,
        )
        signature_step += 1
        payload_symbols_encoded += 1
        payload_renorm_counts.append(int(n_renorm))
        payload_signature_counts.append(int(nb - n_renorm))

        # 检查是否已超出预算/剩余 mixed-radix 状态容量。
        terminal_fits = _joint_state_trailer_to_dna(
            ans_state, parity, parity_bits, dfa_state, dfa_cache,
            target_dna_length - len(dna_rev),
        ) is not None
        if len(dna_rev) > max_payload_budget or not terminal_fits:
            # 回退这一步
            ans_state, dfa_state, parity = snapshot[0], snapshot[1], snapshot[2]
            del dna_rev[snapshot[3]:]
            signature_history, signature_step = snapshot[4], snapshot[5]
            payload_symbols_encoded -= 1
            payload_renorm_counts.pop()
            payload_signature_counts.pop()
            break

    payload_only_emissions = len(dna_rev)

    trailer_emissions = 0

    trailer = _joint_state_trailer_to_dna(
        ans_state, parity, parity_bits, dfa_state, dfa_cache,
        target_dna_length - len(dna_rev),
    )
    if trailer is None:
        raise RuntimeError("joint state trailer unexpectedly does not fit")
    trailer_bases, dfa_state = trailer
    dna_rev.extend(trailer_bases)
    trailer_emissions = len(trailer_bases)

    # 如果超长（极端情况），截断不应发生但保险起见
    if len(dna_rev) > target_dna_length:
        dna_rev = dna_rev[:target_dna_length]

    dna = "".join(dna_rev)
    assert len(dna) == target_dna_length

    # 实际编码的原始序列符号数（解码器需自动检测这个值）
    n_original_symbols = total_available - (total_available - payload_symbols_encoded)
    # 对应的原始序列切片: seq[total_available - payload_symbols_encoded:]
    # 因为是反向编码，编码的是 seq 的后 payload_symbols_encoded 个符号

    meta = {
        "dna_length": int(target_dna_length),
        "n_payload_symbols": int(payload_symbols_encoded),
        "payload_emissions": int(payload_only_emissions),
        "payload_renorm_counts": payload_renorm_counts,
        "payload_signature_counts": payload_signature_counts,
        "signature_emissions": int(sum(payload_signature_counts)),
        "trailer_emissions": int(trailer_emissions),
        "alphabet_size": alpha,
        "table_log": int(table_log),
        "table_size": int(table_size),
        "initial_ans_state": int(table_size),
        "parity_bits": int(parity_bits),
        "ans_memory_k": int(ans_memory_k),
        "signature_mode": str(signature_mode),
        "signature_bases_per_symbol": (
            None if signature_mode == SIGNATURE_MODE_ADAPTIVE_V1
            else int(signature_count_for_mode(signature_mode, 0))
        ),
        "signature_history_init": int(SIGNATURE_HISTORY_INIT),
        "terminal_mode": str(terminal_mode),
    }
    # The decoder must infer/validate these states through the trailer; do not
    # expose their actual values as per-oligo side information.
    meta["terminal_state_side_information"] = False
    if include_model_in_meta:
        meta.update({
            "symbol_values": [int(v) for v in symbols.tolist()],
            "freqs": [int(v) for v in freqs.tolist()],
            "starts": [int(v) for v in starts.tolist()],
        })
    return dna, meta


def tans_decode_fixed_length_dna(
    dna_sequence: str,
    dfa: "ConstraintDFA",
    decode_info: Dict,
    *,
    beam_width: int = 512,
    k_tiers: int = 3,
    min_symbols: int = 1,
    max_symbols: int = 60,
    _int_dfa: "IntDFA | None" = None,
) -> np.ndarray:
    """固定长度 DNA 解码器：自动检测编码符号数。

    通过 Viterbi 跑 max_symbols + n_flush 步，在每步检查 tail 验证。
    第一个使 tail 精确归零且 parity==0 的终止点即为正确解。

    不需要任何 per-oligo side information。
    """
    from stage3_tans_decoder import tans_decode_constrained_dna_direct

    # 尝试不同符号数，从最可能的 20 开始向外扩展
    target = PRESET_SYMBOLS_PER_OLIGO
    search_order = [target]
    for d in range(1, max(target - min_symbols, max_symbols - target) + 1):
        if target - d >= min_symbols:
            search_order.append(target - d)
        if target + d <= max_symbols:
            search_order.append(target + d)

    best_result = None
    best_cost = float("inf")

    for n_payload in search_order:
        di = dict(decode_info)
        di["sequence_length"] = n_payload

        try:
            decoded = tans_decode_constrained_dna_direct(
                dna_sequence, dfa, di,
                beam_width=beam_width, k_tiers=k_tiers, _int_dfa=_int_dfa,
                strict_tail=True,
            )
            # strict_tail=True 确保只有 tail 完美验证（cost==0）时才返回
            best_result = decoded
            break  # 第一个成功的就是答案
        except ValueError:
            continue

    if best_result is None:
        raise ValueError("Fixed-length decode failed: no valid symbol count found")

    return best_result


def tans_encode_chunked_constrained_dna(
    sequence: np.ndarray,
    dfa: ConstraintDFA,
    *,
    symbols_per_chunk: int = 20,
    table_log: int = 12,
    parity_bits: int = 4,
    ans_memory_k: int = 16,
) -> List[Dict]:
    """Encode one long symbol sequence into ordered DNA chunks.

    A chunk takes a fixed number of source symbols (`symbols_per_chunk`).
    No padding is added to the final chunk if it is shorter than `symbols_per_chunk`.
    """
    seq = np.asarray(sequence, dtype=np.int64).reshape(-1)
    if seq.size == 0:
        return []

    symbols_per_chunk = int(max(1, symbols_per_chunk))
    chunks: List[Dict] = []
    start = 0
    segment_index = 0
    n_total = int(seq.size)
    
    while start < n_total:
        end = min(n_total, start + symbols_per_chunk)
        chunk_seq = seq[start:end]
        
        dna, meta = tans_encode_constrained_dna_direct_with_stats(
            chunk_seq,
            dfa,
            table_log=table_log,
            parity_bits=parity_bits,
            ans_memory_k=ans_memory_k,
        )
        
        chunks.append({
            "segment_index": int(segment_index),
            "symbol_start": int(start),
            "symbol_end": int(end),
            "symbol_length": int(end - start),
            "dna": dna,
            "meta": meta,
        })
        start = end
        segment_index += 1
        
    return chunks


def _load_interval_map(stage1_json: Dict) -> Dict[Tuple[int, int], Tuple[float, float]]:
    out: Dict[Tuple[int, int], Tuple[float, float]] = {}
    channels = stage1_json.get("channels", {})
    for ch_key, ch_obj in channels.items():
        ch = int(ch_key)
        q = ch_obj.get("quantization", {})
        intervals = q.get("interval_distribution", ch_obj.get("interval_distribution", []))
        for item in intervals:
            l = int(item["level"])
            a, b = item["interval"]
            out[(ch, l)] = (float(a), float(b))
    return out


def _add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    dest = name.lstrip("-").replace("-", "_")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default)
        return
    parser.add_argument(name, dest=dest, action="store_true")
    parser.add_argument(f"--no-{dest}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def run(
    stage1_json_path: str,
    stage2_json_path: str,
    stage2_npz_path: str,
    out_csv: str,
    *,
    table_log: int = 12,
    use_delta: bool = True,
    homopolymer_h: int = 4,
    gc_window: int = 8,
    gc_min: int | None = None,
    gc_max: int | None = None,
    parity_bits: int = 4,
    ans_memory_k: int = 16,
    symbols_per_chunk: int = 20,
    decode_info_json: str | None = None,
) -> str:
    with open(stage1_json_path, "r", encoding="utf-8") as f:
        s1 = json.load(f)
    with open(stage2_json_path, "r", encoding="utf-8") as f:
        s2 = json.load(f)
    seq_data = np.load(stage2_npz_path, allow_pickle=True)

    interval_map = _load_interval_map(s1)
    channels_info = s2["channels_info"]
    dfa = build_constraint_dfa(h=homopolymer_h, gc_window=gc_window, gc_min=gc_min, gc_max=gc_max)
    decode_info_dump: Dict[str, Dict] = {
        "global": {
            "homopolymer_h": int(dfa.homopolymer_h),
            "gc_window": int(dfa.gc_window),
            "gc_min": int(dfa.gc_min),
            "gc_max": int(dfa.gc_max),
            "start_state": dfa.start_state,
            "parity_bits": int(parity_bits),
            "ans_memory_k": int(ans_memory_k),
            "decoder_must_know": [
                "DFA 约束与起始状态",
                "table_log/table_size",
                "symbol_values（内部 sid 到原始 index 的映射）",
                "freqs/starts（量化频率表）",
                "sequence_length",
                "initial_ans_state",
                "parity_bits",
                "ans_memory_k（有限记忆 trellis）",
            ],
        },
        "rows": {},
    }

    ensure_dir(os.path.dirname(os.path.abspath(out_csv)))
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "channel",
                "level",
                "segment_index",
                "segment_count",
                "symbol_start",
                "symbol_end",
                "segment_symbol_length",
                "quantization_interval",
                "tans_constrained_dna",
                "payload_dna_length",
                "dna_length",
                "ans_emissions",
            ],
        )
        writer.writeheader()

        for ch_key, ch_obj in channels_info.items():
            ch = int(ch_key)
            for item in ch_obj["per_level_sequences"]:
                l = int(item["level"])
                seq_key = item["delta_encoded"]["npz_key"] if use_delta else item["coords_1d"]["npz_key"]
                seq = np.asarray(seq_data[seq_key], dtype=np.int64).reshape(-1)
                chunks = tans_encode_chunked_constrained_dna(
                    seq,
                    dfa,
                    symbols_per_chunk=symbols_per_chunk,
                    table_log=table_log,
                    parity_bits=parity_bits,
                    ans_memory_k=ans_memory_k,
                )
                interval = interval_map.get((ch, l), (float("nan"), float("nan")))
                segment_count = len(chunks)
                for chunk in chunks:
                    dna_seq = chunk["dna"]
                    dna_meta = chunk["meta"]
                    seg_idx = int(chunk["segment_index"])
                    row_key = f"ch{ch}_lvl{l}_seg{seg_idx}"
                    decode_info_dump["rows"][row_key] = {
                        "channel": int(ch),
                        "level": int(l),
                        "segment_index": int(seg_idx),
                        "segment_count": int(segment_count),
                        "symbol_start": int(chunk["symbol_start"]),
                        "symbol_end": int(chunk["symbol_end"]),
                        "decode_info": dna_meta,
                    }
                    writer.writerow(
                        {
                            "channel": ch,
                            "level": l,
                            "segment_index": int(seg_idx),
                            "segment_count": int(segment_count),
                            "symbol_start": int(chunk["symbol_start"]),
                            "symbol_end": int(chunk["symbol_end"]),
                            "segment_symbol_length": int(chunk["symbol_length"]),
                            "quantization_interval": f"[{interval[0]},{interval[1]}]",
                            "tans_constrained_dna": dna_seq,
                            "payload_dna_length": int(dna_meta.get("payload_emissions", len(dna_seq))),
                            "dna_length": len(dna_seq),
                            "ans_emissions": int(dna_meta["emissions"]),
                        }
                    )
    info_path = decode_info_json
    if not info_path:
        root, _ext = os.path.splitext(os.path.abspath(out_csv))
        info_path = root + ".decode_info.json"
    ensure_dir(os.path.dirname(os.path.abspath(info_path)))
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(decode_info_dump, f, ensure_ascii=False, indent=2)
    return os.path.abspath(out_csv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_json", type=str, default=os.path.join("out", "stage1", "stage1_quantization.json"))
    ap.add_argument("--stage2_json", type=str, default=os.path.join("out", "stage2", "stage2_encoding.json"))
    ap.add_argument("--stage2_npz", type=str, default=os.path.join("out", "stage2", "stage2_sequences.npz"))
    ap.add_argument("--out_csv", type=str, default=os.path.join("out", "stage3", "stage3_tans.csv"))
    ap.add_argument("--table_log", type=int, default=12)
    _add_bool_argument(ap, "--use_delta", default=True)
    ap.add_argument("--homopolymer_h", type=int, default=4)
    ap.add_argument("--gc_window", type=int, default=8)
    ap.add_argument("--gc_min", type=int, default=None)
    ap.add_argument("--gc_max", type=int, default=None)
    ap.add_argument("--parity_bits", type=int, default=4)
    ap.add_argument("--ans_memory_k", type=int, default=16)
    ap.add_argument("--symbols_per_chunk", type=int, default=20)
    ap.add_argument("--decode_info_json", type=str, default=None)
    args = ap.parse_args()

    out = run(
        stage1_json_path=args.stage1_json,
        stage2_json_path=args.stage2_json,
        stage2_npz_path=args.stage2_npz,
        out_csv=args.out_csv,
        table_log=args.table_log,
        use_delta=args.use_delta,
        homopolymer_h=args.homopolymer_h,
        gc_window=args.gc_window,
        gc_min=args.gc_min,
        gc_max=args.gc_max,
        parity_bits=args.parity_bits,
        ans_memory_k=args.ans_memory_k,
        symbols_per_chunk=args.symbols_per_chunk,
        decode_info_json=args.decode_info_json,
    )
    print(out)


if __name__ == "__main__":
    main()
