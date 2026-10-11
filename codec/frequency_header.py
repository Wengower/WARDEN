"""方案1：把全局 tANS 频率表编成可自解的 header oligo。

动机
----
`build_global_table` 得到的概率表是**数据依赖**的 side information：它取决于
被存数据自身的字节直方图，解码器无从凭空知道。若不把它写进 DNA，DNA 池就
**不能自解**（此前的实验是从原图 `build_global_table` 重算表递给解码器，属于
循环依赖）。本模块把这张表序列化后，用一个**零 side information 的固定引导模型**
（均匀字节分布 `build_preset_byte_table`）经同一条定长编码器编成若干 header
oligo。解码时只需该固定模型即可先解出 header、还原全局表，再去解 payload——
不再依赖原图。

约定
----
- 紧凑格式要求 `symbols == arange(alphabet_size)`（`build_global_table` 正是如此）。
- 只存 `freqs`（每个占 `table_log` bit）；`starts` 为前缀和、`symbols` 为
  `arange`、`table_log` 存在 4 字节前缀里，均可由 `freqs` 重建。
- header 自身用**均匀模型**，是编解码器共知的算法规范常量，不占 side information。

header 净成本 = 4 + ceil(alphabet_size * table_log / 8) 字节。
alphabet_size=256, table_log=12 → 4 + 384 = 388 字节（编成 ~十几条定长 oligo）。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from core import (
    SIGNATURE_MODE_MIXED_V2,
    TERMINAL_MODE_JOINT_STATE_V1,
    _normalize_counts,
    tans_encode_fixed_length_dna,
)
from decoder import decode_fixed_length_macro

HEADER_VERSION = 1


# ── 零-SI 引导模型：字节字母表上的均匀分布 ────────────────────────────
def build_preset_byte_table(
    table_log: int = 12, alphabet_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """均匀分布 [0, alphabet_size)。固定编解码常量 → 零 side information：
    编码器与解码器各自重建同一张表，因此 header 本身无需再存任何模型即可自解。
    """
    min_log = int(math.ceil(math.log2(max(2, alphabet_size))))
    table_log = int(max(min_log, min(int(table_log), 15)))
    table_size = 1 << table_log
    counts = np.ones(int(alphabet_size), dtype=np.int64)
    freqs = _normalize_counts(counts, table_size)
    starts = np.concatenate(([0], np.cumsum(freqs[:-1], dtype=np.int64)))
    symbols = np.arange(int(alphabet_size), dtype=np.int64)
    return symbols, freqs.astype(np.int64), starts.astype(np.int64), int(table_log)


# ── 定长 index 辅助（与 encode_us 的大端定宽约定一致，本地副本避免跨包导入）──
def required_index_symbols(stream_length: int) -> int:
    maximum = max(0, int(stream_length) - 1)
    return max(1, (maximum.bit_length() + 7) // 8)


def linear_index_symbols(position: int, symbol_count: int) -> np.ndarray:
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


def decode_linear_index(symbols) -> int:
    value = 0
    for symbol in symbols:
        value = (value << 8) | (int(symbol) & 0xFF)
    return value


# ── 频率表 ↔ 字节序列 ────────────────────────────────────────────────
def serialize_freq_table(global_table) -> np.ndarray:
    """(symbols, freqs, starts, table_log) → uint8 序列（紧凑 12-bit 打包）。"""
    symbols, freqs, _starts, table_log = global_table
    freqs = np.asarray(freqs, dtype=np.int64)
    symbols = np.asarray(symbols, dtype=np.int64)
    alphabet_size = int(freqs.size)
    bits = int(table_log)
    if not np.array_equal(symbols, np.arange(alphabet_size)):
        raise ValueError("紧凑 header 要求 symbols == arange(alphabet_size)")
    if int(freqs.min()) < 1 or int(freqs.max()) >= (1 << bits):
        raise ValueError("freq 必须落在 [1, 2^table_log)")
    if not 1 <= bits <= 16 or not 1 <= alphabet_size <= 0xFFFF:
        raise ValueError("table_log / alphabet_size 超出 header 前缀可表示范围")

    out = bytearray()
    out.append(HEADER_VERSION & 0xFF)
    out.append(bits & 0xFF)
    out.append((alphabet_size >> 8) & 0xFF)
    out.append(alphabet_size & 0xFF)

    acc = 0
    nbits = 0
    for freq in freqs.tolist():
        acc = (acc << bits) | int(freq)
        nbits += bits
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
    if nbits > 0:
        out.append((acc << (8 - nbits)) & 0xFF)
    return np.frombuffer(bytes(out), dtype=np.uint8)


def deserialize_freq_table(
    byte_array,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """serialize_freq_table 的逆过程 → (symbols, freqs, starts, table_log)。"""
    data = np.asarray(byte_array, dtype=np.uint8).reshape(-1).tolist()
    if len(data) < 4:
        raise ValueError("header 字节流过短")
    version = data[0]
    if version != HEADER_VERSION:
        raise ValueError(f"未知 header 版本 {version}")
    bits = data[1]
    alphabet_size = (data[2] << 8) | data[3]
    mask = (1 << bits) - 1

    acc = 0
    nbits = 0
    freqs: List[int] = []
    for byte in data[4:]:
        acc = (acc << 8) | int(byte)
        nbits += 8
        while nbits >= bits and len(freqs) < alphabet_size:
            nbits -= bits
            freqs.append((acc >> nbits) & mask)
        if len(freqs) >= alphabet_size:
            break
    if len(freqs) != alphabet_size:
        raise ValueError("header 字节流不足以还原完整频率表")

    freqs_arr = np.asarray(freqs, dtype=np.int64)
    symbols = np.arange(alphabet_size, dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(freqs_arr[:-1], dtype=np.int64)))
    return symbols, freqs_arr, starts, int(bits)


# ── 把任意字节流编成一组定长 oligo（带线性 index）────────────────────
def _encode_stream_fixed(
    symbol_stream: np.ndarray,
    model_table,
    dfa,
    *,
    dna_length: int,
    parity_bits: int,
    signature_mode: str,
    candidate_symbols: int = 96,
) -> Tuple[List[Dict], int]:
    symbol_stream = np.asarray(symbol_stream, dtype=np.int64).reshape(-1)
    index_count = required_index_symbols(symbol_stream.size)
    records: List[Dict] = []
    cursor = 0
    while cursor < symbol_stream.size:
        start = int(cursor)
        max_count = min(int(candidate_symbols), int(symbol_stream.size) - start)
        index_syms = linear_index_symbols(start, index_count)

        def encode_count(count: int):
            inner = np.concatenate(
                (symbol_stream[start:start + int(count)], index_syms[::-1])
            )
            dna, meta = tans_encode_fixed_length_dna(
                inner, dfa, target_dna_length=dna_length,
                parity_bits=parity_bits, global_table=model_table,
                include_model_in_meta=False, signature_mode=signature_mode,
                terminal_mode=TERMINAL_MODE_JOINT_STATE_V1,
            )
            fits = int(meta["n_payload_symbols"]) == int(inner.size)
            return dna, meta, fits

        _probe_dna, probe_meta, _probe_fits = encode_count(max_count)
        count = min(
            max_count,
            max(1, int(probe_meta["n_payload_symbols"]) - index_count),
        )
        dna, meta, fits = encode_count(count)
        while not fits and count > 1:
            count -= 1
            dna, meta, fits = encode_count(count)
        if not fits:
            raise RuntimeError(f"header 编码在 {start} 处无法推进")
        while count < max_count:
            next_dna, next_meta, next_fits = encode_count(count + 1)
            if not next_fits:
                break
            count += 1
            dna, meta = next_dna, next_meta

        records.append({"start": start, "count": int(count), "dna": dna})
        cursor = start + int(count)
    return records, index_count


def encode_table_header(
    global_table,
    dfa,
    *,
    dna_length: int = 130,
    parity_bits: int = 4,
    signature_mode: str = SIGNATURE_MODE_MIXED_V2,
    boot_table_log: int = 12,
) -> Tuple[List[str], Dict]:
    """把全局频率表编成一组自解 header oligo。

    返回 (header_dna_list, info)。header 用均匀引导模型（零 SI）编码，因此可以
    脱离原图独立解码。info 记录字节数、oligo 条数、header 占用的 nt 等，供如实
    计入总码率。
    """
    byte_stream = serialize_freq_table(global_table).astype(np.int64)
    model = build_preset_byte_table(table_log=boot_table_log, alphabet_size=256)
    records, index_count = _encode_stream_fixed(
        byte_stream, model, dfa,
        dna_length=dna_length, parity_bits=parity_bits,
        signature_mode=signature_mode,
    )
    header_dna = [record["dna"] for record in records]
    info = {
        "header_bytes": int(byte_stream.size),
        "n_header_oligos": len(records),
        "header_nt": len(records) * int(dna_length),
        "header_stream_length": int(byte_stream.size),
        "index_symbol_count": int(index_count),
        "boot_model": "uniform_byte_zero_si",
        "dna_length": int(dna_length),
        "parity_bits": int(parity_bits),
        "signature_mode": str(signature_mode),
    }
    return header_dna, info


def decode_table_header(
    header_dna: List[str],
    dfa,
    *,
    header_stream_length: int,
    dna_length: int = 130,
    parity_bits: int = 4,
    signature_mode: str = SIGNATURE_MODE_MIXED_V2,
    boot_table_log: int = 12,
    beam_width: int = 256,
    max_payload_symbols: int = 160,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """仅凭固定均匀模型（不看原图）解出 header，还原全局频率表。"""
    model = build_preset_byte_table(table_log=boot_table_log, alphabet_size=256)
    index_count = required_index_symbols(header_stream_length)
    recovered = np.full(int(header_stream_length), -1, dtype=np.int64)

    for dna in header_dna:
        decode_info = {
            "dna_length": int(dna_length),
            "parity_bits": int(parity_bits),
            "initial_ans_state": 1 << int(model[3]),
            "signature_mode": str(signature_mode),
            "terminal_mode": TERMINAL_MODE_JOINT_STATE_V1,
            "min_payload_symbols": int(index_count),
            "max_payload_symbols": int(max_payload_symbols),
            "linear_index_symbols": int(index_count),
            "source_stream_length": int(header_stream_length),
        }
        decoded, _details = decode_fixed_length_macro(
            dna, dfa, decode_info, global_table=model,
            beam_width=beam_width, return_details=True,
        )
        decoded = np.asarray(decoded, dtype=np.int64)
        if decoded.size < index_count:
            raise ValueError("header oligo 解码结果短于其 index")
        start = decode_linear_index(decoded[-index_count:][::-1])
        payload = decoded[:-index_count]
        if not 0 <= start <= header_stream_length - payload.size:
            raise ValueError(f"header index={start} 越界")
        recovered[start:start + payload.size] = payload

    if int((recovered < 0).sum()) != 0:
        raise ValueError("header 未被完整覆盖，无法还原频率表")
    return deserialize_freq_table(recovered.astype(np.uint8))
