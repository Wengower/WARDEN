from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import numpy as np

# ── Optional Numba JIT accelerator ────────────────────────────────────────
# Install with:  pip install numba --break-system-packages
# First call compiles and caches to disk; subsequent calls use the cache.
try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    def _njit(*_a, **_kw):          # transparent no-op decorator
        def _wrap(fn): return fn
        return _wrap
    _HAS_NUMBA = False

# Frontier / output buffer sizes for the Numba slow-path renorm loop.
# 64 frontier states and 16 outputs per symbol are generous upper bounds.
_NB_F_MAX = 64
_NB_O_MAX = 16


# ── public helpers (used by other modules too) ───────────────────────────
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
    """Multi-bit parity mixing: p' = (p * 5 + hash(u)) & mask."""
    r = int(max(1, r))
    mask = (1 << r) - 1
    h = _hash_from_symbol(int(u)) & mask
    return ((int(p) * 5 + h) & mask)


# ── cost constants ───────────────────────────────────────────────────────
COST_MATCH = 0
COST_SUBSTITUTION = 3
COST_INSERTION = 4
COST_DELETION = 4
COST_TAIL_MISMATCH = 3
COST_TAIL_INSERTION = 4
COST_BEYOND_DNA = 10
COST_PARITY_MISMATCH = 1000000

MAX_INSERTIONS_PER_STEP = 1

# ── tier-beam pruning ────────────────────────────────────────────────────
DEFAULT_K_TIERS = 3
DEFAULT_MAX_BEAM = 512


def _tier_beam_dict(
    candidates: dict,
    k_tiers: int = DEFAULT_K_TIERS,
    max_total: int = DEFAULT_MAX_BEAM,
    cost_idx: int = 0,
) -> dict:
    if len(candidates) <= k_tiers:
        return candidates
    items = sorted(candidates.items(), key=lambda kv: kv[1][cost_idx])
    kept: dict = {}
    tiers_seen = 0
    prev_cost = None
    for key, val in items:
        cost = val[cost_idx]
        if cost != prev_cost:
            tiers_seen += 1
            prev_cost = cost
        if tiers_seen > k_tiers:
            break
        if len(kept) >= max_total:
            break
        kept[key] = val
    return kept


# ── DFA integer-isation ──────────────────────────────────────────────────
_BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}
_ID_TO_BASE = ("A", "C", "G", "T")


class IntDFA:
    """Integer-indexed DFA cache — build once, reuse across sequences."""
    __slots__ = (
        "state_to_id", "id_to_state", "n_states",
        "allowed_bases", "allowed_nexts", "radix",
        "obs_next",   # obs_next[st][base_id] -> next_st or -1
        "start_id",
    )

    def __init__(self, dfa):
        all_states = sorted(dfa.transitions.keys())
        self.state_to_id: Dict[str, int] = {s: i for i, s in enumerate(all_states)}
        self.id_to_state = all_states
        self.n_states = len(all_states)

        if dfa.start_state not in self.state_to_id:
            self.state_to_id[dfa.start_state] = self.n_states
            self.id_to_state.append(dfa.start_state)
            self.n_states += 1
        self.start_id = self.state_to_id[dfa.start_state]

        ns = self.n_states
        # allowed_bases[st] and allowed_nexts[st] are tuples of ints
        self.allowed_bases: List[Tuple[int, ...]] = [() for _ in range(ns)]
        self.allowed_nexts: List[Tuple[int, ...]] = [() for _ in range(ns)]
        self.radix: List[int] = [0] * ns
        # obs_next: flat lookup, obs_next[st * 4 + base_id] = next_st or -1
        self.obs_next = np.full(ns * 4, -1, dtype=np.int32)

        s2i = self.state_to_id
        for st_str, nxt_map in dfa.transitions.items():
            sid = s2i[st_str]
            if not nxt_map:
                continue
            bases_sorted = sorted(nxt_map.keys())
            base_ids = tuple(_BASE_TO_ID[b] for b in bases_sorted)
            next_ids = tuple(s2i[nxt_map[b]] for b in bases_sorted)
            self.allowed_bases[sid] = base_ids
            self.allowed_nexts[sid] = next_ids
            self.radix[sid] = len(base_ids)
            base4 = sid * 4
            for b_str in bases_sorted:
                self.obs_next[base4 + _BASE_TO_ID[b_str]] = s2i[nxt_map[b_str]]


def _build_parity_lut(alpha: int, parity_bits: int) -> List[List[int]]:
    """Precompute par_lut[old_par][sid] -> new_par for all combinations.

    改法四说明
    ----------
    parity_bits=4 时奇偶判别力只有 1/16 ≈ 6.25%，beam=512 时约有 32 个
    假阳性路径通过奇偶校验，导致约 16% 的 mismatch。
    强烈建议编码端将 parity_bits 从 4 升到 8（1/256 ≈ 0.4%），与编码器
    保持一致后重新生成 decode_info 即可，此函数无需其他修改。
    """
    n_par = 1 << parity_bits
    lut: List[List[int]] = []
    for p in range(n_par):
        row = [0] * alpha
        for sid in range(alpha):
            row[sid] = update_parity(p, sid, parity_bits)
        lut.append(row)
    return lut


# ── Numba helper: flat DFA arrays ─────────────────────────────────────────

def _prepare_dfa_arrays(D: IntDFA) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, int]:
    """Convert IntDFA to flat numpy arrays suitable for Numba JIT.

    Returns (dfa_r_arr, dfa_ab_flat, dfa_an_flat, max_radix).
    dfa_ab_flat[st * max_radix + k] = k-th allowed base id at state st (-1 if none).
    dfa_an_flat[st * max_radix + k] = corresponding next state id (-1 if none).
    """
    ns = D.n_states
    max_radix = int(max(D.radix)) if D.radix else 1
    max_radix = max(max_radix, 1)

    dfa_r_arr   = np.array(D.radix,  dtype=np.int32)
    dfa_ab_flat = np.full(ns * max_radix, -1, dtype=np.int32)
    dfa_an_flat = np.full(ns * max_radix, -1, dtype=np.int32)

    for st in range(ns):
        ab = D.allowed_bases[st]
        an = D.allowed_nexts[st]
        for k in range(len(ab)):
            dfa_ab_flat[st * max_radix + k] = ab[k]
            dfa_an_flat[st * max_radix + k] = an[k]

    return dfa_r_arr, dfa_ab_flat, dfa_an_flat, max_radix


# ── Numba JIT: single-symbol step (改法三) ────────────────────────────────

@_njit(cache=True)
def _step_symbol_nb(
    ans0: np.int64, st0: np.int32, pos0: np.int32,
    f: np.int64, c: np.int64, table_log: np.int32,
    par_eager: np.int32, parity_mask: np.int32,
    dfa_r:    np.ndarray,   # int32[n_states]
    dfa_ab:   np.ndarray,   # int32[n_states * max_radix]
    dfa_an:   np.ndarray,   # int32[n_states * max_radix]
    dfa_obs:  np.ndarray,   # int32[n_states * 4]
    max_radix: np.int32,
    dna_ids:  np.ndarray,   # int32[dna_len]
    dna_len:  np.int32,
    C_SUB: np.int32, C_DEL: np.int32, C_BEYOND: np.int32,
    threshold: np.int32,
    out_ans:  np.ndarray,   # int64[_NB_O_MAX]  — output buffer
    out_st:   np.ndarray,   # int32[_NB_O_MAX]
    out_par:  np.ndarray,   # int32[_NB_O_MAX]
    out_pos:  np.ndarray,   # int32[_NB_O_MAX]
    out_cost: np.ndarray,   # int32[_NB_O_MAX]
) -> np.int32:
    """Numba-compiled single-symbol step.

    Returns n_out (number of candidates written to out_* arrays).
    par_eager must be pre-computed by the Python caller via par_lut[par0][sid].
    Handles both the fast (no-mismatch) and slow (branching frontier) paths.
    """
    O_MAX = np.int32(len(out_ans))
    F_MAX = np.int32(64)

    # ─── allocate frontier arrays on Numba stack ──────────────────────────
    f_a = np.empty(64, dtype=np.int64)
    f_s = np.empty(64, dtype=np.int32)
    f_p = np.empty(64, dtype=np.int32)
    f_c = np.empty(64, dtype=np.int32)

    # ─── FAST PATH ────────────────────────────────────────────────────────
    t_ans = ans0
    t_st  = st0
    t_pos = pos0
    cost  = np.int32(0)
    slow  = False

    while True:
        r = dfa_r[t_st]
        if r == np.int32(0):
            return np.int32(0)
        if t_ans < f * r:
            break
        if cost > threshold:
            return np.int32(0)
        idx   = int((t_ans + par_eager) % r)
        bid   = dfa_ab[t_st * max_radix + idx]
        nst   = dfa_an[t_st * max_radix + idx]
        if t_pos >= dna_len:
            t_ans //= r; t_st = nst; t_pos += np.int32(1); cost += C_BEYOND
        elif dna_ids[t_pos] == bid:
            t_ans //= r; t_st = nst; t_pos += np.int32(1)
        else:
            slow = True
            break

    if not slow:
        # absorption
        ans1    = ((t_ans // f) << table_log) + (t_ans % f) + c
        sig_r   = dfa_r[t_st]
        if sig_r == np.int32(0):
            return np.int32(0)
        sig_idx = int((ans1 + par_eager) % sig_r)
        sig_bid = dfa_ab[t_st * max_radix + sig_idx]
        sig_nst = dfa_an[t_st * max_radix + sig_idx]
        n_out   = np.int32(0)
        if t_pos >= dna_len:
            out_ans[0]=ans1; out_st[0]=sig_nst; out_par[0]=par_eager
            out_pos[0]=t_pos+np.int32(1); out_cost[0]=cost+C_BEYOND
            return np.int32(1)
        elif dna_ids[t_pos] == sig_bid:
            out_ans[0]=ans1; out_st[0]=sig_nst; out_par[0]=par_eager
            out_pos[0]=t_pos+np.int32(1); out_cost[0]=cost
            return np.int32(1)
        else:
            out_ans[0]=ans1; out_st[0]=sig_nst; out_par[0]=par_eager
            out_pos[0]=t_pos+np.int32(1); out_cost[0]=cost+C_SUB
            n_out = np.int32(1)
            osn = dfa_obs[t_st * np.int32(4) + dna_ids[t_pos]]
            if osn >= np.int32(0) and n_out < O_MAX:
                out_ans[n_out]=ans1; out_st[n_out]=osn; out_par[n_out]=par_eager
                out_pos[n_out]=t_pos+np.int32(1); out_cost[n_out]=cost+C_SUB+np.int32(1)
                n_out += np.int32(1)
            if n_out < O_MAX:
                out_ans[n_out]=ans1; out_st[n_out]=sig_nst; out_par[n_out]=par_eager
                out_pos[n_out]=t_pos; out_cost[n_out]=cost+C_DEL
                n_out += np.int32(1)
            return n_out

    # ─── SLOW PATH: branching frontier ────────────────────────────────────
    # Initialise frontier from the mismatch point.
    r       = dfa_r[t_st]
    idx     = int((t_ans + par_eager) % r)
    nst_exp = dfa_an[t_st * max_radix + idx]
    ans_nxt = t_ans // r
    obs     = dna_ids[t_pos]
    flen    = np.int32(0)

    # substitution
    if cost + C_SUB <= threshold:
        f_a[flen]=ans_nxt; f_s[flen]=nst_exp
        f_p[flen]=t_pos+np.int32(1); f_c[flen]=cost+C_SUB
        flen += np.int32(1)
    # substitution + DFA sync
    osn0 = dfa_obs[t_st * np.int32(4) + obs]
    if osn0 >= np.int32(0) and cost+C_SUB+np.int32(1) <= threshold and flen < F_MAX:
        f_a[flen]=ans_nxt; f_s[flen]=osn0
        f_p[flen]=t_pos+np.int32(1); f_c[flen]=cost+C_SUB+np.int32(1)
        flen += np.int32(1)
    # deletion (check duplicate with sub before inserting)
    c_del = cost + C_DEL
    if c_del <= threshold:
        found = False
        for k in range(flen):
            if f_a[k] == ans_nxt and f_s[k] == nst_exp and f_p[k] == t_pos:
                if c_del < f_c[k]:
                    f_c[k] = c_del
                found = True
                break
        if not found and flen < F_MAX:
            f_a[flen]=ans_nxt; f_s[flen]=nst_exp
            f_p[flen]=t_pos; f_c[flen]=c_del
            flen += np.int32(1)

    # Continue renorm over frontier
    nf_a = np.empty(64, dtype=np.int64)
    nf_s = np.empty(64, dtype=np.int32)
    nf_p = np.empty(64, dtype=np.int32)
    nf_c = np.empty(64, dtype=np.int32)

    while flen > np.int32(0):
        all_done = True
        for fi in range(flen):
            if f_a[fi] >= f * dfa_r[f_s[fi]]:
                all_done = False
                break
        if all_done:
            break

        nflen = np.int32(0)
        for fi in range(flen):
            a2 = f_a[fi]; s2 = f_s[fi]; p2 = f_p[fi]; c2 = f_c[fi]
            if c2 > threshold:
                continue
            r2 = dfa_r[s2]
            if a2 < f * r2:
                # already done — carry forward (dedup by min cost)
                found = False
                for k in range(nflen):
                    if nf_a[k]==a2 and nf_s[k]==s2 and nf_p[k]==p2:
                        if c2 < nf_c[k]: nf_c[k] = c2
                        found = True; break
                if not found and nflen < F_MAX:
                    nf_a[nflen]=a2; nf_s[nflen]=s2; nf_p[nflen]=p2; nf_c[nflen]=c2
                    nflen += np.int32(1)
                continue
            idx2   = int((a2 + par_eager) % r2)
            sn2    = dfa_an[s2 * max_radix + idx2]
            an2    = a2 // r2
            if p2 >= dna_len:
                cn = c2 + C_BEYOND
                if cn <= threshold and nflen < F_MAX:
                    nf_a[nflen]=an2; nf_s[nflen]=sn2; nf_p[nflen]=p2+np.int32(1); nf_c[nflen]=cn
                    nflen += np.int32(1)
            elif dna_ids[p2] == dfa_ab[s2 * max_radix + idx2]:
                found = False
                for k in range(nflen):
                    if nf_a[k]==an2 and nf_s[k]==sn2 and nf_p[k]==p2+np.int32(1):
                        if c2 < nf_c[k]: nf_c[k] = c2
                        found = True; break
                if not found and nflen < F_MAX:
                    nf_a[nflen]=an2; nf_s[nflen]=sn2; nf_p[nflen]=p2+np.int32(1); nf_c[nflen]=c2
                    nflen += np.int32(1)
            else:
                ob2 = dna_ids[p2]
                cs  = c2 + C_SUB
                if cs <= threshold and nflen < F_MAX:
                    nf_a[nflen]=an2; nf_s[nflen]=sn2; nf_p[nflen]=p2+np.int32(1); nf_c[nflen]=cs
                    nflen += np.int32(1)
                on2 = dfa_obs[s2 * np.int32(4) + ob2]
                if on2 >= np.int32(0) and c2+C_SUB+np.int32(1) <= threshold and nflen < F_MAX:
                    nf_a[nflen]=an2; nf_s[nflen]=on2; nf_p[nflen]=p2+np.int32(1)
                    nf_c[nflen]=c2+C_SUB+np.int32(1); nflen += np.int32(1)
                cd = c2 + C_DEL
                if cd <= threshold:
                    found = False
                    for k in range(nflen):
                        if nf_a[k]==an2 and nf_s[k]==sn2 and nf_p[k]==p2:
                            if cd < nf_c[k]: nf_c[k] = cd
                            found = True; break
                    if not found and nflen < F_MAX:
                        nf_a[nflen]=an2; nf_s[nflen]=sn2; nf_p[nflen]=p2; nf_c[nflen]=cd
                        nflen += np.int32(1)

        flen = nflen
        f_a[:nflen] = nf_a[:nflen]
        f_s[:nflen] = nf_s[:nflen]
        f_p[:nflen] = nf_p[:nflen]
        f_c[:nflen] = nf_c[:nflen]

    # Absorption + signature for every frontier state
    n_out = np.int32(0)
    for fi in range(flen):
        a3 = f_a[fi]; s3 = f_s[fi]; p3 = f_p[fi]; c3 = f_c[fi]
        a1   = ((a3 // f) << table_log) + (a3 % f) + c
        sr   = dfa_r[s3]
        if sr == np.int32(0):
            continue
        si   = int((a1 + par_eager) % sr)
        sb   = dfa_ab[s3 * max_radix + si]
        ss   = dfa_an[s3 * max_radix + si]
        if n_out >= O_MAX:
            break
        if p3 >= dna_len:
            out_ans[n_out]=a1; out_st[n_out]=ss; out_par[n_out]=par_eager
            out_pos[n_out]=p3+np.int32(1); out_cost[n_out]=c3+C_BEYOND
            n_out += np.int32(1)
        elif dna_ids[p3] == sb:
            out_ans[n_out]=a1; out_st[n_out]=ss; out_par[n_out]=par_eager
            out_pos[n_out]=p3+np.int32(1); out_cost[n_out]=c3
            n_out += np.int32(1)
        else:
            if n_out < O_MAX:
                out_ans[n_out]=a1; out_st[n_out]=ss; out_par[n_out]=par_eager
                out_pos[n_out]=p3+np.int32(1); out_cost[n_out]=c3+C_SUB
                n_out += np.int32(1)
            osn3 = dfa_obs[s3 * np.int32(4) + dna_ids[p3]]
            if osn3 >= np.int32(0) and n_out < O_MAX:
                out_ans[n_out]=a1; out_st[n_out]=osn3; out_par[n_out]=par_eager
                out_pos[n_out]=p3+np.int32(1); out_cost[n_out]=c3+C_SUB+np.int32(1)
                n_out += np.int32(1)
            if n_out < O_MAX:
                out_ans[n_out]=a1; out_st[n_out]=ss; out_par[n_out]=par_eager
                out_pos[n_out]=p3; out_cost[n_out]=c3+C_DEL
                n_out += np.int32(1)

    return n_out


# ── main decoder ─────────────────────────────────────────────────────────
def tans_decode_constrained_dna_direct(
    dna_sequence: str,
    dfa,
    decode_info: Dict,
    *,
    beam_width: int = 512,
    k_tiers: int = DEFAULT_K_TIERS,
    _int_dfa: Optional[IntDFA] = None,
    strict_tail: bool = False,
    return_cost: bool = False,
) -> np.ndarray:
    """Viterbi beam-search decoder for constrained-DNA tANS.

    If ``return_cost=True``, returns ``(symbols, best_cost)`` where best_cost
    is the terminal-verified cost of the chosen path (0 == perfect tail/parity
    match). This is the erasure signal of erasure_syn.pdf 方向②.

    改法二：Rolling layers — 用 numpy parent_sids/parent_slots 替换全量
            layers:List[Dict]，内存从 O(n×B×200B) 降到 O(n×B×4B)。

    改法三：Numba JIT — 若安装了 numba，_step_symbol_nb 替代 Python 版本，
            热循环速度提升 50–100×；未安装时自动 fallback 到纯 Python。

    改法四：parity_bits 请在编码端升至 8（见 _build_parity_lut 注释）。

    Pass ``_int_dfa`` to reuse across multiple calls with the same DFA.
    """
    dna_bytes = dna_sequence.encode("ascii") if isinstance(dna_sequence, str) else dna_sequence
    n_payload = int(decode_info["sequence_length"])
    n_flush   = int(decode_info.get("n_flush_symbols", 0))
    n         = n_payload + n_flush
    if n_payload == 0:
        return (np.asarray([], dtype=np.int64), 0.0) if return_cost else np.asarray([], dtype=np.int64)

    symbols      = np.asarray(decode_info["symbol_values"], dtype=np.int64)
    freqs_np     = np.asarray(decode_info["freqs"],         dtype=np.int64)
    starts_np    = np.asarray(decode_info["starts"],        dtype=np.int64)
    table_log    = int(decode_info["table_log"])
    table_size   = int(decode_info["table_size"])
    initial_ans_state = int(decode_info["initial_ans_state"])
    ans_memory_k = int(decode_info.get("ans_memory_k", 16))
    parity_bits  = int(decode_info["parity_bits"])

    _max_ans_estimate = ((table_size - 1) << table_log) + table_size * 2
    _min_k       = max(_max_ans_estimate.bit_length(), initial_ans_state.bit_length())
    ans_memory_k = max(ans_memory_k, _min_k)
    ans_mask     = (1 << int(max(1, ans_memory_k))) - 1
    alpha        = int(freqs_np.size)
    beam_width   = int(max(32, beam_width))

    # ── precompute ──────────────────────────────────────────────────────────
    D       = _int_dfa if _int_dfa is not None else IntDFA(dfa)
    par_lut = _build_parity_lut(alpha, parity_bits)

    freqs  = freqs_np.tolist()
    starts = starts_np.tolist()
    sorted_sids = sorted(range(alpha), key=lambda x: freqs[x], reverse=True)

    dfa_ab  = D.allowed_bases
    dfa_an  = D.allowed_nexts
    dfa_r   = D.radix
    dfa_obs = D.obs_next           # numpy int32 flat array

    _b2i   = _BASE_TO_ID
    dna_ids = [_b2i.get(chr(b), -1) if isinstance(b, int) else _b2i.get(b, -1)
               for b in dna_bytes]
    dna_len = len(dna_ids)

    # ── Numba-ready flat DFA arrays (built once, shared) ────────────────────
    if _HAS_NUMBA:
        _dfa_r_nb, _dfa_ab_nb, _dfa_an_nb, _max_radix = _prepare_dfa_arrays(D)
        _dna_ids_nb = np.array(dna_ids, dtype=np.int32)
        _dna_len_nb = np.int32(dna_len)
        _tlog_nb    = np.int32(table_log)
        _C_SUB_nb   = np.int32(COST_SUBSTITUTION)
        _C_DEL_nb   = np.int32(COST_DELETION)
        _C_BEY_nb   = np.int32(COST_BEYOND_DNA)
        _parity_mask_nb = np.int32((1 << parity_bits) - 1)
        # Pre-allocate output buffers (reused across all _step_symbol_nb calls)
        _nb_out_ans  = np.empty(_NB_O_MAX, dtype=np.int64)
        _nb_out_st   = np.empty(_NB_O_MAX, dtype=np.int32)
        _nb_out_par  = np.empty(_NB_O_MAX, dtype=np.int32)
        _nb_out_pos  = np.empty(_NB_O_MAX, dtype=np.int32)
        _nb_out_cost = np.empty(_NB_O_MAX, dtype=np.int32)

    # ── _step_symbol (Python fallback) — fully inlined ──────────────────────
    def _step_symbol(ans0, st0, par0, sid, p0, threshold):
        if threshold < 0: return []
        f = freqs[sid]; c = starts[sid]
        par_eager = par_lut[par0][sid]
        ans = ans0; st = st0; p = p0; cost = 0; single = True
        while True:
            r = dfa_r[st]
            if ans < f * r: break
            if cost > threshold: return []
            idx = (ans + par_eager) % r
            base_id = dfa_ab[st][idx]; st_next = dfa_an[st][idx]; ans_next = ans // r
            if p >= dna_len:
                ans = ans_next; st = st_next; p += 1; cost += COST_BEYOND_DNA
            elif dna_ids[p] == base_id:
                ans = ans_next; st = st_next; p += 1
            else:
                single = False; break
        if single:
            if cost > threshold: return []
            ans1 = ((ans // f) << table_log) + (ans % f) + c
            sig_r = dfa_r[st]
            if sig_r == 0: return []
            sig_idx = (ans1 + par_eager) % sig_r
            sig_base_id = dfa_ab[st][sig_idx]; sig_st = dfa_an[st][sig_idx]
            if p >= dna_len:
                return [(ans1, sig_st, par_eager, p + 1, cost + COST_BEYOND_DNA)]
            elif dna_ids[p] == sig_base_id:
                return [(ans1, sig_st, par_eager, p + 1, cost)]
            else:
                out = [(ans1, sig_st, par_eager, p + 1, cost + COST_SUBSTITUTION)]
                obs_sig_nxt = dfa_obs[st * 4 + dna_ids[p]]
                if obs_sig_nxt >= 0:
                    out.append((ans1, obs_sig_nxt, par_eager, p + 1, cost + COST_SUBSTITUTION + 1))
                out.append((ans1, sig_st, par_eager, p, cost + COST_DELETION))
                return out
        # slow path (unchanged from v1)
        r = dfa_r[st]; idx = (ans + par_eager) % r
        st_next = dfa_an[st][idx]; ans_next = ans // r
        nf: Dict[Tuple[int, int, int], int] = {}
        obs = dna_ids[p]
        if cost + COST_SUBSTITUTION <= threshold:
            nf[(ans_next, st_next, p + 1)] = cost + COST_SUBSTITUTION
        obs_nxt = dfa_obs[st * 4 + obs]
        if obs_nxt >= 0 and cost + COST_SUBSTITUTION + 1 <= threshold:
            nf[(ans_next, int(obs_nxt), p + 1)] = cost + COST_SUBSTITUTION + 1
        key_del = (ans_next, st_next, p); c_del = cost + COST_DELETION
        if c_del <= threshold:
            prev_del = nf.get(key_del)
            if prev_del is None or c_del < prev_del:
                nf[key_del] = c_del
        f_ans = []; f_st = []; f_p = []; f_cost = []
        for (a, s, pp), cc in nf.items():
            f_ans.append(a); f_st.append(s); f_p.append(pp); f_cost.append(cc)
        flen = len(f_ans)
        while True:
            if flen == 0: return []
            all_done = True
            for fi in range(flen):
                if f_ans[fi] >= f * dfa_r[f_st[fi]]: all_done = False; break
            if all_done: break
            nf = {}
            for fi in range(flen):
                a2 = f_ans[fi]; s2 = f_st[fi]; p2 = f_p[fi]; c2 = f_cost[fi]
                if c2 > threshold: continue
                if a2 < f * dfa_r[s2]:
                    key = (a2, s2, p2); prev = nf.get(key)
                    if prev is None or c2 < prev: nf[key] = c2
                    continue
                r2 = dfa_r[s2]; idx2 = (a2 + par_eager) % r2
                bid2 = dfa_ab[s2][idx2]; sn2 = dfa_an[s2][idx2]; an2 = a2 // r2
                if p2 >= dna_len:
                    key = (an2, sn2, p2 + 1); cn = c2 + COST_BEYOND_DNA
                    prev = nf.get(key)
                    if prev is None or cn < prev: nf[key] = cn
                    continue
                ob2 = dna_ids[p2]
                if ob2 == bid2:
                    key = (an2, sn2, p2 + 1); prev = nf.get(key)
                    if prev is None or c2 < prev: nf[key] = c2
                else:
                    cs = c2 + COST_SUBSTITUTION; key = (an2, sn2, p2 + 1)
                    prev = nf.get(key)
                    if prev is None or cs < prev: nf[key] = cs
                    on2 = dfa_obs[s2 * 4 + ob2]
                    if on2 >= 0:
                        co = c2 + COST_SUBSTITUTION + 1; key2 = (an2, int(on2), p2 + 1)
                        prev2 = nf.get(key2)
                        if prev2 is None or co < prev2: nf[key2] = co
                    cd = c2 + COST_DELETION; key3 = (an2, sn2, p2)
                    prev3 = nf.get(key3)
                    if prev3 is None or cd < prev3: nf[key3] = cd
            nf_len = len(nf)
            if nf_len > beam_width:
                items_sorted = sorted(nf.items(), key=lambda kv: kv[1])
                kept = []; ts = 0; pc = None
                for kv in items_sorted:
                    cc = kv[1]
                    if cc != pc: ts += 1; pc = cc
                    if ts > k_tiers or len(kept) >= beam_width: break
                    kept.append(kv)
                items = kept
            else:
                items = nf.items()
            f_ans = []; f_st = []; f_p = []; f_cost = []
            for (a, s, pp), cc in items:
                f_ans.append(a); f_st.append(s); f_p.append(pp); f_cost.append(cc)
            flen = len(f_ans)
        out = []
        for fi in range(flen):
            a3 = f_ans[fi]; s3 = f_st[fi]; p3 = f_p[fi]; c3 = f_cost[fi]
            a1 = ((a3 // f) << table_log) + (a3 % f) + c
            sr = dfa_r[s3]
            if sr == 0: continue
            si = (a1 + par_eager) % sr; sb = dfa_ab[s3][si]; ss = dfa_an[s3][si]
            if p3 >= dna_len:
                out.append((a1, ss, par_eager, p3 + 1, c3 + COST_BEYOND_DNA))
            elif dna_ids[p3] == sb:
                out.append((a1, ss, par_eager, p3 + 1, c3))
            else:
                out.append((a1, ss, par_eager, p3 + 1, c3 + COST_SUBSTITUTION))
                osn = dfa_obs[s3 * 4 + dna_ids[p3]]
                if osn >= 0:
                    out.append((a1, int(osn), par_eager, p3 + 1, c3 + COST_SUBSTITUTION + 1))
                out.append((a1, ss, par_eager, p3, c3 + COST_DELETION))
        return out

    # ── absorb insertions：返回 (expanded_dict, source_raw_slot_map) ─────────
    # source_raw_slot_map[key] = index in the RAW (pre-insertion) slot list
    # that this state ultimately derives from.  Needed for rolling-layer tracking.
    def _absorb_insertions_tracked(raw_slot_list):
        """Expand by at most MAX_INSERTIONS_PER_STEP insertions.

        Returns
        -------
        expanded : dict  key -> (cost, dummy_parent, dummy_sid, ans_full)
        src_slot : dict  key -> raw slot index (in raw_slot_list)
        """
        expanded = {}
        src_slot: Dict = {}
        for j, (key, val) in enumerate(raw_slot_list):
            expanded[key] = val
            src_slot[key] = j

        current = list(enumerate(raw_slot_list))   # [(raw_slot, (key, val)), ...]
        for _ in range(MAX_INSERTIONS_PER_STEP):
            nxt_cur = []
            for raw_j, (key, val) in current:
                ans_r, st, par, p = key
                if p >= dna_len:
                    continue
                obs_nxt = dfa_obs[st * 4 + dna_ids[p]]
                if obs_nxt < 0:
                    continue
                new_key  = (ans_r, int(obs_nxt), par, p + 1)
                new_cost = val[0] + COST_INSERTION
                cur      = expanded.get(new_key)
                if cur is None or new_cost < cur[0]:
                    new_entry          = (new_cost, val[1], val[2], val[3])
                    expanded[new_key]  = new_entry
                    src_slot[new_key]  = raw_j
                    nxt_cur.append((raw_j, (new_key, new_entry)))
            if not nxt_cur:
                break
            current = nxt_cur
        return expanded, src_slot

    # ====================== Viterbi main loop (改法二: rolling layers) =====
    # Instead of layers:List[Dict] we keep only the *current* beam and store
    # parent information in compact numpy int16 arrays:
    #
    #   parent_sids [step][slot] = symbol sid chosen at this step for slot
    #   parent_slots[step][slot] = slot index in the RAW beam at step-1
    #                              that is this slot's ancestor
    #
    # Memory:  2 × n × B × 2 bytes  (vs. original ~n × B × 200 bytes)
    # ─────────────────────────────────────────────────────────────────────
    parent_sids  = np.empty((n, beam_width), dtype=np.int16)
    parent_slots = np.empty((n, beam_width), dtype=np.int16)
    curr_sizes   = np.empty(n, dtype=np.int32)

    # curr_slot_list: ordered list of (key, (cost, _, _, ans_full)) for current beam
    init_key      = (initial_ans_state & ans_mask, D.start_id, 0, 0)
    init_val      = (0, -1, None, int(initial_ans_state))
    curr_slot_list: List = [(init_key, init_val)]

    for i in range(n):
        # ── insertion expansion with source tracking ──────────────────────
        prev_expanded, prev_src = _absorb_insertions_tracked(curr_slot_list)
        if len(prev_expanded) > beam_width:
            prev_expanded = _tier_beam_dict(prev_expanded,
                                            k_tiers=k_tiers,
                                            max_total=beam_width,
                                            cost_idx=0)
            # keep only surviving keys in src map
            prev_src = {k: v for k, v in prev_src.items() if k in prev_expanded}

        # ── symbol expansion ──────────────────────────────────────────────
        nxt: Dict = {}
        best_cost_nxt = float('inf')
        gap = k_tiers * 2 + 2

        sorted_prev = sorted(prev_expanded.items(), key=lambda kv: kv[1][0])
        for key, val in sorted_prev:
            if val[0] > best_cost_nxt + gap:
                continue
            base_cost  = val[0]
            ans0_full  = val[3]
            _, st0, par0, p0 = key
            raw_parent = prev_src[key]   # ← slot index into curr_slot_list

            for sid in sorted_sids:
                threshold = best_cost_nxt + gap - base_cost
                if threshold > 100:
                    threshold = 100

                # ── Numba path (改法三) ────────────────────────────────────
                if _HAS_NUMBA:
                    par_eager = par_lut[par0][sid]
                    n_out = _step_symbol_nb(
                        np.int64(ans0_full), np.int32(st0), np.int32(p0),
                        np.int64(freqs[sid]), np.int64(starts[sid]), _tlog_nb,
                        np.int32(par_eager), _parity_mask_nb,
                        _dfa_r_nb, _dfa_ab_nb, _dfa_an_nb, dfa_obs, np.int32(_max_radix),
                        _dna_ids_nb, _dna_len_nb,
                        _C_SUB_nb, _C_DEL_nb, _C_BEY_nb, np.int32(threshold),
                        _nb_out_ans, _nb_out_st, _nb_out_par, _nb_out_pos, _nb_out_cost,
                    )
                    for oi in range(n_out):
                        ans1     = int(_nb_out_ans[oi])
                        st1      = int(_nb_out_st[oi])
                        par1     = int(_nb_out_par[oi])
                        p1       = int(_nb_out_pos[oi])
                        add_cost = int(_nb_out_cost[oi])
                        total    = base_cost + add_cost
                        if total < best_cost_nxt:
                            best_cost_nxt = total
                        nk  = (ans1 & ans_mask, st1, par1, p1)
                        cur = nxt.get(nk)
                        if cur is None or total < cur[0]:
                            nxt[nk] = (total, raw_parent, sid, ans1)
                else:
                    # ── Python fallback path ───────────────────────────────
                    outcomes = _step_symbol(ans0_full, st0, par0, sid, p0, threshold)
                    for ans1, st1, par1, p1, add_cost in outcomes:
                        total = base_cost + add_cost
                        if total < best_cost_nxt:
                            best_cost_nxt = total
                        nk  = (ans1 & ans_mask, st1, par1, p1)
                        cur = nxt.get(nk)
                        if cur is None or total < cur[0]:
                            nxt[nk] = (total, raw_parent, sid, ans1)

        if not nxt:
            raise ValueError(f"Viterbi failed at step {i}: no viable paths")
        if len(nxt) > beam_width:
            nxt = _tier_beam_dict(nxt, k_tiers=k_tiers, max_total=beam_width, cost_idx=0)

        # ── pack into ordered slot list; store parent arrays ──────────────
        curr_slot_list = sorted(nxt.items(), key=lambda kv: kv[1][0])
        n_slots = len(curr_slot_list)
        curr_sizes[i] = n_slots
        for j in range(n_slots):
            _val = curr_slot_list[j][1]
            parent_slots[i, j] = _val[1]          # raw slot in previous beam
            parent_sids [i, j] = _val[2] if _val[2] is not None else 0

    # ====================== Tail verification ==============================
    terminal_ans     = int(decode_info.get("final_ans_state", 1))
    use_tail_verify  = (terminal_ans <= 1)

    best_slot = -1
    best_cost = None

    for j, (key, val) in enumerate(curr_slot_list):
        ans_full = val[3]
        cost     = val[0]
        _, st_id, par_val, dna_pos = key

        if use_tail_verify:
            temp_ans = ans_full; temp_st = st_id; tail_cost = 0; pos = dna_pos
            while temp_ans > terminal_ans and pos < dna_len:
                r = dfa_r[temp_st]
                if r == 0:
                    tail_cost += COST_PARITY_MISMATCH; break
                expected_digit   = temp_ans % r
                expected_base_id = dfa_ab[temp_st][expected_digit]
                temp_st_next     = dfa_an[temp_st][expected_digit]
                if pos < dna_len:
                    if dna_ids[pos] != expected_base_id:
                        tail_cost += COST_TAIL_MISMATCH
                else:
                    tail_cost += COST_BEYOND_DNA
                temp_ans //= r; temp_st = temp_st_next; pos += 1
            if temp_ans != terminal_ans:
                tail_cost += COST_PARITY_MISMATCH
            if par_val != 0:
                tail_cost += COST_PARITY_MISMATCH
            cost += tail_cost
        else:
            if ans_full != terminal_ans:
                cost += 1000

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_slot = j

    if best_slot < 0:
        raise ValueError("Viterbi termination failed: no candidate paths")
    if strict_tail and best_cost > 0:
        raise ValueError(f"strict_tail: best path has cost {best_cost} (tail mismatch)")

    # ====================== Backtrack via parent arrays ====================
    sid_encode_order = [0] * n
    slot = best_slot
    for i in range(n - 1, -1, -1):
        sid_encode_order[i] = int(parent_sids[i, slot])
        slot = int(parent_slots[i, slot])

    sid_original_order = list(reversed(sid_encode_order[:n_payload]))
    out = symbols[np.asarray(sid_original_order, dtype=np.int64)]
    if return_cost:
        return np.asarray(out, dtype=np.int64), float(best_cost)
    return np.asarray(out, dtype=np.int64)
