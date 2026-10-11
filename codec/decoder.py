"""Forward symbol-macro beam decoder for the fixed-length DNA codec.

Each source-symbol candidate is encoded deterministically into its whole DNA
word: rANS renormalisation bases plus the number of path-verification bases
specified by the codec metadata.  A thresholded local edit alignment compares
that word with the remaining received DNA, so edit histories are merged inside
one macro instead of becoming global beam paths.

The payload length may be unknown.  Candidate depths are terminated and
strongly verified with the current joint ANS/parity state trailer.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from core import (
    SIGNATURE_HISTORY_INIT,
    SIGNATURE_MODE_ADAPTIVE_V1,
    SIGNATURE_MODE_MIXED_V2,
    TERMINAL_MODE_JOINT_STATE_V1,
    VALID_SIGNATURE_MODES,
    _joint_state_trailer_to_dna,
    _mix64,
    _precompute_dfa_cache,
    signature_count_for_mode,
    signature_index,
    update_parity,
)


_U64_MASK = (1 << 64) - 1


def _numeric_dfa_cache(dfa_cache):
    """Replace string DFA states by dense integer ids in decoder hot loops."""
    state_names = tuple(dfa_cache.keys())
    state_to_id = {state: index for index, state in enumerate(state_names)}
    numeric = []
    for state in state_names:
        choices, next_states, radix = dfa_cache[state]
        numeric.append((
            choices,
            tuple(state_to_id[next_state] for next_state in next_states),
            int(radix),
        ))
    return tuple(numeric), state_names, state_to_id


def _precompute_signature_terms(
    alpha: int, parity_states: int, max_steps: int, max_position: int,
):
    """Precompute all mixed_v2 operands independent of a beam node.

    The two SplitMix64 finalizers still run for every candidate because their
    inputs contain the path-dependent history and ANS state.  Symbol, parity,
    step, and DNA-position multiplications/masks need not be repeated.
    """
    return (
        tuple(
            (sid + 0x9E3779B97F4A7C15) & _U64_MASK
            for sid in range(alpha)
        ),
        tuple(
            ((step + 1) * 0xD6E8FEB86659FD93) & _U64_MASK
            for step in range(max_steps)
        ),
        tuple(
            ((parity + 1) * 0xA24BAED4963EE407) & _U64_MASK
            for parity in range(parity_states)
        ),
        tuple(
            ((sid + 1) * 0x9FB21C651E98DF25) & _U64_MASK
            for sid in range(alpha)
        ),
        tuple(
            ((step + 1) * 0xC13FA9A902A6328F) & _U64_MASK
            for step in range(max_steps)
        ),
        tuple(
            ((position + 1) * 0x91E10DA5C79E7B1D) & _U64_MASK
            for position in range(max_position + 1)
        ),
    )


def _matches_after_removing_one(longer: str, shorter: str) -> bool:
    """Whether deleting exactly one character from ``longer`` yields shorter."""
    if len(longer) != len(shorter) + 1:
        return False
    i = j = 0
    skipped = False
    while i < len(longer) and j < len(shorter):
        if longer[i] == shorter[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            i += 1
    return True


def _prefix_edit_outcomes(
    word: str,
    received: str,
    start: int,
    edit_radius: int,
) -> List[Tuple[int, int]]:
    """Align ``word`` to prefixes of ``received[start:]`` within a threshold.

    Returns nondominated ``(new_received_position, edit_count)`` outcomes.  A
    different endpoint is semantically different because it represents a
    different insertion/deletion offset at the next symbol boundary.
    """
    edit_radius = int(edit_radius)
    if edit_radius < 0:
        return []
    n = len(word)
    available = len(received) - int(start)

    # Hot paths for the intended low-error oligo regime.  These avoid building
    # a DP matrix for the hundreds of thousands of symbol candidates that are
    # rejected after an exact match or one edit.
    if edit_radius == 0:
        if available >= n and received[start:start + n] == word:
            return [(start + n, 0)]
        return []
    if edit_radius == 1:
        outcomes: List[Tuple[int, int]] = []
        if available >= n:
            obs = received[start:start + n]
            mismatches = sum(a != b for a, b in zip(word, obs))
            if mismatches <= 1:
                outcomes.append((start + n, mismatches))
        # One true-DNA deletion: expected word has one extra base.
        if n > 0 and available >= n - 1:
            obs = received[start:start + n - 1]
            if _matches_after_removing_one(word, obs):
                outcomes.append((start + n - 1, 1))
        # One channel insertion: observed prefix has one extra base.
        if available >= n + 1:
            obs = received[start:start + n + 1]
            if _matches_after_removing_one(obs, word):
                outcomes.append((start + n + 1, 1))
        # Endpoint/cost is the complete state for this local alignment.
        return sorted(set(outcomes))

    max_consume = min(available, n + edit_radius)
    min_consume = max(0, n - edit_radius)
    inf = edit_radius + n + max_consume + 1

    prev = list(range(max_consume + 1))  # leading observed bases = insertions
    for a in range(1, n + 1):
        curr = [inf] * (max_consume + 1)
        curr[0] = a  # deletions
        lo = max(1, a - edit_radius)
        hi = min(max_consume, a + edit_radius)
        for b in range(lo, hi + 1):
            sub = prev[b - 1] + (word[a - 1] != received[start + b - 1])
            ins = curr[b - 1] + 1
            delete = prev[b] + 1
            curr[b] = min(sub, ins, delete)
        prev = curr

    out: List[Tuple[int, int]] = []
    for consumed in range(min_consume, max_consume + 1):
        cost = int(prev[consumed])
        if cost <= edit_radius:
            out.append((start + consumed, cost))
    return out


def _bounded_edit_distance(a: str, b: str, threshold: int) -> int | None:
    """Thresholded global Levenshtein distance using Myers bit vectors.

    Python integers provide an arbitrary-width bitset, so a complete 130-nt
    trailer is processed in O(len(b)) big-integer operations instead of an
    O(len(a)*len(b)) Python DP matrix.
    """
    threshold = int(threshold)
    if abs(len(a) - len(b)) > threshold:
        return None
    if not a:
        value = len(b)
        return value if value <= threshold else None

    peq = {}
    for offset, char in enumerate(a):
        peq[char] = peq.get(char, 0) | (1 << offset)
    high_bit = 1 << (len(a) - 1)
    pv = ~0
    mv = 0
    value = len(a)
    for char in b:
        eq = peq.get(char, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh
        if ph & high_bit:
            value += 1
        elif mh & high_bit:
            value -= 1
        ph = (ph << 1) | 1
        mh <<= 1
        pv = mh | ~(xv | ph)
        mv = ph & xv
    return value if value <= threshold else None


def _model_from_inputs(decode_info: Dict, global_table=None):
    if global_table is not None:
        symbols, freqs, starts, table_log = global_table
        return (
            np.asarray(symbols, dtype=np.int64),
            np.asarray(freqs, dtype=np.int64),
            np.asarray(starts, dtype=np.int64),
            int(table_log),
        )
    return (
        np.asarray(decode_info["symbol_values"], dtype=np.int64),
        np.asarray(decode_info["freqs"], dtype=np.int64),
        np.asarray(decode_info["starts"], dtype=np.int64),
        int(decode_info["table_log"]),
    )


def _encode_symbol_word(
    sid: int,
    ans: int,
    dfa_state: str,
    parity: int,
    clean_pos: int,
    sig_history: int,
    sig_step: int,
    *,
    freqs: Sequence[int],
    starts: Sequence[int],
    table_log: int,
    parity_bits: int,
    dfa_cache,
    signature_mode: str,
    parity_transition=None,
    signature_terms=None,
) -> Tuple[int, str, int, int, int, str]:
    """Pure one-symbol encoder used to form a deterministic macro word."""
    sid = int(sid)
    f = int(freqs[sid])
    c = int(starts[sid])
    # During beam decoding the same (parity, sid) pairs are evaluated many
    # thousands of times.  Their update is independent of ANS/DFA/history, so
    # use the decoder-wide 2-D table when available instead of re-hashing sid.
    parity_new = (
        int(parity_transition[int(parity)][sid])
        if parity_transition is not None
        else update_parity(parity, sid, parity_bits)
    )
    chars: List[str] = []
    x = int(ans)
    st = dfa_state
    while True:
        choices, nexts, radix = dfa_cache[st]
        if x < f * radix:
            break
        idx = (x + parity_new) % radix
        chars.append(choices[idx])
        st = nexts[idx]
        x //= radix
    x = ((x // f) << table_log) + (x % f) + c
    n_renorm = len(chars)
    if signature_terms is not None and signature_mode == SIGNATURE_MODE_MIXED_V2:
        choices, nexts, radix = dfa_cache[st]
        sig_pos = int(clean_pos) + len(chars)
        (
            history_sid, history_step, signature_parity,
            signature_sid, signature_step, signature_position,
        ) = signature_terms
        history_new = _mix64(
            int(sig_history) ^ history_sid[sid] ^ history_step[sig_step]
        )
        mixed = (
            (int(x) & _U64_MASK)
            ^ signature_parity[parity_new]
            ^ signature_sid[sid]
            ^ signature_step[sig_step]
            ^ (
                signature_position[sig_pos]
                if sig_pos < len(signature_position)
                else ((sig_pos + 1) * 0x91E10DA5C79E7B1D) & _U64_MASK
            )
            ^ history_new
        )
        sig_idx = (_mix64(mixed) * radix) >> 64
        chars.append(choices[sig_idx])
        st = nexts[sig_idx]
    else:
        history_new = int(sig_history)
        n_signatures = signature_count_for_mode(signature_mode, n_renorm)
        for signature_ordinal in range(n_signatures):
            choices, nexts, radix = dfa_cache[st]
            sig_pos = int(clean_pos) + len(chars)
            sig_idx, history_new = signature_index(
                x, parity_new, sid, sig_step, sig_pos, history_new,
                radix, signature_mode,
                signature_ordinal=signature_ordinal,
                renorm_emissions=n_renorm,
            )
            chars.append(choices[sig_idx])
            st = nexts[sig_idx]
    return x, st, parity_new, int(history_new), sig_step + 1, "".join(chars)


def _mix64_array(values):
    """Vectorized SplitMix64 finalizer, bit-identical to ``_mix64``."""
    values = np.asarray(values, dtype=np.uint64).copy()
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return values


def _multiply_high_small_u64(values, radices):
    """Return high64(values * radix) exactly for radix in {2,3,4}."""
    values = np.asarray(values, dtype=np.uint64)
    radices = np.asarray(radices, dtype=np.uint8)
    result = np.zeros(values.shape, dtype=np.uint8)
    # floor(u*r/2**64) is the count of exact rational boundaries crossed.
    result += (values >= np.uint64(1 << 63)) & (radices == 2)
    mask3 = radices == 3
    result += mask3 & (values >= np.uint64((1 << 64) // 3 + 1))
    result += mask3 & (values >= np.uint64(((2 << 64) + 2) // 3))
    mask4 = radices == 4
    result += mask4 & (values >= np.uint64(1 << 62))
    result += mask4 & (values >= np.uint64(1 << 63))
    result += mask4 & (values >= np.uint64(3 << 62))
    return result


_NUMPY_BATCH_MIN_SYMBOLS = 160


def _build_numpy_batch_tables(
    dfa_cache, parity_transition, signature_terms, freqs, starts,
):
    """Build dense NumPy views once per decoder invocation."""
    state_count = len(dfa_cache)
    dfa_radix = np.empty(state_count, dtype=np.uint8)
    dfa_next = np.zeros((state_count, 4), dtype=np.int32)
    dfa_base = np.zeros((state_count, 4), dtype=np.uint8)
    for state, (choices, next_states, radix) in enumerate(dfa_cache):
        dfa_radix[state] = int(radix)
        for index in range(int(radix)):
            dfa_next[state, index] = int(next_states[index])
            dfa_base[state, index] = ord(choices[index])
    return (
        dfa_radix,
        dfa_next,
        dfa_base,
        np.asarray(parity_transition, dtype=np.uint8),
        tuple(np.asarray(values, dtype=np.uint64) for values in signature_terms),
        np.asarray(freqs, dtype=np.uint64),
        np.asarray(starts, dtype=np.uint64),
    )


def _encode_symbol_words_batch_scalar(
    sids,
    ans: int,
    dfa_state: int,
    parity: int,
    clean_pos: int,
    sig_history: int,
    sig_step: int,
    *,
    freqs,
    starts,
    table_log: int,
    dfa_cache,
    parity_transition,
    signature_terms,
):
    """Generate all requested mixed_v2 symbol macros for one parent.

    Parent state and precomputed tables are localized once.  This replaces up
    to 256 Python calls to ``_encode_symbol_word`` in the decoder hot loop.
    """
    results = {}
    parent_ans = int(ans)
    parent_state = int(dfa_state)
    parent_parity = int(parity)
    parent_clean_pos = int(clean_pos)
    parent_history = int(sig_history)
    step = int(sig_step)
    next_step = step + 1
    history_sid, history_step, signature_parity, signature_sid, signature_step, signature_position = signature_terms
    history_step_value = history_step[step]
    signature_step_value = signature_step[step]
    position_count = len(signature_position)
    parity_row = parity_transition[parent_parity]
    dfa_local = dfa_cache
    freqs_local = freqs
    starts_local = starts
    for sid_value in sids:
        sid = int(sid_value)
        f = int(freqs_local[sid])
        c = int(starts_local[sid])
        parity_new = int(parity_row[sid])
        chars = []
        x = parent_ans
        st = parent_state
        while True:
            choices, nexts, radix = dfa_local[st]
            if x < f * radix:
                break
            idx = (x + parity_new) % radix
            chars.append(choices[idx])
            st = nexts[idx]
            x //= radix
        x = ((x // f) << table_log) + (x % f) + c
        choices, nexts, radix = dfa_local[st]
        sig_pos = parent_clean_pos + len(chars)
        history_new = _mix64(
            parent_history ^ history_sid[sid] ^ history_step_value
        )
        position_term = (
            signature_position[sig_pos]
            if sig_pos < position_count
            else ((sig_pos + 1) * 0x91E10DA5C79E7B1D) & _U64_MASK
        )
        mixed = (
            (x & _U64_MASK)
            ^ signature_parity[parity_new]
            ^ signature_sid[sid]
            ^ signature_step_value
            ^ position_term
            ^ history_new
        )
        sig_idx = (_mix64(mixed) * radix) >> 64
        chars.append(choices[sig_idx])
        st = nexts[sig_idx]
        results[sid] = (
            x, st, parity_new, history_new, next_step, "".join(chars),
        )
    return results


def _encode_symbol_words_batch(
    sids,
    ans: int,
    dfa_state: int,
    parity: int,
    clean_pos: int,
    sig_history: int,
    sig_step: int,
    *,
    freqs,
    starts,
    table_log: int,
    dfa_cache,
    parity_transition,
    signature_terms,
    numpy_batch_tables=None,
):
    """Generate one parent's mixed_v2 macros with masked NumPy operations."""
    sid_values = tuple(int(value) for value in sids)
    count = len(sid_values)
    if count == 0:
        return {}
    # NumPy's fixed dispatch/allocation cost is larger than the scalar loop
    # after index-prefix constraints have reduced the alphabet substantially.
    # The crossover on the production 256-symbol model is about 150 symbols.
    if (
        numpy_batch_tables is None
        or count < _NUMPY_BATCH_MIN_SYMBOLS
        # Renormalization evaluates (x + parity) before taking a radix.  Keep
        # that addition inside uint64; unusual larger states use exact Python
        # integers in the scalar implementation.
        or not 0 <= int(ans) <= _U64_MASK - (len(parity_transition) - 1)
    ):
        return _encode_symbol_words_batch_scalar(
            sid_values, ans, dfa_state, parity, clean_pos, sig_history, sig_step,
            freqs=freqs, starts=starts, table_log=table_log,
            dfa_cache=dfa_cache, parity_transition=parity_transition,
            signature_terms=signature_terms,
        )
    sid_array = np.fromiter(sid_values, dtype=np.int32, count=count)
    (
        dfa_radix, dfa_next, dfa_base, parity_np, signature_np,
        freqs_np, starts_np,
    ) = numpy_batch_tables
    f = freqs_np[sid_array]
    c = starts_np[sid_array]
    parity_new = parity_np[int(parity), sid_array]
    x = np.full(count, np.uint64(int(ans)), dtype=np.uint64)
    states = np.full(count, int(dfa_state), dtype=np.int32)
    lengths = np.zeros(count, dtype=np.uint8)
    # A normalized symbol macro is tiny in this codec.  Grow only for an
    # unexpected table/DFA combination rather than allocating Python lists.
    codes = np.empty((count, 16), dtype=np.uint8)
    row_indices = np.arange(count, dtype=np.int32)
    while True:
        radices = dfa_radix[states].astype(np.uint64)
        active = x >= f * radices
        if not bool(np.any(active)):
            break
        rows = row_indices[active]
        if int(lengths[rows].max(initial=0)) >= codes.shape[1]:
            codes = np.pad(codes, ((0, 0), (0, codes.shape[1])))
        active_radices = radices[active]
        digits = ((x[active] + parity_new[active]) % active_radices).astype(np.int32)
        active_states = states[active]
        columns = lengths[active].astype(np.int32)
        codes[rows, columns] = dfa_base[active_states, digits]
        states[active] = dfa_next[active_states, digits]
        x[active] //= active_radices
        lengths[active] += 1
    x = ((x // f) << np.uint64(int(table_log))) + (x % f) + c

    history_sid, history_step, signature_parity, signature_sid, signature_step, signature_position = signature_np
    step = int(sig_step)
    history_new = _mix64_array(
        np.uint64(int(sig_history))
        ^ history_sid[sid_array]
        ^ history_step[step]
    )
    sig_positions = int(clean_pos) + lengths.astype(np.int32)
    position_terms = np.empty(count, dtype=np.uint64)
    in_range = sig_positions < len(signature_position)
    position_terms[in_range] = signature_position[sig_positions[in_range]]
    if np.any(~in_range):
        position_terms[~in_range] = (
            (sig_positions[~in_range].astype(np.uint64) + np.uint64(1))
            * np.uint64(0x91E10DA5C79E7B1D)
        )
    mixed = (
        x
        ^ signature_parity[parity_new]
        ^ signature_sid[sid_array]
        ^ signature_step[step]
        ^ position_terms
        ^ history_new
    )
    signature_u = _mix64_array(mixed)
    signature_radices = dfa_radix[states]
    signature_digits = _multiply_high_small_u64(
        signature_u, signature_radices,
    ).astype(np.int32)
    columns = lengths.astype(np.int32)
    if int(columns.max(initial=0)) >= codes.shape[1]:
        codes = np.pad(codes, ((0, 0), (0, codes.shape[1])))
    codes[row_indices, columns] = dfa_base[states, signature_digits]
    states = dfa_next[states, signature_digits]
    lengths += 1

    next_step = step + 1
    results = {}
    for row, sid in enumerate(sid_values):
        word = codes[row, :int(lengths[row])].tobytes().decode("ascii")
        results[sid] = (
            int(x[row]), int(states[row]), int(parity_new[row]),
            int(history_new[row]), next_step, word,
        )
    return results


def _select_boundary_offset_diverse(
    candidates: Dict,
    limit: int,
    target_length: int,
    dna_len: int,
    rank_hints: Dict | None = None,
):
    """Select candidates round-robin over indel offsets and clean boundaries.

    ``delta = received_pos - clean_pos`` is the synchronization hypothesis:
    for a single edit it is normally -1 (deletion), 0 (no indel/substitution),
    or +1 (insertion).  ``clean_pos`` is kept as a second-level bucket because
    variable-length symbol macros can end at different clean DNA boundaries.
    """
    if limit <= 0 or not candidates:
        return {}
    if len(candidates) <= limit:
        return dict(candidates)

    rank_hints = rank_hints or {}
    buckets: Dict[Tuple[int, int, int], deque] = defaultdict(deque)
    for key, val in candidates.items():
        ans, st, par, pos, clean_pos, hist = key
        delta = int(pos) - int(clean_pos)
        length_lb = abs((target_length - clean_pos) - (dna_len - pos))
        # Cost is included because this helper is also used when unused tier
        # quota is borrowed from a mixed-cost remainder.  For candidates that
        # were not lookahead-ranked, use a neutral collision count after all
        # exact predicted matches; this avoids treating "not evaluated" as
        # stronger evidence than an evaluated exact lookahead.
        hint = tuple(rank_hints.get(key, (0, 1 << 30, 0)))
        # The final fields only make ties deterministic; they do not carry a
        # probabilistic preference over ANS/DFA/parity states.
        item_rank = (
            int(val[0]) + 4 * int(length_lb), int(val[0]), hint,
            int(length_lb), int(pos), int(clean_pos), int(ans),
            int(par), str(st), int(hist),
        )
        buckets[(int(val[0]), delta, int(clean_pos))].append((item_rank, key, val))

    for bucket in buckets.values():
        ordered = sorted(bucket, key=lambda item: item[0])
        bucket.clear()
        bucket.extend(ordered)

    # Cost is part of the bucket key.  Consequently, when this helper receives
    # mixed positive-cost candidates, every observed accumulated-cost tier and
    # synchronization hypothesis gets a first representative before any
    # individual bucket gets a second one.  No maximum cumulative cost exists.
    bucket_order = sorted(
        buckets, key=lambda z: (z[0], abs(z[1]), z[1], z[2]),
    )

    kept = {}
    while len(kept) < limit:
        progressed = False
        for bucket_key in bucket_order:
            bucket = buckets[bucket_key]
            if bucket:
                _rank, key, val = bucket.popleft()
                kept[key] = val
                progressed = True
                if len(kept) >= limit:
                    break
        if not progressed:
            break
    return kept


def _prune_diverse(
    candidates: Dict,
    beam_width: int,
    target_length: int,
    dna_len: int,
    *,
    reserve_edited_fraction: float = 0.5,
    rank_hints: Dict | None = None,
):
    """Reserve beam space for all positive accumulated-cost paths.

    Positive paths may have cost 4, 8, 12, ... without an upper bound.  Unused
    reservations are returned to the common pool.
    """
    beam_width = int(beam_width)
    if len(candidates) <= beam_width:
        return candidates

    frac = min(1.0, max(0.0, float(reserve_edited_fraction)))
    edited_cap = int(round(beam_width * frac))
    cost0_cap = beam_width - edited_cap
    zero = {key: val for key, val in candidates.items() if int(val[0]) == 0}
    edited = {key: val for key, val in candidates.items() if int(val[0]) > 0}

    # When both classes exist, do not fill unused zero-cost reservation with a
    # flood of speculative edited paths.  If only one class remains, it may use
    # the full beam.  This is a soft beam occupancy rule, not a cost ceiling.
    def select_edited(source: Dict, limit: int):
        if not source or limit <= 0:
            return {}
        min_cost = min(int(val[0]) for val in source.values())
        frontier = {
            key: val for key, val in source.items() if int(val[0]) == min_cost
        }
        higher = {
            key: val for key, val in source.items() if int(val[0]) > min_cost
        }
        if not higher:
            return _select_boundary_offset_diverse(
                frontier, limit, target_length, dna_len, rank_hints,
            )
        higher_cap = max(1, int(round(limit * 0.25)))
        frontier_cap = limit - higher_cap
        selected = _select_boundary_offset_diverse(
            frontier, frontier_cap, target_length, dna_len, rank_hints,
        )
        selected.update(_select_boundary_offset_diverse(
            higher, higher_cap, target_length, dna_len, rank_hints,
        ))
        return selected

    if zero and edited:
        kept = _select_boundary_offset_diverse(
            zero, cost0_cap, target_length, dna_len, rank_hints,
        )
        kept.update(select_edited(edited, edited_cap))
        return kept
    if zero:
        return _select_boundary_offset_diverse(
            zero, beam_width, target_length, dna_len, rank_hints,
        )
    return select_edited(edited, beam_width)


def _one_symbol_lookahead_rank(
    key,
    next_step: int,
    received: str,
    local_edit_radius: int,
    target_length: int,
    *,
    alpha: int,
    freqs: Sequence[int],
    starts: Sequence[int],
    table_log: int,
    parity_bits: int,
    dfa_cache,
    signature_mode: str,
    candidate_sids=None,
    macro_cache: Dict | None = None,
    alignment_cache: Dict | None = None,
    parity_transition=None,
    signature_terms=None,
    numpy_batch_tables=None,
) -> Tuple[Tuple[int, int, int], int, Dict]:
    """Rank a node by its best *uncommitted* next-symbol observation.

    This is deliberately only a pruning hint: it neither consumes the next
    symbol nor adds its edit cost.  The next decoder iteration performs the
    real transition, avoiding double counting.  Fewer equally good next-symbol
    matches are preferred because they carry more local discrimination.
    """
    ans, st, par, pos, clean_pos, hist = key
    best_edit = int(local_edit_radius) + 1
    best_count = 0
    longest_best_word = 0
    probes = 0
    eligible_sids = range(alpha) if candidate_sids is None else candidate_sids
    transitions = {}
    if macro_cache is None and signature_terms is not None:
        transitions = _encode_symbol_words_batch(
            eligible_sids, ans, st, par, clean_pos, hist, next_step,
            freqs=freqs, starts=starts, table_log=table_log,
            dfa_cache=dfa_cache, parity_transition=parity_transition,
            signature_terms=signature_terms,
            numpy_batch_tables=numpy_batch_tables,
        )
    handoff = {}
    for sid in eligible_sids:
        sid = int(sid)
        macro_key = (
            int(next_step), int(ans), st, int(par), int(clean_pos),
            int(hist), int(sid),
        )
        transition = transitions.get(sid)
        if transition is None and macro_cache is not None:
            transition = macro_cache.get(macro_key)
        if transition is None:
            transition = _encode_symbol_word(
                sid, ans, st, par, clean_pos, hist, next_step,
                freqs=freqs, starts=starts, table_log=table_log,
                parity_bits=parity_bits, dfa_cache=dfa_cache,
                signature_mode=signature_mode,
                parity_transition=parity_transition,
                signature_terms=signature_terms,
            )
            if macro_cache is not None:
                macro_cache[macro_key] = transition
        _x, _st, _p, _h, _step, word = transition
        if clean_pos + len(word) > target_length:
            continue
        probes += 1
        alignment_key = (word, int(pos), int(local_edit_radius))
        outcomes = (
            None if alignment_cache is None
            else alignment_cache.get(alignment_key)
        )
        if outcomes is None:
            outcomes = _prefix_edit_outcomes(
                word, received, pos, local_edit_radius,
            )
            if alignment_cache is not None:
                alignment_cache[alignment_key] = outcomes
        handoff[sid] = (transition, int(local_edit_radius), outcomes)
        local_best = min((cost for _end, cost in outcomes), default=None)
        if local_best is None:
            continue
        if local_best < best_edit:
            best_edit = int(local_best)
            best_count = 1
            longest_best_word = len(word)
        elif local_best == best_edit:
            best_count += 1
            longest_best_word = max(longest_best_word, len(word))
    # Prefer a smaller predicted edit count, then fewer colliding next symbols,
    # then a longer discriminating word.  No component changes actual cost.
    return (best_edit, best_count, -longest_best_word), probes, handoff


def decode_fixed_length_macro(
    dna_sequence: str,
    dfa,
    decode_info: Dict,
    *,
    global_table=None,
    beam_width: int = 256,
    edit_penalty: int = 4,
    relative_cost_window: int = 4,
    reserve_edited_fraction: float = 0.5,
    short_macro_lookahead: int = 4,
    lookahead_pool_factor: float = 1.5,
    share_macro_transitions: bool = False,
    share_alignment_results: bool = True,
    use_numpy_batch: bool = True,
    return_details: bool = False,
):
    """Decode one fixed-length oligo with symbol macro transitions.

    If ``decode_info['n_payload_symbols']`` is absent, candidate terminal depths
    are discovered automatically by jointly verifying the state trailer.
    """
    received = str(dna_sequence)
    symbols, freqs_np, starts_np, table_log = _model_from_inputs(
        decode_info, global_table,
    )
    freqs = freqs_np.tolist()
    starts = starts_np.tolist()
    alpha = len(freqs)
    target_length = int(decode_info.get("dna_length", decode_info.get("expected_dna_length", 130)))
    n_payload = int(
        decode_info.get("n_payload_symbols", decode_info.get("sequence_length", -1))
    )
    auto_payload_length = n_payload < 0
    payload_step_limit = (
        int(decode_info.get("max_payload_symbols", target_length))
        if auto_payload_length else n_payload
    )
    min_payload_symbols = int(decode_info.get("min_payload_symbols", 1))
    linear_index_symbols = int(decode_info.get("linear_index_symbols", 0))
    source_stream_length = int(decode_info.get("source_stream_length", 0))
    parity_bits = int(decode_info["parity_bits"])
    initial_ans = int(decode_info["initial_ans_state"])
    signature_mode = str(
        decode_info.get("signature_mode", SIGNATURE_MODE_MIXED_V2)
    )
    if signature_mode not in VALID_SIGNATURE_MODES:
        raise ValueError(
            f"signature_mode must be one of {VALID_SIGNATURE_MODES}"
        )
    terminal_mode = str(
        decode_info.get("terminal_mode", TERMINAL_MODE_JOINT_STATE_V1)
    )
    if terminal_mode != TERMINAL_MODE_JOINT_STATE_V1:
        raise ValueError(
            "only terminal_mode='joint_state_v1' is supported"
        )
    history_init = int(decode_info.get("signature_history_init", SIGNATURE_HISTORY_INIT))
    string_dfa_cache = _precompute_dfa_cache(dfa)
    dfa_cache, dfa_state_names, dfa_state_to_id = _numeric_dfa_cache(
        string_dfa_cache,
    )
    # Only 2**parity_bits * alphabet_size entries are needed (2048 for the
    # current 3-bit/256-symbol codec).  This removes the repeated symbol hash
    # and mask construction from every formal and lookahead transition.
    parity_transition = tuple(
        tuple(update_parity(parity, sid, parity_bits) for sid in range(alpha))
        for parity in range(1 << parity_bits)
    )
    signature_terms = (
        _precompute_signature_terms(
            alpha, 1 << parity_bits, payload_step_limit + 1, target_length,
        )
        if signature_mode == SIGNATURE_MODE_MIXED_V2 else None
    )
    numpy_batch_tables = (
        _build_numpy_batch_tables(
            dfa_cache, parity_transition, signature_terms, freqs, starts,
        )
        if signature_terms is not None and use_numpy_batch else None
    )
    edit_penalty = int(max(1, edit_penalty))
    relative_cost_window = int(max(0, relative_cost_window))
    reserve_edited_fraction = min(
        1.0, max(0.0, float(reserve_edited_fraction)),
    )
    short_macro_lookahead = int(max(0, short_macro_lookahead))
    lookahead_pool_factor = max(1.0, float(lookahead_pool_factor))

    # Source paths use shared backpointers.  Synchronization hypotheses often
    # share the same symbol prefix, so copying ``path + (sid,)`` into every
    # beam node was both redundant and allocation-heavy.
    path_parent = [-1]
    path_sid = [-1]
    path_depth = [0]
    path_index_value = [0]
    path_tuple_cache = {0: ()}
    path_extension_cache = {}

    def extend_path(parent_id: int, sid: int) -> int:
        extension_key = (int(parent_id), int(sid))
        existing = path_extension_cache.get(extension_key)
        if existing is not None:
            return existing
        new_id = len(path_parent)
        depth = path_depth[parent_id] + 1
        path_parent.append(int(parent_id))
        path_sid.append(int(sid))
        path_depth.append(depth)
        if depth <= linear_index_symbols:
            index_value = (
                (path_index_value[parent_id] << 8)
                | int(symbols[int(sid)])
            )
        else:
            index_value = path_index_value[parent_id]
        path_index_value.append(int(index_value))
        path_extension_cache[extension_key] = new_id
        return new_id

    def path_as_tuple(path_id: int) -> Tuple[int, ...]:
        cached = path_tuple_cache.get(int(path_id))
        if cached is not None:
            return cached
        values = []
        cursor = int(path_id)
        while cursor:
            values.append(path_sid[cursor])
            cursor = path_parent[cursor]
        result = tuple(reversed(values))
        path_tuple_cache[int(path_id)] = result
        return result

    def valid_next_sids(path_id: int):
        if linear_index_symbols == 0:
            return range(alpha)
        if linear_index_symbols < 1 or source_stream_length <= 0:
            raise ValueError(
                "indexed macro decoding requires a positive index width and "
                "source stream length"
            )
        depth = path_depth[path_id]
        if depth < linear_index_symbols:
            valid = []
            for sid in range(alpha):
                prefix = (
                    (path_index_value[path_id] << 8)
                    | int(symbols[int(sid)])
                )
                new_depth = depth + 1
                minimum_start = prefix << (
                    8 * (linear_index_symbols - new_depth)
                )
                if minimum_start < source_stream_length:
                    valid.append(sid)
            return valid
        next_payload_count = depth + 1 - linear_index_symbols
        if path_index_value[path_id] + next_payload_count > source_stream_length:
            return ()
        return range(alpha)

    # key=(ans, dfa_state_id, parity, received_pos, clean_pos, sig_history)
    # val=(cost, path_id, lookahead_handoff_for_this_parent)
    beam: Dict = {
        (
            initial_ans, dfa_state_to_id[dfa.start_state], 0, 0, 0,
            history_init,
        ): (0, 0, None)
    }
    nodes_expanded = 0
    lookahead_nodes = 0
    lookahead_candidates = 0
    transition_cache_hits = 0
    alignment_cache_hits = 0
    macro_memo_hits = 0
    macro_memo_misses = 0
    shared_alignment_hits = 0
    shared_alignment_misses = 0
    local_radius_histogram = defaultdict(int)

    def augmented_cost(key, path_cost: int) -> int:
        """Cost plus an admissible lower bound from the remaining lengths."""
        _ans, _st, _par, pos, clean_pos, _hist = key
        length_lb = abs(
            (target_length - int(clean_pos)) - (len(received) - int(pos))
        )
        return int(path_cost) + edit_penalty * int(length_lb)

    terminal_best_cost = None

    def terminal_candidates(current_beam: Dict, symbol_count: int):
        nonlocal terminal_best_cost
        found = []
        for key, (base_cost, path_id, _handoff) in current_beam.items():
            ans, st, par, pos, clean_pos, hist = key
            trailer = _joint_state_trailer_to_dna(
                ans, par, parity_bits, dfa_state_names[st],
                string_dfa_cache,
                target_length - clean_pos,
            )
            if trailer is None:
                continue
            suffix = "".join(trailer[0])
            n_flush = 0
            if terminal_best_cost is None:
                terminal_radius = max(len(suffix), len(received) - pos)
            else:
                terminal_radius = (
                    int(terminal_best_cost) + relative_cost_window
                    - int(base_cost)
                ) // edit_penalty
                if terminal_radius < 0:
                    continue
            terminal_cost = _bounded_edit_distance(
                suffix, received[pos:], int(terminal_radius),
            )
            if terminal_cost is None:
                continue
            total_cost = int(base_cost) + terminal_cost * edit_penalty
            if terminal_best_cost is None or total_cost < terminal_best_cost:
                terminal_best_cost = int(total_cost)
            found.append((
                total_cost, path_as_tuple(path_id), n_flush, pos, clean_pos,
                int(symbol_count),
            ))
        return found

    terminals = []
    macro_transition_cache: Dict = {}
    shared_alignment_cache: Dict = {}
    for step in range(payload_step_limit):
        nxt: Dict = {}
        needs_lookahead: Dict = {}
        parent_best = min(
            augmented_cost(key, value[0]) for key, value in beam.items()
        )
        parent_threshold = parent_best + relative_cost_window
        for key, (base_cost, path_id, parent_handoff) in beam.items():
            ans, st, par, pos, clean_pos, hist = key
            parent_score = augmented_cost(key, base_cost)
            local_edit_radius = max(
                0, (parent_threshold - parent_score) // edit_penalty,
            )
            local_radius_histogram[int(local_edit_radius)] += 1
            eligible_sids = valid_next_sids(path_id)
            if not eligible_sids:
                continue

            # Lookahead results live directly on the surviving parent.  Only
            # the symbols not already generated there enter this batch.
            transitions = {}
            missing_sids = []
            for sid in eligible_sids:
                cached = (
                    parent_handoff.get(int(sid))
                    if parent_handoff is not None else None
                )
                if cached is None:
                    missing_sids.append(int(sid))
                else:
                    transitions[int(sid)] = cached
                    transition_cache_hits += 1

            if missing_sids and not share_macro_transitions and signature_terms is not None:
                batch = _encode_symbol_words_batch(
                    missing_sids, ans, st, par, clean_pos, hist, step,
                    freqs=freqs, starts=starts, table_log=table_log,
                    dfa_cache=dfa_cache,
                    parity_transition=parity_transition,
                    signature_terms=signature_terms,
                    numpy_batch_tables=numpy_batch_tables,
                )
                transitions.update(
                    (sid, (transition, None, None))
                    for sid, transition in batch.items()
                )
            else:
                for sid in missing_sids:
                    macro_key = (
                        int(step), int(ans), st, int(par), int(clean_pos),
                        int(hist), int(sid),
                    )
                    transition = (
                        macro_transition_cache.get(macro_key)
                        if share_macro_transitions else None
                    )
                    if transition is None:
                        if share_macro_transitions:
                            macro_memo_misses += 1
                        transition = _encode_symbol_word(
                            sid, ans, st, par, clean_pos, hist, step,
                            freqs=freqs, starts=starts, table_log=table_log,
                            parity_bits=parity_bits, dfa_cache=dfa_cache,
                            signature_mode=signature_mode,
                            parity_transition=parity_transition,
                            signature_terms=signature_terms,
                        )
                        if share_macro_transitions:
                            macro_transition_cache[macro_key] = transition
                    else:
                        macro_memo_hits += 1
                    transitions[sid] = (transition, None, None)

            for sid in eligible_sids:
                transition, cached_radius, cached_outcomes = transitions[int(sid)]
                x1, st1, p1, h1, _step1, word = transition
                k1 = clean_pos + len(word)
                if k1 > target_length:
                    continue
                nodes_expanded += 1
                if cached_radius == local_edit_radius:
                    outcomes = cached_outcomes
                else:
                    alignment_key = (word, int(pos), int(local_edit_radius))
                    outcomes = (
                        shared_alignment_cache.get(alignment_key)
                        if share_alignment_results else None
                    )
                    if outcomes is None:
                        if share_alignment_results:
                            shared_alignment_misses += 1
                        outcomes = _prefix_edit_outcomes(
                            word, received, pos, local_edit_radius,
                        )
                        if share_alignment_results:
                            shared_alignment_cache[alignment_key] = outcomes
                    else:
                        shared_alignment_hits += 1
                if cached_radius == local_edit_radius:
                    alignment_cache_hits += 1
                for pos1, local_cost in outcomes:
                    total = int(base_cost) + int(local_cost) * edit_penalty
                    nk = (x1, st1, p1, pos1, k1, h1)
                    old = nxt.get(nk)
                    if old is None or total < old[0]:
                        # Defer allocating the backpointer until this candidate
                        # survives the cost window and diversity preselection.
                        nxt[nk] = (total, path_id, int(sid))
                        needs_lookahead[nk] = (
                            short_macro_lookahead > 0
                            and len(word) <= short_macro_lookahead
                        )
        if not nxt:
            if auto_payload_length and terminals:
                break
            raise ValueError(f"macro beam failed at payload symbol {step}")

        # This is a relative branch-and-bound window, not a cumulative error
        # budget.  The best score may rise at every layer, so paths containing
        # arbitrarily many edits remain representable as long as they stay
        # competitive with the current frontier.
        candidate_best = min(
            augmented_cost(key, value[0]) for key, value in nxt.items()
        )
        candidate_threshold = candidate_best + relative_cost_window
        nxt = {
            key: value for key, value in nxt.items()
            if augmented_cost(key, value[0]) <= candidate_threshold
        }

        # A short current macro often exposes too little information to choose
        # among colliding symbols.  Before the real beam cut, retain a wider
        # cost/offset/boundary-diverse pool and use the next symbol only as an
        # uncommitted ranking hint.  The next iteration still performs and
        # scores that transition normally.
        rank_hints: Dict = {}
        next_macro_transition_cache: Dict = {}
        use_lookahead = (
            len(nxt) > int(beam_width)
            and short_macro_lookahead > 0
            and step + 1 < payload_step_limit
        )
        if use_lookahead:
            pool_width = max(
                int(beam_width),
                int(round(int(beam_width) * lookahead_pool_factor)),
            )
        else:
            pool_width = int(beam_width)
        pool_pending = _prune_diverse(
            nxt, pool_width, target_length, len(received),
            reserve_edited_fraction=reserve_edited_fraction,
        )
        pool = {
            key: (cost, extend_path(parent_id, sid), None)
            for key, (cost, parent_id, sid) in pool_pending.items()
        }
        if use_lookahead:
            for candidate_key, (
                candidate_cost, candidate_path_id, _unused_handoff,
            ) in list(pool.items()):
                if not needs_lookahead.get(candidate_key, False):
                    continue
                lookahead_radius = max(
                    0,
                    (
                        candidate_threshold
                        - augmented_cost(candidate_key, candidate_cost)
                    ) // edit_penalty,
                )
                hint, probes, handoff = _one_symbol_lookahead_rank(
                    candidate_key, step + 1, received,
                    lookahead_radius, target_length,
                    alpha=alpha, freqs=freqs, starts=starts,
                    table_log=table_log, parity_bits=parity_bits,
                    dfa_cache=dfa_cache, signature_mode=signature_mode,
                    candidate_sids=valid_next_sids(candidate_path_id),
                    macro_cache=(
                        next_macro_transition_cache
                        if share_macro_transitions else None
                    ),
                    alignment_cache=(
                        shared_alignment_cache
                        if share_alignment_results else None
                    ),
                    parity_transition=parity_transition,
                    signature_terms=signature_terms,
                    numpy_batch_tables=numpy_batch_tables,
                )
                pool[candidate_key] = (
                    candidate_cost, candidate_path_id, handoff,
                )
                rank_hints[candidate_key] = hint
                lookahead_candidates += 1
                lookahead_nodes += int(probes)

        beam = _prune_diverse(
            pool, int(beam_width), target_length, len(received),
            reserve_edited_fraction=reserve_edited_fraction,
            rank_hints=rank_hints,
        )
        macro_transition_cache = next_macro_transition_cache

        if auto_payload_length and step + 1 >= min_payload_symbols:
            terminals.extend(terminal_candidates(beam, step + 1))
            # Costs are non-negative.  An exact payload+trailer match is
            # therefore globally optimal; the indexed encoder is injective,
            # so continuing to greater source depths cannot improve it.
            if terminal_best_cost == 0:
                break

    if not auto_payload_length:
        terminals = terminal_candidates(beam, n_payload)

    if not terminals:
        raise ValueError("macro terminal verification failed")
    terminals.sort(key=lambda z: z[0])
    best_cost = terminals[0][0]
    best_paths = {item[1] for item in terminals if item[0] == best_cost}
    if len(best_paths) != 1:
        raise ValueError(f"macro decoder ambiguous at cost {best_cost}")
    best_path = next(iter(best_paths))
    best_symbol_count = next(
        item[5] for item in terminals
        if item[0] == best_cost and item[1] == best_path
    )
    original_sids = list(reversed(best_path))
    output = symbols[np.asarray(original_sids, dtype=np.int64)]
    details = {
        "cost": int(best_cost),
        "terminal_candidates": int(len(terminals)),
        "best_path_multiplicity": int(len(best_paths)),
        "nodes_expanded": int(nodes_expanded),
        "lookahead_nodes_expanded": int(lookahead_nodes),
        "lookahead_candidates_ranked": int(lookahead_candidates),
        "lookahead_transition_cache_hits": int(transition_cache_hits),
        "lookahead_alignment_cache_hits": int(alignment_cache_hits),
        "macro_transition_memo_hits": int(macro_memo_hits),
        "macro_transition_memo_misses": int(macro_memo_misses),
        "shared_alignment_cache_hits": int(shared_alignment_hits),
        "shared_alignment_cache_misses": int(shared_alignment_misses),
        "share_macro_transitions": bool(share_macro_transitions),
        "use_numpy_batch": bool(use_numpy_batch),
        "share_alignment_results": bool(share_alignment_results),
        "beam_final": int(len(beam)),
        "edit_penalty": int(edit_penalty),
        "cumulative_edit_budget": None,
        "relative_cost_window": int(relative_cost_window),
        "local_alignment_edit_radius": "dynamic from relative cost slack",
        "local_alignment_radius_histogram": {
            str(radius): int(count)
            for radius, count in sorted(local_radius_histogram.items())
        },
        "reserve_edited_fraction": float(reserve_edited_fraction),
        "short_macro_lookahead": int(short_macro_lookahead),
        "lookahead_pool_factor": float(lookahead_pool_factor),
        "joint_trailer_required": True,
        "terminal_mode": terminal_mode,
        "auto_payload_length": bool(auto_payload_length),
        "decoded_inner_symbol_count": int(best_symbol_count),
        "linear_index_symbols": int(linear_index_symbols),
        "source_stream_length": int(source_stream_length),
    }
    if return_details:
        return np.asarray(output, dtype=np.int64), details
    return np.asarray(output, dtype=np.int64)
