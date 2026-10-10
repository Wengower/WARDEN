from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return to_jsonable(x.item())
        return to_jsonable(x.tolist())
    if hasattr(x, "__dict__"):
        return to_jsonable(vars(x))
    return str(x)


def load_mat(path: str) -> Dict[str, Any]:
    import scipy.io as sio

    try:
        return sio.loadmat(path)
    except NotImplementedError:
        import h5py

        out: Dict[str, Any] = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = np.array(f[k])
        return out


def list_numeric_arrays(mat: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    candidates: List[Tuple[str, np.ndarray]] = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in {"f", "i", "u"}:
            candidates.append((k, v))
    return candidates


def pick_default_abundance_key(mat: Dict[str, Any]) -> str:
    candidates = list_numeric_arrays(mat)
    if not candidates:
        raise ValueError("MAT 文件中未找到数值数组")

    def score(item: Tuple[str, np.ndarray]) -> Tuple[int, int, int]:
        k, v = item
        is_float = 1 if v.dtype.kind == "f" else 0
        size = int(np.prod(v.shape)) if v.size else 0
        is_2d_or_3d = 1 if v.ndim in {2, 3} else 0
        return (is_2d_or_3d, is_float, size)

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0][0]


def as_channels_first(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"期望 2D 或 3D 数组，实际 ndim={arr.ndim} shape={arr.shape}")
    if arr.shape[0] <= 16:
        return arr
    if arr.shape[2] <= 16:
        return np.transpose(arr, (2, 0, 1))
    if arr.shape[1] <= 16:
        return np.transpose(arr, (1, 0, 2))
    return arr


@dataclass(frozen=True)
class HistPDF:
    bin_edges: np.ndarray
    bin_midpoints: np.ndarray
    pmf: np.ndarray
    counts: np.ndarray


def estimate_pdf_hist(samples: np.ndarray, bins: int) -> HistPDF:
    x = np.asarray(samples).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("样本为空或全为 NaN/Inf")

    counts, bin_edges = np.histogram(x, bins=bins)
    pmf = counts.astype(np.float64)
    total = pmf.sum()
    if total <= 0:
        raise ValueError("直方图计数总和为 0")
    pmf /= total
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    return HistPDF(bin_edges=bin_edges, bin_midpoints=bin_midpoints, pmf=pmf, counts=counts)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.ndim != 1 or w.ndim != 1 or v.shape[0] != w.shape[0]:
        raise ValueError("values/weights 必须是一维且等长")
    if np.any(w < 0):
        raise ValueError("weights 不能为负")
    s = w.sum()
    if s <= 0:
        raise ValueError("weights 总和必须为正")
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    qs = np.asarray(qs, dtype=np.float64)
    qs = np.clip(qs, 0.0, 1.0)
    return np.interp(qs, cw, v)


@dataclass
class LloydMaxResult:
    L: int
    boundaries: np.ndarray
    reproduction: np.ndarray
    region_pmf: np.ndarray
    iters: int
    converged: bool


def lloyd_max_from_hist(
    bin_midpoints: np.ndarray,
    pmf: np.ndarray,
    L: int,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> LloydMaxResult:
    if L < 2:
        raise ValueError("L 必须 >= 2")
    m = np.asarray(bin_midpoints, dtype=np.float64).reshape(-1)
    p = np.asarray(pmf, dtype=np.float64).reshape(-1)
    if m.shape[0] != p.shape[0]:
        raise ValueError("bin_midpoints 与 pmf 长度不一致")
    if np.any(p < 0):
        raise ValueError("pmf 不能为负")
    ps = p.sum()
    if not np.isfinite(ps) or ps <= 0:
        raise ValueError("pmf 总和必须为正")
    p = p / ps

    v_min = float(m.min())
    v_max = float(m.max())
    if not (np.isfinite(v_min) and np.isfinite(v_max) and v_max > v_min):
        raise ValueError("样本范围无效")

    qs = (np.arange(L, dtype=np.float64) + 0.5) / L
    r = weighted_quantile(m, p, qs)
    r = np.clip(r, v_min, v_max)
    r = np.maximum.accumulate(r)

    b = np.empty(L + 1, dtype=np.float64)
    b[0] = v_min
    b[-1] = v_max

    eps = (v_max - v_min) * 1e-12 + 1e-15
    converged = False
    region_pmf = np.zeros(L, dtype=np.float64)

    for it in range(1, max_iter + 1):
        b[1:-1] = 0.5 * (r[:-1] + r[1:])
        for i in range(1, L):
            if b[i] <= b[i - 1]:
                b[i] = b[i - 1] + eps
        if b[-1] <= b[-2]:
            b[-1] = b[-2] + eps

        r_prev = r.copy()
        region_pmf[:] = 0.0
        for l in range(L):
            left = b[l]
            right = b[l + 1]
            if l == L - 1:
                mask = (m >= left) & (m <= right)
            else:
                mask = (m >= left) & (m < right)
            pw = p[mask].sum()
            region_pmf[l] = float(pw)
            if pw > 0:
                r[l] = float((p[mask] * m[mask]).sum() / pw)
            else:
                r[l] = 0.5 * (left + right)

        r = np.clip(r, v_min, v_max)
        r = np.maximum.accumulate(r)

        diff = float(np.max(np.abs(r - r_prev)))
        if diff <= tol * max(1.0, (v_max - v_min)):
            converged = True
            return LloydMaxResult(L=L, boundaries=b.copy(), reproduction=r.copy(), region_pmf=region_pmf.copy(), iters=it, converged=True)

    return LloydMaxResult(L=L, boundaries=b.copy(), reproduction=r.copy(), region_pmf=region_pmf.copy(), iters=max_iter, converged=False)


def lloyd_max_from_samples(
    samples: np.ndarray,
    L: int,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> LloydMaxResult:
    if L < 2:
        raise ValueError("L 必须 >= 2")

    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("样本为空或全为 NaN/Inf")

    v_min = float(x.min())
    v_max = float(x.max())
    if not (np.isfinite(v_min) and np.isfinite(v_max) and v_max > v_min):
        raise ValueError("样本范围无效")

    qs = (np.arange(L, dtype=np.float64) + 0.5) / L
    r = np.quantile(x, qs)
    r = np.clip(r, v_min, v_max)
    r = np.sort(r)

    b = np.empty(L + 1, dtype=np.float64)
    b[0] = v_min
    b[-1] = v_max

    eps = (v_max - v_min) * 1e-12 + 1e-15
    rng = np.random.default_rng(0)

    region_pmf = np.zeros(L, dtype=np.float64)
    for it in range(1, max_iter + 1):
        b[1:-1] = 0.5 * (r[:-1] + r[1:])
        for i in range(1, L):
            if b[i] <= b[i - 1]:
                b[i] = b[i - 1] + eps
        if b[-1] <= b[-2]:
            b[-1] = b[-2] + eps

        q = np.digitize(x, b[1:-1], right=False).astype(np.int64)
        q = np.clip(q, 0, L - 1)
        counts = np.bincount(q, minlength=L).astype(np.int64)
        region_pmf = counts.astype(np.float64) / float(x.size)

        r_prev = r.copy()
        empty = counts == 0
        for l in range(L):
            if counts[l] > 0:
                r[l] = float(x[q == l].mean())
        if np.any(empty):
            # 重新安置空级，避免长期空簇导致退化；不改变 L。
            min_dist = np.min(np.abs(x[:, None] - r[None, :]), axis=1)
            candidate = np.argsort(min_dist)[::-1]
            ptr = 0
            for l in np.where(empty)[0]:
                while ptr < candidate.shape[0] and not np.isfinite(min_dist[candidate[ptr]]):
                    ptr += 1
                if ptr < candidate.shape[0]:
                    r[l] = float(x[candidate[ptr]])
                    ptr += 1
                else:
                    r[l] = float(x[rng.integers(0, x.size)])

        r = np.clip(r, v_min, v_max)
        r = np.sort(r)

        diff = float(np.max(np.abs(r - r_prev)))
        if diff <= tol * max(1.0, (v_max - v_min)):
            b[1:-1] = 0.5 * (r[:-1] + r[1:])
            q = np.digitize(x, b[1:-1], right=False).astype(np.int64)
            q = np.clip(q, 0, L - 1)
            counts = np.bincount(q, minlength=L).astype(np.int64)
            region_pmf = counts.astype(np.float64) / float(x.size)
            return LloydMaxResult(L=L, boundaries=b.copy(), reproduction=r.copy(), region_pmf=region_pmf.copy(), iters=it, converged=True)

    b[1:-1] = 0.5 * (r[:-1] + r[1:])
    q = np.digitize(x, b[1:-1], right=False).astype(np.int64)
    q = np.clip(q, 0, L - 1)
    counts = np.bincount(q, minlength=L).astype(np.int64)
    region_pmf = counts.astype(np.float64) / float(x.size)
    return LloydMaxResult(L=L, boundaries=b.copy(), reproduction=r.copy(), region_pmf=region_pmf.copy(), iters=max_iter, converged=False)


def quantize_with_boundaries(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    b = np.asarray(boundaries, dtype=np.float64).reshape(-1)
    if b.ndim != 1 or b.shape[0] < 3:
        raise ValueError("boundaries 形状无效")
    inner = b[1:-1]
    x = np.asarray(values, dtype=np.float64)
    levels = np.digitize(x, inner, right=False).astype(np.int16)
    return levels


def linear_index_row_major(coords_rc0: np.ndarray, width: int) -> np.ndarray:
    rc = np.asarray(coords_rc0, dtype=np.int64)
    if rc.ndim != 2 or rc.shape[1] != 2:
        raise ValueError("coords_rc0 必须是 (N,2) 的 [row,col] 0-based 坐标数组")
    r = rc[:, 0]
    c = rc[:, 1]
    return (r * int(width) + c + 1).astype(np.int64)
