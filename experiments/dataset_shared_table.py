"""数据集共用表完整流程(按正确顺序,每步记录依据):
① 选 transform: 聚合残差熵最小
② 定 gtable_log: 真实编码代表图, 净密度(纯payload, 共用表完全不计)最高
③ 建最终共用表: 用①②建并保存
④ 后续: 报共用 vs 逐图净密度对比
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np

def mod256(a, mode):
    a=a.astype(np.int16); r=a.copy()
    if mode=="raw": pass
    elif mode=="row_first": r[:,1:]=a[:,1:]-a[:,:-1]
    elif mode=="column_first": r[1:,:]=a[1:,:]-a[:-1,:]
    elif mode=="row_second":
        if a.shape[1]>1: r[:,1]=a[:,1]-a[:,0]
        if a.shape[1]>2: r[:,2:]=a[:,2:]-2*a[:,1:-1]+a[:,:-2]
    return (r%256).astype(np.uint8)

def entropy_hist(h):
    h=h.astype(np.float64); t=h.sum(); nz=h[h>0]
    return float(-(nz/t*np.log2(nz/t)).sum()) if t>0 else 0.0


def _table_from_hist(counts, table_log, alphabet_size=256):
    """直接从直方图计数建 tANS 表, 避免 np.repeat 展开大数组(NIH 会爆内存)."""
    import numpy as _np, math as _math
    from stage3_tans import _normalize_counts
    table_log=int(max(int(_math.ceil(_math.log2(max(2,alphabet_size)))),min(int(table_log),15)))
    tsize=1<<table_log
    c=_np.maximum(_np.asarray(counts,dtype=_np.int64)[:alphabet_size],1)
    freqs=_normalize_counts(c,tsize)
    starts=_np.concatenate(([0],_np.cumsum(freqs[:-1],dtype=_np.int64)))
    symbols=_np.arange(alphabet_size,dtype=_np.int64)
    return symbols,freqs.astype(_np.int64),starts.astype(_np.int64),int(table_log)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--code-dir", default="codec")
    ap.add_argument("--logs", default="8,9,10,11,12,13,14,15")
    ap.add_argument("--n-encode", type=int, default=32)
    ap.add_argument("--pb", type=int, default=3)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.code_dir))
    from stage3_tans import (build_global_table, build_constraint_dfa,
        tans_encode_constrained_dna_direct_with_stats)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None
    dfa=build_constraint_dfa(h=5, gc_window=6)

    exts=("*.png","*.jpg","*.jpeg","*.bmp","*.tif")
    files=[]
    for e in exts: files+=glob.glob(os.path.join(args.img_dir,"**",e),recursive=True)
    files=sorted(f for f in files if "mask" not in os.path.basename(f).lower())
    n_total=len(files)
    t0=time.time()
    modes=("raw","row_first","row_second","column_first")

    # ── ① 第一遍聚合: 4种transform各累加聚合直方图 → 选熵最小 ──
    agg={m:np.zeros(256,dtype=np.int64) for m in modes}
    for i,f in enumerate(files):
        try: im=np.asarray(Image.open(f).convert("L"),dtype=np.uint8)
        except: continue
        for m in modes: agg[m]+=np.bincount(mod256(im,m).reshape(-1),minlength=256).astype(np.int64)
        if (i+1)%2000==0: print(f"  ①聚合 {i+1}/{n_total}",flush=True)
    ent={m:entropy_hist(agg[m]) for m in modes}
    best_tf=min(ent,key=ent.get)
    print(f"① transform: "+", ".join(f"{m}={ent[m]:.3f}" for m in modes)+f" → 选 {best_tf}",flush=True)

    # ── ② 扫 gtable_log: 真实编码代表图, 净密度(纯payload)最高 ──
    idx=np.linspace(0,n_total-1,min(args.n_encode,n_total)).astype(int)
    enc_res=[]
    for j in idx:
        im=np.asarray(Image.open(files[j]).convert("L"),dtype=np.uint8)
        enc_res.append(mod256(im,best_tf).reshape(-1).astype(np.int64))
    logs=[int(x) for x in args.logs.split(",")]
    sweep=[]
    for log in logs:
        shared=_table_from_hist(agg[best_tf], log, 256)
        ps=[]; po=[]; sb=[]
        for res in enc_res:
            n=res.size; sb.append(n*8)
            _d,ms=tans_encode_constrained_dna_direct_with_stats(res,dfa,table_log=log,parity_bits=args.pb,
                global_table=shared,include_model_in_meta=False)
            ps.append(int(ms["emissions"]))
            own=build_global_table(res.astype(np.uint8),table_log=log,alphabet_size=256)
            _d2,mo=tans_encode_constrained_dna_direct_with_stats(res,dfa,table_log=log,parity_bits=args.pb,
                global_table=own,include_model_in_meta=False)
            po.append(int(mo["emissions"]))
        ps=np.array(ps,float); po=np.array(po,float); sb=np.array(sb,float)
        header_bytes=4+(256*log+7)//8
        hdr_nt=int(np.ceil(header_bytes*8/1.4))  # 逐图每图付一次(仅用于逐图对比)
        nd_shared=float(np.mean(sb/ps))                  # 共用: 纯payload, 表完全不计
        nd_own=float(np.mean(sb/(po+hdr_nt)))            # 逐图: 每图付整个header
        sweep.append({"log":log,"net_density_shared":round(nd_shared,5),
            "net_density_own":round(nd_own,5),"shared_minus_own":round(nd_shared-nd_own,5),
            "extra_payload_real_mean":round(float((ps-po).mean()),1),"header_nt_per_img":hdr_nt})
        print(f"② log={log}: 共用={nd_shared:.4f} 逐图={nd_own:.4f} 差={nd_shared-nd_own:+.4f}",flush=True)
    best_log=max(sweep,key=lambda r:r["net_density_shared"])["log"]
    best_row=[r for r in sweep if r["log"]==best_log][0]
    print(f"② gtable_log → 选 {best_log} (共用净密度={best_row['net_density_shared']})",flush=True)

    # ── ③ 建最终共用表 ──
    final=_table_from_hist(agg[best_tf], best_log, 256)
    sym,frq,srt,tl=final
    result={"name":args.name,"img_dir":os.path.abspath(args.img_dir),"n_images":n_total,
        "step1_transform":{"chosen":best_tf,"aggregate_residual_entropy":{m:round(ent[m],4) for m in modes},
            "criterion":"聚合残差熵最小"},
        "step2_gtable_log":{"chosen":best_log,"n_encoded":len(enc_res),
            "criterion":"真实编码代表图,共用净密度(纯payload,表完全不计)最高","sweep":sweep},
        "step3_final_shared_table":{"symbols":sym.tolist(),"freqs":frq.tolist(),
            "starts":srt.tolist(),"table_log":int(tl)},
        "elapsed_seconds":round(time.time()-t0,1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)),exist_ok=True)
    json.dump(result,open(args.out,"w"),ensure_ascii=False,indent=2)
    print(f"③ 最终共用表已存 (transform={best_tf}, log={best_log})")
    print(f"saved={os.path.abspath(args.out)}")

if __name__=="__main__": main()
