#!/usr/bin/env python3
"""Fit and select frozen VQ codebooks using train BNF only."""
import argparse, csv, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)

def sample_frames(root, seed, fit_n=80000, held_n=20000):
    rows = list(csv.DictReader((root / "data/metadata/train_manifest.csv").open(encoding="utf-8")))
    rng = random.Random(seed); rng.shuffle(rows)
    fit, held = [], []
    for row in rows:
        value = np.load(root / "data/bnf" / row["speaker"] / (row["utt_id"] + ".npy"), mmap_mode="r")
        if value.shape[0] != 144: raise RuntimeError(f"bad BNF {row['utt_id']}: {value.shape}")
        idx = np.linspace(0, value.shape[1] - 1, min(12, value.shape[1]), dtype=np.int64)
        target = fit if len(fit) < fit_n else held
        target.extend(np.asarray(value[:, idx].T, dtype=np.float32))
        if len(fit) >= fit_n and len(held) >= held_n: break
    return np.asarray(fit[:fit_n]), np.asarray(held[:held_n])

def assign(x, centers, chunk=4096):
    ids=[]; errors=[]
    for part in x.split(chunk):
        d = part.square().sum(1, keepdim=True) + centers.square().sum(1)[None] - 2 * part @ centers.T
        e, i = d.min(1); ids.append(i); errors.append(e.clamp_min_(0))
    return torch.cat(ids), torch.cat(errors)

def fit_kmeans(x, k, seed, epochs=12):
    g=torch.Generator(device=x.device).manual_seed(seed+k)
    centers=x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()
    for _ in range(epochs):
        ids,_=assign(x, centers); sums=torch.zeros_like(centers); counts=torch.bincount(ids,minlength=k)
        sums.index_add_(0,ids,x); live=counts>0; centers[live]=sums[live]/counts[live,None]
        if (~live).any(): centers[~live]=x[torch.randperm(len(x),generator=g,device=x.device)[:int((~live).sum())]]
    return centers

def metrics(x, centers):
    ids, sq=assign(x,centers); counts=torch.bincount(ids,minlength=len(centers)).float(); p=counts/counts.sum()
    recon=centers[ids]
    cosine=torch.nn.functional.cosine_similarity(x,recon,dim=1).mean()
    perplexity=torch.exp(-(p[p>0]*p[p>0].log()).sum())
    return {"mse":float(sq.mean()/x.shape[1]),"mae":float((x-recon).abs().mean()),"cosine":float(cosine),"utilization":float((counts>0).float().mean()),"dead_codes":int((counts==0).sum()),"perplexity":float(perplexity),"perplexity_ratio":float(perplexity/len(centers))}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--project-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--seed",type=int,default=1234); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); started=time.time()
    fit_np, held_np=sample_frames(a.project_root,a.seed)
    fit=torch.from_numpy(fit_np).cuda(); held=torch.from_numpy(held_np).cuda()
    result={"protocol":"train_manifest_only; deterministic disjoint frame samples; select lowest heldout MSE among healthy codebooks","seed":a.seed,"fit_frames":len(fit),"heldout_frames":len(held),"candidates":{}}
    for k in (128,256,512):
        c=fit_kmeans(fit,k,a.seed); m=metrics(held,c); np.save(a.output/f"codebook_k{k}.npy",c.cpu().numpy()); result["candidates"][str(k)]=m; atomic_json(a.output/"probe_status.json",result)
    eligible=[int(k) for k,v in result["candidates"].items() if v["utilization"]>=.90 and v["perplexity_ratio"]>=.20 and v["cosine"]>=.90]
    if not eligible: eligible=[int(k) for k in result["candidates"]]
    selected=min(eligible,key=lambda k:(result["candidates"][str(k)]["mse"],k))
    result.update(selected_k=selected,selection_eligible=eligible,elapsed_seconds=time.time()-started,status="complete")
    atomic_json(a.output/"probe_results.json",result); (a.output/"SELECTED_K.txt").write_text(str(selected)+"\n")

if __name__=="__main__": main()
