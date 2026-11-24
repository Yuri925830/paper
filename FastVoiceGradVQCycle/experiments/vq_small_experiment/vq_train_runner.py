#!/usr/bin/env python3
"""Run the audited FastVoiceGrad trainer with a frozen universal BNF codebook."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import fastvoicegrad_formal as formal

class FrozenVQ:
    def __init__(self, path):
        self.path=Path(path); self.numpy=np.load(self.path).astype("float32"); self.cache={}
    def __call__(self,x):
        torch=sys.modules.get("torch") or __import__("torch")
        key=(str(x.device),str(x.dtype)); c=self.cache.get(key)
        if c is None: c=torch.as_tensor(self.numpy,device=x.device,dtype=x.dtype); self.cache[key]=c
        shape=x.shape; flat=x.transpose(-2,-1).reshape(-1,shape[-2])
        out=[]
        for part in flat.split(4096):
            d=part.square().sum(1,keepdim=True)+c.square().sum(1)[None]-2*part@c.T
            out.append(c[d.argmin(1)])
        return torch.cat(out).reshape(*shape[:-2],shape[-1],shape[-2]).transpose(-2,-1).contiguous()

def main():
    p=argparse.ArgumentParser(add_help=False); p.add_argument("--vq-codebook",required=True); known,rest=p.parse_known_args()
    q=FrozenVQ(known.vq_codebook)
    original_load=formal.load_runtime
    def load_runtime(root):
        runtime=original_load(root)
        old_slice=runtime.training._slice_batch
        def sliced(*values,**options):
            result=old_slice(*values,**options); result["bnf"]=q(result["bnf"]); return result
        runtime.training._slice_batch=sliced
        return runtime
    formal.load_runtime=load_runtime
    original_build=formal.build_components
    def build(runtime,args,device):
        c=original_build(runtime,args,device); old=c["bridge"].extract
        c["bridge"].extract=lambda *v,**kw:q(old(*v,**kw)); c["vq_codebook"]=str(q.path); c["vq_quantizer"]=q
        return c
    formal.build_components=build
    formal.KIND="fastvoicegrad_frozen_vq_probe_v1_5090_20261116"
    formal.main(rest)

if __name__=="__main__": main()
