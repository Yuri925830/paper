"""Continue Drive VoiceGrad with its native noise loss plus six-step cycle loss."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from drive_voicegrad_cycle import cycle_term

root = Path(__file__).resolve().parent
for folder in ("compute_bnf", "drive_hifigan", "source"):
    sys.path.insert(0, str(root / folder))
import train
from env import AttrDict
from models import Generator
from conformer_ppg_model.build_ppg_model import load_ppg_model


def main():
    h = root / "drive_hifigan"
    gen = Generator(AttrDict(json.loads((h / "config_16k.json").read_text()))).cuda()
    gen.load_state_dict(torch.load(h / "g_00105000", map_location="cpu", weights_only=False)["generator"])
    gen.eval().requires_grad_(False)
    gen.remove_weight_norm()
    p = root / "compute_bnf/conformer_ppg_model/en_conformer_ctc_att"
    encoder = load_ppg_model(p / "config.yaml", p / "24epoch.pth", "cuda").requires_grad_(False)
    mean, std = [torch.as_tensor(np.load(root / f"data/stats/mel_{name}.npy")).float().cuda().reshape(1,80,1) for name in ("mean","std")]
    def extract(x):
        audio = gen(x * std + mean).squeeze(1)
        lengths = torch.full((len(audio),), audio.shape[1], device=x.device, dtype=torch.long)
        return F.interpolate(encoder(audio, lengths).transpose(1,2), size=x.shape[-1], mode="linear", align_corners=False)
    out = root / "runs/cycle"
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "latest.pt"
    c = torch.load(saved if saved.exists() else root / "source/checkpoints/model_epoch_4000.pt",map_location="cpu",weights_only=False)
    cfg = c["config"]
    model = train.VoiceGrad(n_mels=80,n_bnf=144,n_channels=512,n_spk=4).cuda()
    model.load_state_dict(c["model"])
    opt = torch.optim.Adam(model.parameters(),lr=cfg["lr"])
    opt.load_state_dict(c["optimizer"])
    diffusion = train.VoiceGradDiffusion(n_levels=20,offset=0.008).cuda()
    loader = train.get_dataloader(str(root/"data"),split="train",batch_size=16,num_workers=4)
    for epoch in range(c["epoch"]+1,4006):
        train.set_seed(cfg["seed"]+epoch)
        model.train()
        for i,b in enumerate(loader):
            mel,bnf,spk = [b[k].cuda() for k in ("mel","bnf","spk_id")]
            t = torch.randint(0,20,(len(mel),),device="cuda")
            noise = torch.randn_like(mel)
            loss = F.l1_loss(model(diffusion.q_sample(mel,t,noise),t,spk,bnf),noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            n = max(1,3*len(mel)//4)
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(cfg["seed"]+epoch*1000+i)
                target = (spk[:n]+torch.randint(1,4,(n,),device="cuda"))%4
                for j in range(n):
                    s = slice(j,j+1)
                    term = cycle_term(model,diffusion,mel[s],spk[s],target[s],bnf[s],extract)
                    (term/n).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip_max_norm"],error_if_nonfinite=True)
            opt.step()
            if i%10==0: print(epoch,i,float(loss.detach()),float(term.detach()),flush=True)
        tmp=saved.with_suffix(".tmp")
        torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),config=cfg),tmp)
        tmp.replace(saved)
        print("completed epoch",epoch,flush=True)

if __name__=="__main__":
    main()
