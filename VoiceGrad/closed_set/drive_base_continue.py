"""Continue the Drive checkpoint using its native trainer, five additional epochs."""
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "source"))
import train


def main():
    out = root / "runs" / "base"
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "latest.pt"
    state = torch.load(saved if saved.exists() else root / "source/checkpoints/model_epoch_4000.pt",
                       map_location="cpu", weights_only=False)
    cfg = state["config"]
    assert 4000 <= state["epoch"] <= 4005 and cfg["batch_size"] == 16
    model = train.VoiceGrad(n_mels=80, n_bnf=144, n_channels=512, n_spk=4).cuda()
    model.load_state_dict(state["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    optimizer.load_state_dict(state["optimizer"])
    diffusion = train.VoiceGradDiffusion(n_levels=20, offset=0.008).cuda()
    loader = train.get_dataloader(str(root / "data"), split="train", batch_size=16, num_workers=4)
    for epoch in range(state["epoch"] + 1, 4006):
        train.set_seed(cfg["seed"] + epoch)
        loss, _, _ = train.train_one_epoch(model, diffusion, loader, optimizer, "cuda")
        print(f"completed epoch {epoch}: {loss}", flush=True)
        tmp = saved.with_suffix(".tmp")
        torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        config=cfg, train_loss=loss), tmp)
        tmp.replace(saved)


if __name__ == "__main__":
    main()
