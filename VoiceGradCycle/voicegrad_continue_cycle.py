#!/usr/bin/env python3
"""Continue a mature VoiceGrad checkpoint with an optional paper-scaled Cycle loss."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--extra-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--lambda-cycle", type=float, default=1.0)
    parser.add_argument("--cycle-steps", type=int, default=6)
    parser.add_argument("--cycle-start-step", type=int, default=950)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def target_indices(names, device):
    size = len(names)
    indices = []
    for source in range(size):
        target = (source + 1) % size
        while names[target] == names[source]:
            target = (target + 1) % size
            if target == source:
                raise RuntimeError("A batch must contain at least two speakers for Cycle training.")
        indices.append(target)
    return torch.tensor(indices, device=device, dtype=torch.long)


def masked_cycle_loss(reconstructed, source, lengths):
    frames = source.shape[-1]
    mask = torch.arange(frames, device=source.device)[None, :] < lengths[:, None]
    mask = mask[:, None, :].to(source.dtype)
    per_sample_mae = ((reconstructed - source).abs() * mask).sum((1, 2))
    per_sample_mae = per_sample_mae / (mask.sum((1, 2)).clamp_min(1.0) * source.shape[1])
    return per_sample_mae.mean() / source.shape[1]


def cycle_loss(model, diffusion, batch, device, steps, start_step, cycle_seed):
    source = batch["mel"].to(device, non_blocking=True)
    bnf = batch["bnf"].to(device, non_blocking=True)
    original_speaker = batch["spk_emb"].to(device, non_blocking=True)
    lengths = batch["mel_len"].to(device, non_blocking=True)
    targets = target_indices(batch["spk_name"], device)
    target_speaker = original_speaker.index_select(0, targets)

    generator = torch.Generator(device=device).manual_seed(cycle_seed)
    with torch.no_grad():
        converted = diffusion.sample(
            model,
            source,
            target_speaker,
            bnf,
            num_steps=steps,
            start_step=start_step,
            generator=generator,
            diffuse_source=False,
        )
    reconstructed = type(diffusion).sample.__wrapped__(
        diffusion,
        model,
        converted.detach(),
        original_speaker,
        bnf,
        num_steps=steps,
        start_step=start_step,
        generator=generator,
        diffuse_source=False,
    )
    return masked_cycle_loss(reconstructed, source, lengths)


def main():
    args = parse_args()
    if args.batch_size != 32:
        raise ValueError("This experiment fixes the paper batch size at 32.")
    if args.cycle and args.lambda_cycle != 1.0:
        raise ValueError("The aligned Cycle experiment fixes lambda_cycle=1 after division by 80.")

    root = args.project_root.resolve()
    os.environ["VOICEGRAD_PROJECT_ROOT"] = str(root)
    import sys

    sys.path.insert(0, str(root))
    from config import PAPER_CONFIG
    from dataset import get_dataloader
    from diffusion import VoiceGradDiffusion
    from train import build_model, diffusion_loss

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    loader = get_dataloader(
        root / "data",
        "train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        segment_length=PAPER_CONFIG.train_frames,
        expected_speakers=PAPER_CONFIG.expected_train_speakers,
    )
    model = build_model().to(device)
    diffusion = VoiceGradDiffusion(PAPER_CONFIG.diffusion_steps, PAPER_CONFIG.cosine_offset).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=PAPER_CONFIG.learning_rate,
        betas=(PAPER_CONFIG.adam_beta1, PAPER_CONFIG.adam_beta2),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng_state"])
    parent_epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint.get("global_step", 0))
    end_epoch = parent_epoch + args.extra_epochs

    contract = {
        "model": "VoiceGrad",
        "parent_checkpoint": str(args.checkpoint.resolve()),
        "parent_epoch": parent_epoch,
        "end_epoch": end_epoch,
        "physical_batch_size": 32,
        "effective_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "cycle_enabled": bool(args.cycle),
        "lambda_cycle": 1.0 if args.cycle else 0.0,
        "cycle_reduction": "masked_MAE_mean_then_divide_by_80",
        "cycle_steps_per_leg": args.cycle_steps if args.cycle else 0,
        "first_leg_stop_gradient": True if args.cycle else None,
        "second_leg_gradient": True if args.cycle else None,
        "seed": args.seed,
    }
    atomic_json(output / "contract.json", contract)
    history = []
    started = time.time()

    for epoch in range(parent_epoch + 1, end_epoch + 1):
        model.train()
        base_total = 0.0
        cycle_total = 0.0
        batches = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            base = diffusion_loss(model, diffusion, batch, device)
            if args.cycle:
                cyc = cycle_loss(
                    model,
                    diffusion,
                    batch,
                    device,
                    args.cycle_steps,
                    args.cycle_start_step,
                    args.seed + 50_000_000 + global_step,
                )
                total = base + args.lambda_cycle * cyc
            else:
                cyc = base.new_zeros(())
                total = base
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite loss at global_step={global_step}.")
            total.backward()
            optimizer.step()
            global_step += 1
            batches += 1
            base_total += float(base.detach())
            cycle_total += float(cyc.detach())

        record = {
            "epoch": epoch,
            "global_step": global_step,
            "base_l1": base_total / batches,
            "cycle_raw": cycle_total / batches,
            "total": (base_total + args.lambda_cycle * cycle_total) / batches,
            "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        payload = {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "rng_state": capture_rng(),
            "paper_config": PAPER_CONFIG.as_dict(),
            "experiment_contract": contract,
            "history": history,
        }
        atomic_save(output / "latest.pt", payload)
        atomic_json(status_path, {"state": "training", "latest": record, "contract": contract})

    atomic_save(output / f"epoch_{end_epoch:04d}.pt", payload)
    atomic_json(
        output / "completion.json",
        {"state": "complete", "latest": history[-1], "history": history, "contract": contract},
    )
    atomic_json(status_path, {"state": "complete", "latest": history[-1], "contract": contract})


if __name__ == "__main__":
    main()
