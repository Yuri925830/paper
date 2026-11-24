"""Cycle term for the Drive VoiceGrad sampler; caller supplies its generated-mel BNF extractor."""
import torch
from torch.nn import functional as F


def cycle_term(model, diffusion, mel, source_ids, target_ids, source_bnf, extract_bnf):
    if torch.any(source_ids == target_ids):
        raise ValueError("Cycle targets must differ from their source speakers")
    with torch.no_grad():
        converted = diffusion.sample(model, mel, target_ids, source_bnf, start_level=6)
        converted_bnf = extract_bnf(converted)
    # Reuse the original six reverse updates, with gradients enabled for the return leg.
    returned = type(diffusion).sample.__wrapped__(
        diffusion, model, converted, source_ids, converted_bnf, start_level=6
    )
    return F.l1_loss(returned, mel) / mel.size(1)
