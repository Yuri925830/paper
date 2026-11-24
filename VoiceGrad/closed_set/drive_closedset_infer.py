"""Generate the original notebook's 384 closed-set conversions for one arm."""
import argparse
import json
import sys
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("arm", choices=("base", "cycle"))
arm = parser.parse_args().arm
root = Path(__file__).resolve().parent
src = root / "source"
sys.path.insert(0, str(src))
out = root / "runs" / arm / "closedset"
out.mkdir(exist_ok=True)
cells = json.loads((src / "closedset_evaluation.ipynb").read_text())["cells"]
hifi = root / "drive_hifigan"
ns = dict(PROJECT_ROOT=str(root), DATA_ROOT=str(root/"data"), OUT_ROOT=str(out),
          CKPT_PATH=str(out.parent/"latest.pt"), GEN_WAV_ROOT=str(out/"generated_wavs"),
          HIFIGAN_ROOT=str(hifi), HIFIGAN_CONFIG=str(hifi/"config_16k.json"),
          HIFIGAN_CKPT=str(hifi/"g_00105000"))
for index in range(4,15):
    if index == 14:
        assert ns["epoch"] == 4005
        torch.manual_seed(1234)
    exec(compile("".join(cells[index]["source"]), f"closedset_cell_{index}", "exec"), ns)
assert len(ns["meta_df"]) == 384 and all(Path(p).is_file() for p in ns["meta_df"].wav_out_path)
