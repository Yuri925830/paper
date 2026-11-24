import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cells = json.loads((root/"source/closedset_evaluation.ipynb").read_text())["cells"]
for arm in ("base", "cycle"):
    out = root/"runs"/arm/"closedset"
    with (out/"generated_meta.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 384
    ns = dict(PROJECT_ROOT=str(root), OUT_ROOT=str(out), DATA_ROOT=str(root/"data"),
              TEXT_PATH=str(root/"data/arctic_bdl_transcript.txt"), all_meta=rows)
    for i in [4,6,16,17,18,19,20,21,22]:
        code = "".join(cells[i]["source"])
        if i == 17:
            code = code.replace("facebook/wav2vec2-large-960h-lv60-self", str(root/"asr_lv60"))
        if i == 19:
            assert all(row["utt_id"] in ns["ref_map"] for row in rows)
        exec(compile(code, f"closedset_cell_{i}", "exec"), ns)
    assert len(ns["cer_df"]) == 384
