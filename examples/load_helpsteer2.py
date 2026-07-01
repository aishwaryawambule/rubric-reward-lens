"""Pull a slice of nvidia/HelpSteer2 (real prompts + model responses + human
helpfulness scores) into the rubric-reward-lens responses.json format.

Uses the Hugging Face datasets-server JSON API — no `datasets` install needed.
Run:  python3 examples/load_helpsteer2.py --n 150 --out examples/helpsteer2_responses.json
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"
DATASET = "nvidia/HelpSteer2"


def fetch(n: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < n:
        length = min(100, n - len(rows))  # API caps length at 100 per call
        q = urllib.parse.urlencode(
            {"dataset": DATASET, "config": "default", "split": "train",
             "offset": offset, "length": length}
        )
        with urllib.request.urlopen(f"{API}?{q}", timeout=30) as resp:
            batch = json.load(resp)["rows"]
        if not batch:
            break
        rows.extend(batch)
        offset += length
    return rows[:n]


def to_responses(rows: list[dict]) -> list[dict]:
    out = []
    for i, item in enumerate(rows):
        r = item["row"]
        out.append({
            "id": f"hs2_{item['row_idx']}",
            "prompt": r["prompt"],
            "text": r["response"],
            # HelpSteer2 helpfulness is 0..4; normalise to [0,1] as the human label.
            "human_score": round(r["helpfulness"] / 4.0, 3),
        })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="examples/helpsteer2_responses.json")
    args = ap.parse_args()

    responses = to_responses(fetch(args.n))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(responses, fh, indent=1, ensure_ascii=False)
    print(f"wrote {len(responses)} real responses -> {args.out}")
