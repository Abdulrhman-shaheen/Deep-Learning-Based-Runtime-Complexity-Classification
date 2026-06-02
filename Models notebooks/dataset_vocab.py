"""
patch_checkpoint.py
───────────────────
Rebuilds the node_type_vocab from the dataset and saves it
into the existing checkpoint — no retraining required.
"""

import ast
import json
import torch
from collections import Counter

CHECKPOINT  = "../Models/GNN/best_gnn_model.pt"       # path to your checkpoint
DATASET     = "../Codeforces_DataSet/code_complexity.jsonl"   # path to your dataset

# ── Rebuild vocab (same logic as training) ────────────────────────────────────

def extract_node_types(code: str):
    tree = ast.parse(code)
    return [type(n).__name__ for n in ast.walk(tree)]

print("Rebuilding vocabulary from dataset...")
all_node_types = []

with open(DATASET, "r") as f:
    for i, line in enumerate(f):
        sample = json.loads(line)
        try:
            all_node_types.extend(extract_node_types(sample["code"]))
        except SyntaxError:
            pass

unique_types    = sorted(set(all_node_types))
node_type_vocab = {nt: idx for idx, nt in enumerate(unique_types)}
print(f"Vocab size: {len(node_type_vocab)} unique node types")

# ── Patch checkpoint ──────────────────────────────────────────────────────────

ckpt = torch.load(CHECKPOINT, map_location="cpu")
ckpt["node_type_vocab"] = node_type_vocab
torch.save(ckpt, CHECKPOINT)

print(f"Checkpoint patched and saved to '{CHECKPOINT}'")
print("You can now run inference.py normally.")