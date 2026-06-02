import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool

# ── Config ─────────────────────────────────────────────────────────────────────

CHECKPOINT   = "../Models/GNN/best_gnn_model.pt"
SAMPLES_DIR  = "samples"          # folder of .py files named <label>_<n>.py

LABEL_TO_COMPLEXITY = {
    0: "constant", 1: "cubic",  2: "linear",
    3: "logn",     4: "nlogn",  5: "np",     6: "quadratic"
}


# ── Model ──────────────────────────────────────────────────────────────────────

class EnhancedGNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_gin_layers=5, num_classes=7, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.node_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.gin_layers  = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.gin_layers.append(GINConv(nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ), train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_gin_layers - 1):
            self.gin_layers.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ), train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_embedding(x.squeeze(1))

        edge_index = torch.cat(
            [edge_index, torch.stack([edge_index[1], edge_index[0]])], dim=1
        )

        for i, (gin, bn) in enumerate(zip(self.gin_layers, self.batch_norms)):
            x_new = F.dropout(F.relu(bn(gin(x, edge_index))),
                              p=self.dropout, training=self.training)
            x = x + x_new if i > 0 else x_new

        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch),
                       global_add_pool(x, batch)], dim=1)
        return self.classifier(x)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_samples(directory: str):
    """
    Loads all .py files from `directory`.
    Filename format: <label>_<anything>.py  e.g. constant_1.py, nlogn_v2.py
    Label is everything before the first underscore.
    """
    samples = []
    for path in sorted(Path(directory).glob("*.py")):
        label = path.stem.split("_")[0]   # e.g. "constant" from "constant_1"
        code  = path.read_text()
        samples.append({"file": path.name, "label": label, "code": code})
    return samples


def build_graph(code: str, vocab: dict):
    tree = ast.parse(code)
    nodes, edges = [], []

    def traverse(node, parent=-1):
        idx = len(nodes)
        nodes.append(type(node).__name__)
        if parent != -1:
            edges.append((parent, idx))
        for child in ast.iter_child_nodes(node):
            traverse(child, idx)

    traverse(tree)

    x = torch.tensor([vocab.get(n, 0) for n in nodes],
                     dtype=torch.long).unsqueeze(1)
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.empty((2, 0), dtype=torch.long))
    return x, edge_index


def predict(code: str, model, vocab: dict, device: torch.device):
    model.eval()
    x, edge_index = build_graph(code, vocab)

    data       = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.shape[0], dtype=torch.long)
    data       = data.to(device)

    with torch.no_grad():
        logits = model(data)
        probs  = F.softmax(logits, dim=1).squeeze()
        pred   = logits.argmax(dim=1).item()

    prob_dict = {LABEL_TO_COMPLEXITY[i]: float(probs[i])
                 for i in range(len(probs))}
    return logits, LABEL_TO_COMPLEXITY[pred], prob_dict


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    vocab = ckpt.get("node_type_vocab", {})
    if not vocab:
        raise RuntimeError("node_type_vocab missing — run patch_checkpoint.py first.")

    model = EnhancedGNNClassifier(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint  (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})\n")

    # Load samples from folder
    samples = load_samples(SAMPLES_DIR)
    print(f"Found {len(samples)} sample(s) in '{SAMPLES_DIR}/'\n")

    for i, sample in enumerate(samples, 1):
        logits, predicted, probs = predict(sample["code"], model, vocab, device)

        actual = sample["label"]
        match  = "✓" if predicted == actual else "✗"

        print(f"{'─' * 60}")
        print(f"File   : {sample['file']}")
        print(f"Actual : {actual:10s}  Predicted: {predicted:10s}  {match}")
        print(f"Logits : {logits.cpu().numpy()}")
        print("Probabilities:")
        for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(p * 40)
            tag = ""
            if cls == predicted: tag += " ← predicted"
            if cls == actual:    tag += " (actual)"
            print(f"  {cls:12s} {p:.4f}  {bar}{tag}")
        print()