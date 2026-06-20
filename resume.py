"""
Atropos v5 -- RESUME script.

Use this instead of re-running atropos_v5.py from scratch. It loads the
artifacts you already produced and saved to ./data/ (sequences.json,
encoder.pt, upproj.pt) and continues from section 8 (FAISS indexing)
onward. None of the 18-hour scan/train work is repeated.

Run with:
    python atropos_resume.py
"""

import json
import time as _time

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "PAD": 5}
MAX_LEN = 700
EMBED_DIM = 256
UPPROJ_DIM = 5000


def tokenize(seq, max_len=MAX_LEN):
    ids = [VOCAB.get(b, VOCAB["N"]) for b in seq[:max_len]]
    ids += [VOCAB["PAD"]] * (max_len - len(ids))
    return ids


class DNAEncoderCNN(nn.Module):
    def __init__(self, vocab_size=6, embed_dim=EMBED_DIM):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, 128, padding_idx=VOCAB["PAD"])
        self.conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=9, padding=4, dilation=1), nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=9, padding=8, dilation=2), nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=9, padding=16, dilation=4), nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=9, padding=4, stride=2), nn.GELU(),
        )
        self.pool_proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        h = self.token_embed(x).transpose(1, 2)
        h = self.conv(h)
        pooled = h.mean(dim=-1)
        emb = self.pool_proj(pooled)
        return F.normalize(emb, dim=-1)


class UpProjector(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, out_dim=UPPROJ_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


print("=== Loading cached artifacts (no retraining) ===")

with open("./data/sequences.json") as f:
    records = json.load(f)
print(f"  loaded {len(records):,} records from sequences.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = DNAEncoderCNN()
encoder.load_state_dict(torch.load("./data/encoder.pt", map_location=device))
encoder.to(device)
encoder.eval()
print("  loaded encoder.pt")

upproj = UpProjector()
upproj.load_state_dict(torch.load("./data/upproj.pt", map_location=device))
upproj.to(device)
upproj.eval()
print("  loaded upproj.pt")

print("Pre-tokenizing all sequences...")
_t0 = _time.time()
all_tokenized = torch.tensor([tokenize(r["sequence"]) for r in records], dtype=torch.long)
print(f"  done in {_time.time()-_t0:.1f}s")

# ----------------------------------------------------------------------
# SECTION 8: Build the searchable reference index (FAISS) + taxonomy lookup
# ----------------------------------------------------------------------
print("\n=== 8. Build the searchable reference index (FAISS) + taxonomy lookup ===")

import faiss
import numpy as np

all_embeds = []
with torch.no_grad():
    for i in range(0, len(records), 512):
        batch = all_tokenized[i:i + 512].to(device)
        z = encoder(batch).cpu().numpy()
        all_embeds.append(z)
all_embeds = np.concatenate(all_embeds, axis=0).astype("float32")

index = faiss.IndexFlatIP(EMBED_DIM)
index.add(all_embeds)
faiss.write_index(index, "./data/reference.index")

taxonomy_lookup = [
    {k: r.get(k, "") for k in
     ["phylum_name", "class_name", "order_name", "family_name", "genus_name", "species_name"]}
    for r in records
]
with open("./data/taxonomy_lookup.json", "w") as f:
    json.dump(taxonomy_lookup, f)

print(f"Index built with {index.ntotal} reference sequences.")

# ----------------------------------------------------------------------
# SECTION 9: Query function -- raw sequence -> nearest taxa + confidence
# ----------------------------------------------------------------------
print("\n=== 9. Query function: raw sequence -> nearest taxa + confidence ===")


def predict_taxon(raw_sequence, k=5):
    raw_sequence = raw_sequence.strip().upper().replace("U", "T")
    toks = torch.tensor([tokenize(raw_sequence)]).to(device)
    with torch.no_grad():
        z = encoder(toks).cpu().numpy().astype("float32")
    sims, idxs = index.search(z, k)
    sims, idxs = sims[0], idxs[0]

    results = []
    for sim, idx in zip(sims, idxs):
        tax = taxonomy_lookup[idx]
        results.append({
            "similarity": float(sim),
            "genus": tax.get("genus_name") or "(unknown)",
            "species": tax.get("species_name") or "(unknown)",
            "family": tax.get("family_name") or "(unknown)",
            "phylum": tax.get("phylum_name") or "(unknown)",
        })

    genus_votes = Counter()
    for r in results:
        genus_votes[r["genus"]] += r["similarity"]
    best_genus, score = genus_votes.most_common(1)[0]
    confidence = score / sum(r["similarity"] for r in results)

    return {
        "best_guess_genus": best_genus,
        "confidence": round(float(confidence), 3),
        "nearest_matches": results,
    }


example_seq = records[0]["sequence"]
print(predict_taxon(example_seq))

# ----------------------------------------------------------------------
# SECTION 10: 2D map of the embedding space (UMAP)
# ----------------------------------------------------------------------
print("\n=== 10. 2D map of the embedding space (UMAP) -- see evolutionary/clustering structure ===")

import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
coords_2d = reducer.fit_transform(all_embeds)

phyla = [t.get("phylum_name") or "unknown" for t in taxonomy_lookup]
unique_phyla = list(set(phyla))
color_map = {p: i for i, p in enumerate(unique_phyla)}
colors = [color_map[p] for p in phyla]

plt.figure(figsize=(12, 9))
plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, cmap="tab20", s=3, alpha=0.6)
plt.title("Atropos v5 embedding space -- colored by phylum")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.savefig("./data/embedding_map.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# SECTION 11: Simple researcher-facing UI (Gradio)
# ----------------------------------------------------------------------
print("\n=== 11. Simple researcher-facing UI (Gradio) ===")

import gradio as gr


def gradio_predict(sequence_text, top_k):
    if not sequence_text or len(sequence_text.strip()) < 30:
        return "Please paste a DNA sequence (FASTA or raw bases, at least ~30bp)."
    seq = sequence_text.strip()
    if seq.startswith(">"):
        seq = "".join(seq.split("\n")[1:])
    result = predict_taxon(seq, k=int(top_k))
    lines = [f"**Best guess genus:** {result['best_guess_genus']}  (confidence: {result['confidence']})", ""]
    lines.append("**Nearest reference matches:**")
    for m in result["nearest_matches"]:
        lines.append(f"- {m['species']} (genus: {m['genus']}, family: {m['family']}, "
                      f"phylum: {m['phylum']}) -- similarity {m['similarity']:.3f}")
    return "\n".join(lines)


demo = gr.Interface(
    fn=gradio_predict,
    inputs=[gr.Textbox(label="Paste raw eDNA sequence or FASTA", lines=6),
            gr.Slider(1, 20, value=5, step=1, label="Top-K matches")],
    outputs=gr.Markdown(label="Result"),
    title="Atropos v5 -- eDNA Taxonomic Identifier (CNN encoder)",
    description="Paste a DNA sequence to find its closest known taxonomic match based on the trained embedding model."
)
demo.launch(share=False)