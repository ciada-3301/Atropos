"""
Atropos Explorer -- Flask web app
==================================
A researcher-facing tool: paste a raw DNA sequence, watch it get embedded
and dropped into a rotatable 3D map of the reference space, see its nearest
neighbors ranked by distance, and see which taxonomic regions of the space
it landed near.

Requires the artifacts already produced by atropos_v5.py / atropos_resume.py:
    ./data/encoder.pt
    ./data/reference.index       (FAISS index, built from the 256-dim embeddings)
    ./data/taxonomy_lookup.json

ALSO requires a precomputed 3D layout:
    ./data/coords_3d.json

IMPORTANT: this app does NOT run UMAP itself, on purpose. UMAP fitting on
~200k points is slow (its default spectral initialization can hang for 30+
minutes on CPU -- see precompute_3d.py for the full explanation). Run that
script ONCE, offline, before starting this server:

    python precompute_3d.py    # one-time, several minutes
    python app.py              # instant startup every time after

If coords_3d.json is missing, this script exits immediately with a clear
message instead of silently trying to compute it inline.

NEW QUERY POINTS: rather than calling UMAP's .transform() at request time
(which is also slow, and would make every single prediction sluggish), new
sequences are placed in the 3D scene via inverse-distance-weighted
averaging of their k nearest neighbors' EXISTING 3D coordinates. This is a
standard, fast, well-understood approximation (think: "place the new point
near where its neighbors already are, weighted by how close each one is")
and runs in milliseconds since it reuses the FAISS search you're already
doing for the nearest-neighbor panel -- no separate dimensionality
reduction call needed per request.

Run with:
    python app.py
Then open http://127.0.0.1:5000
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_DIR = "./data"
ENCODER_PATH = os.path.join(DATA_DIR, "encoder.pt")
INDEX_PATH = os.path.join(DATA_DIR, "reference.index")
LOOKUP_PATH = os.path.join(DATA_DIR, "taxonomy_lookup.json")
COORDS_3D_PATH = os.path.join(DATA_DIR, "coords_3d.json")

VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "PAD": 5}
MAX_LEN = 700
EMBED_DIM = 256

# How many reference points to actually send to the browser. Sending all
# ~200k points as JSON is wasteful (the page would ship megabytes and the
# WebGL scene would have nothing to gain visually past a certain density).
# A random subsample keeps the visual structure intact while keeping
# payload size sane. Region color-coding (see build_viewport_payload)
# compensates for the reduced point density by making taxon regions legible
# even where individual points are sparse.
MAX_POINTS_FOR_VIEWPORT = 20000

TOP_K_DEFAULT = 10

# How many neighbors to use when placing a new query point in 3D space via
# weighted averaging. Doesn't need to match the user-facing match count.
PLACEMENT_K = 12


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


# ----------------------------------------------------------------------
# Load model + index + lookup + precomputed 3D layout, once at startup
# ----------------------------------------------------------------------
print("Loading encoder...")
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = DNAEncoderCNN()
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.to(device)
encoder.eval()

print("Loading FAISS index...")
import faiss
index = faiss.read_index(INDEX_PATH)
all_embeds = index.reconstruct_n(0, index.ntotal).astype("float32")
print(f"  {index.ntotal:,} reference vectors, dim={all_embeds.shape[1]}")

print("Loading taxonomy lookup...")
with open(LOOKUP_PATH) as f:
    taxonomy_lookup = json.load(f)

if not os.path.exists(COORDS_3D_PATH):
    raise SystemExit(
        f"\n{COORDS_3D_PATH} not found.\n\n"
        "This app loads a PRECOMPUTED 3D layout instead of running UMAP "
        "itself (UMAP fitting on ~200k points is too slow to do inline --\n"
        "it can hang 30+ minutes on its default spectral init).\n\n"
        "Run this once, then start app.py again:\n\n"
        "    python precompute_3d.py\n"
    )

print("Loading precomputed 3D layout...")
with open(COORDS_3D_PATH) as f:
    _cached = json.load(f)
coords_3d = np.array(_cached["coords"], dtype="float32")

if coords_3d.shape[0] != all_embeds.shape[0]:
    raise SystemExit(
        f"\ncoords_3d.json has {coords_3d.shape[0]:,} points but the FAISS "
        f"index has {all_embeds.shape[0]:,}. They're out of sync -- "
        "delete coords_3d.json and re-run precompute_3d.py.\n"
    )
print(f"  {coords_3d.shape[0]:,} cached 3D points loaded")

# Normalize 3D coords into a [-1, 1] cube so the viewport's axis gizmo and
# camera framing are predictable regardless of UMAP's raw output scale.
_coord_min = coords_3d.min(axis=0)
_coord_max = coords_3d.max(axis=0)
_coord_range = np.maximum(_coord_max - _coord_min, 1e-6)


def normalize_coords(raw_xyz):
    return ((raw_xyz - _coord_min) / _coord_range) * 2.0 - 1.0


coords_3d_norm = normalize_coords(coords_3d)


# ----------------------------------------------------------------------
# Build a per-point-subsample for the viewport, plus taxon centroid labels
# ----------------------------------------------------------------------
def build_viewport_payload():
    """
    Subsamples reference points for the browser, and computes a centroid +
    representative color per phylum so the viewport can render soft labeled
    "regions" of the space. Region labels + color-coding are what let the
    user read taxonomic structure even though only a fraction of the 200k
    reference points are actually sent to the browser.
    """
    n = coords_3d_norm.shape[0]
    if n > MAX_POINTS_FOR_VIEWPORT:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, size=MAX_POINTS_FOR_VIEWPORT, replace=False)
    else:
        sample_idx = np.arange(n)

    phyla = [taxonomy_lookup[i].get("phylum_name") or "unknown" for i in range(n)]
    unique_phyla = sorted(set(phyla))
    phylum_to_id = {p: i for i, p in enumerate(unique_phyla)}

    points = []
    for i in sample_idx:
        points.append({
            "x": float(coords_3d_norm[i, 0]),
            "y": float(coords_3d_norm[i, 1]),
            "z": float(coords_3d_norm[i, 2]),
            "phylum_id": phylum_to_id[phyla[i]],
        })

    # Centroid per phylum, computed over ALL points (not just the
    # subsample) so region labels stay accurate regardless of viewport
    # density.
    centroids = defaultdict(list)
    for i in range(n):
        centroids[phyla[i]].append(coords_3d_norm[i])

    region_labels = []
    for phylum, pts in centroids.items():
        if len(pts) < 20:
            continue  # skip tiny/noisy groups, not worth labeling
        arr = np.array(pts)
        region_labels.append({
            "phylum": phylum,
            "phylum_id": phylum_to_id[phylum],
            "x": float(arr[:, 0].mean()),
            "y": float(arr[:, 1].mean()),
            "z": float(arr[:, 2].mean()),
            "count": len(pts),
        })
    region_labels.sort(key=lambda r: -r["count"])

    return {
        "points": points,
        "regions": region_labels,
        "num_phyla": len(unique_phyla),
    }


print("Building viewport payload...")
VIEWPORT_PAYLOAD = build_viewport_payload()
print(f"  {len(VIEWPORT_PAYLOAD['points']):,} points, "
      f"{len(VIEWPORT_PAYLOAD['regions'])} labeled regions")


# ----------------------------------------------------------------------
# Query logic
# ----------------------------------------------------------------------
def place_query_point(z_query, k=PLACEMENT_K):
    """
    Places a new 256-dim query embedding into the precomputed 3D space
    WITHOUT calling UMAP. Finds its k nearest neighbors in the original
    256-dim FAISS index (already needed for the match panel, so this is
    free -- no extra search), then takes an inverse-distance-weighted
    average of those neighbors' EXISTING 3D coordinates.

    This is a standard, fast nearest-neighbor interpolation approach: the
    new point lands close to wherever its closest known relatives already
    sit in the 3D layout, weighted more toward whichever neighbors are
    most similar. It's an approximation (not a true UMAP projection of the
    new point), but it's the right tradeoff here -- it runs in
    milliseconds and keeps every prediction snappy.
    """
    sims, idxs = index.search(z_query, k)
    sims, idxs = sims[0], idxs[0]

    # similarity -> non-negative weight; clamp to avoid negative/zero
    # weights from numerical noise on near-orthogonal matches.
    weights = np.clip(sims, 1e-4, None)
    weights = weights / weights.sum()

    neighbor_coords = coords_3d_norm[idxs]  # (k, 3)
    placed_xyz = (neighbor_coords * weights[:, None]).sum(axis=0)
    return placed_xyz


def predict_taxon(raw_sequence, k=TOP_K_DEFAULT):
    raw_sequence = raw_sequence.strip().upper().replace("U", "T")
    toks = torch.tensor([tokenize(raw_sequence)]).to(device)
    with torch.no_grad():
        z = encoder(toks).cpu().numpy().astype("float32")

    sims, idxs = index.search(z, k)
    sims, idxs = sims[0], idxs[0]

    matches = []
    for sim, idx in zip(sims, idxs):
        idx = int(idx)
        tax = taxonomy_lookup[idx]
        # cosine similarity (from inner product on normalized vectors) ->
        # a [0, 2] "distance" so the UI can show a literal distance bar
        # (0 = identical, larger = further). 1 - cos_sim is the conventional
        # cosine distance; we keep it in that convention.
        distance = max(0.0, 1.0 - float(sim))
        matches.append({
            "distance": round(distance, 4),
            "similarity": round(float(sim), 4),
            "genus": tax.get("genus_name") or "(unknown)",
            "species": tax.get("species_name") or "(unknown)",
            "family": tax.get("family_name") or "(unknown)",
            "phylum": tax.get("phylum_name") or "(unknown)",
        })

    # Place the query point in the SAME 3D space as the reference cloud,
    # via kNN-weighted averaging -- no UMAP call at request time.
    query_xyz_norm = place_query_point(z)

    genus_votes = Counter()
    for m in matches:
        genus_votes[m["genus"]] += m["similarity"]
    best_genus, score = genus_votes.most_common(1)[0]
    total_sim = sum(m["similarity"] for m in matches) or 1.0
    confidence = score / total_sim

    return {
        "best_guess_genus": best_genus,
        "confidence": round(float(confidence), 3),
        "matches": matches,
        "query_position": {
            "x": float(query_xyz_norm[0]),
            "y": float(query_xyz_norm[1]),
            "z": float(query_xyz_norm[2]),
        },
    }


# ----------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/api/viewport-data")
def api_viewport_data():
    """Static reference cloud + region labels -- fetched once on page load."""
    return jsonify(VIEWPORT_PAYLOAD)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True, silent=True) or {}
    raw_seq = (body.get("sequence") or "").strip()

    if raw_seq.startswith(">"):
        # strip a FASTA header line if pasted with one
        lines = raw_seq.split("\n")
        raw_seq = "".join(lines[1:])

    cleaned = "".join(ch for ch in raw_seq.upper() if ch.isalpha())

    if len(cleaned) < 30:
        return jsonify({"error": "Sequence too short -- paste at least ~30bp of raw bases."}), 400

    k = int(body.get("k", TOP_K_DEFAULT))
    k = max(1, min(k, 50))

    try:
        result = predict_taxon(cleaned, k=k)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    print("\nAtropos Explorer running at http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)