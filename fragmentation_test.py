"""
Atropos -- Fragmentation robustness test
==========================================
Truncates known reference sequences to a range of lengths and checks where
prediction quality actually falls off, instead of guessing.

For each test sequence, this:
  1. Picks the FULL sequence and confirms it self-matches with distance ~0.
  2. Truncates it to a range of lengths (600, 500, 400, 300, 200, 150, 100,
     75, 50, 30 bp) and re-runs prediction on each fragment.
  3. Prints, for every length, whether the top match is still the CORRECT
     genus, what the distance/confidence look like, and flags the point
     where it stops being reliable.

This uses your already-trained encoder + FAISS index -- no retraining, no
re-scanning the TSV. Takes well under a minute.

Run with:
    python test_fragmentation.py
"""

import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "PAD": 5}
MAX_LEN = 700
EMBED_DIM = 256

# Lengths to test, from full-length down to very short fragments.
TEST_LENGTHS = [650, 600, 500, 400, 300, 200, 150, 100, 75, 50, 30]

# How many different reference sequences to test this against (picked from
# different taxa so the result isn't a fluke of one easy/hard case).
NUM_TEST_SEQUENCES = 5


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


print("Loading model + index + lookup...")
device = "cpu"

encoder = DNAEncoderCNN()
encoder.load_state_dict(torch.load("./data/encoder.pt", map_location=device))
encoder.to(device)
encoder.eval()

import faiss
index = faiss.read_index("./data/reference.index")

with open("./data/taxonomy_lookup.json") as f:
    taxonomy_lookup = json.load(f)

with open("./data/sequences.json") as f:
    records = json.load(f)

print(f"  {index.ntotal:,} reference vectors loaded\n")


def predict_top(raw_sequence, k=5):
    raw_sequence = raw_sequence.strip().upper().replace("U", "T")
    toks = torch.tensor([tokenize(raw_sequence)]).to(device)
    with torch.no_grad():
        z = encoder(toks).cpu().numpy().astype("float32")
    sims, idxs = index.search(z, k)
    sim, idx = float(sims[0][0]), int(idxs[0][0])
    tax = taxonomy_lookup[idx]
    distance = max(0.0, 1.0 - sim)
    return {
        "distance": distance,
        "similarity": sim,
        "genus": tax.get("genus_name") or "(unknown)",
        "species": tax.get("species_name") or "(unknown)",
    }


# Pick a handful of test sequences from different taxa, long enough to
# actually test the full TEST_LENGTHS range. ALSO filter out records with
# missing/NaN genus labels -- without this, a "test case" can have no real
# genus to match against at all, which makes every length look like a
# failure (including the untouched full-length sequence) for reasons that
# have nothing to do with fragmentation. That's a data-quality artifact,
# not a model result, so it has to be excluded before sampling.
random.seed(7)


def has_real_genus(r):
    g = r.get("genus_name")
    if g is None:
        return False
    g = str(g).strip()
    return g != "" and g.lower() != "nan"


candidates = [
    r for r in records
    if len(r["sequence"]) >= max(TEST_LENGTHS) and has_real_genus(r)
]
print(f"{len(candidates):,} reference sequences are long enough (>= {max(TEST_LENGTHS)}bp) "
      f"AND have a real genus label, usable as test cases.\n")

test_records = random.sample(candidates, min(NUM_TEST_SEQUENCES, len(candidates)))

print("=" * 78)
print("FRAGMENTATION TEST")
print("=" * 78)

for rec_i, rec in enumerate(test_records, 1):
    true_genus = rec.get("genus_name") or "(unknown)"
    true_species = rec.get("species_name") or "(unknown)"
    full_seq = rec["sequence"]

    print(f"\n[{rec_i}/{len(test_records)}] TRUE TAXON: {true_species}  (genus: {true_genus})")
    print(f"  full sequence length: {len(full_seq)}bp")
    print(f"  {'length':>8}  {'top match':<28} {'genus OK?':<10} {'distance':>9}  {'similarity':>10}")
    print(f"  {'-'*8}  {'-'*28} {'-'*10} {'-'*9}  {'-'*10}")

    first_failure_length = None

    for length in TEST_LENGTHS:
        fragment = full_seq[:length]
        result = predict_top(fragment)

        genus_ok = result["genus"] == true_genus
        flag = "  OK" if genus_ok else "  WRONG"

        if not genus_ok and first_failure_length is None:
            first_failure_length = length

        match_label = result["species"]
        print(f"  {length:>6}bp  {match_label:<28} {flag:<10} {result['distance']:>9.4f}  {result['similarity']:>10.4f}")

    if first_failure_length is not None:
        print(f"  --> genus-level prediction broke at or below {first_failure_length}bp")
    else:
        print(f"  --> genus-level prediction held correct down to the shortest tested length "
              f"({min(TEST_LENGTHS)}bp)")

print("\n" + "=" * 78)
print("Done. Read the per-sequence tables above -- the length where 'genus OK?'")
print("first flips to WRONG is roughly where this encoder's effective floor is,")
print("for THIS architecture (mean-pooling over up to 700 tokens). This is a")
print("clean-fragment test (no simulated sequencing errors/damage) -- real")
print("degraded DNA would likely break down at longer lengths than shown here,")
print("since this only tests pure truncation, not base-level damage.")
print("=" * 78)