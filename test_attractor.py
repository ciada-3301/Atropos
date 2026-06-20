"""
Atropos -- Attractor-point test
=================================
Checks whether very short, heavily-padded fragments collapse onto a small
set of repeat "attractor" answers regardless of true taxon -- as opposed to
genuinely varied (if wrong) predictions.

Method: take MANY different reference sequences (broad taxonomic spread),
truncate each to a single SHORT length (default 100bp, matching where the
fragmentation test showed trouble), predict, and look at the DISTRIBUTION
of top-match answers. 

  - If predictions are spread across many different species/genera (even
    if individually wrong), that's "confused but trying" -- consistent
    with real signal, just not enough of it at this length.
  - If a tiny handful of species/genera show up as the top match for a
    disproportionate fraction of totally unrelated queries, that's an
    "attractor" -- a structural artifact of how short+padded inputs get
    embedded, not a biological judgment at all.

Run with:
    python test_attractors.py
"""

import json
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "PAD": 5}
MAX_LEN = 700
EMBED_DIM = 256

# The short length to stress-test -- pick something in the range where the
# fragmentation test showed trouble (below ~150-200bp).
FRAGMENT_LENGTH = 100

# How many different reference sequences to sample and truncate. Larger =
# more statistically convincing, but each one is a fast forward pass so
# this is cheap to raise.
NUM_QUERIES = 300


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


def has_real_genus(r):
    g = r.get("genus_name")
    if g is None:
        return False
    g = str(g).strip()
    return g != "" and g.lower() != "nan"


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


def predict_top(raw_sequence, k=1):
    raw_sequence = raw_sequence.strip().upper().replace("U", "T")
    toks = torch.tensor([tokenize(raw_sequence)]).to(device)
    with torch.no_grad():
        z = encoder(toks).cpu().numpy().astype("float32")
    sims, idxs = index.search(z, k)
    sim, idx = float(sims[0][0]), int(idxs[0][0])
    tax = taxonomy_lookup[idx]
    return {
        "similarity": sim,
        "genus": tax.get("genus_name") or "(unknown)",
        "species": tax.get("species_name") or "(unknown)",
    }


random.seed(11)
candidates = [
    r for r in records
    if len(r["sequence"]) >= FRAGMENT_LENGTH and has_real_genus(r)
]
print(f"{len(candidates):,} candidates available, sampling {NUM_QUERIES} broadly across taxa.\n")

queries = random.sample(candidates, min(NUM_QUERIES, len(candidates)))

print(f"Truncating each to {FRAGMENT_LENGTH}bp and predicting top-1 match...")
top_match_counter = Counter()
true_genus_counter = Counter()
correct = 0
similarities = []

for rec in queries:
    fragment = rec["sequence"][:FRAGMENT_LENGTH]
    result = predict_top(fragment)
    top_match_counter[result["species"]] += 1
    true_genus_counter[rec.get("genus_name")] += 1
    similarities.append(result["similarity"])
    if result["genus"] == rec.get("genus_name"):
        correct += 1

print("\n" + "=" * 78)
print(f"RESULTS  (fragment length = {FRAGMENT_LENGTH}bp, {len(queries)} queries from "
      f"{len(true_genus_counter)} distinct TRUE genera)")
print("=" * 78)

print(f"\nGenus-level accuracy at {FRAGMENT_LENGTH}bp: {correct}/{len(queries)} "
      f"({100*correct/len(queries):.1f}%)")
print(f"Mean top-1 similarity: {sum(similarities)/len(similarities):.4f}")

print(f"\nTrue queries were spread across {len(true_genus_counter)} distinct genera "
      f"(input diversity -- expect predictions to be similarly spread if the\n"
      f"model is responding to real signal rather than collapsing).")

print(f"\nTop-1 PREDICTED species, ranked by how often each was returned:")
print(f"{'species':<35} {'times returned':>15} {'% of all queries':>18}")
print(f"{'-'*35} {'-'*15} {'-'*18}")
most_common = top_match_counter.most_common(15)
for species, count in most_common:
    pct = 100 * count / len(queries)
    flag = "  <-- ATTRACTOR" if pct > (100 / len(true_genus_counter)) * 5 else ""
    print(f"{species:<35} {count:>15} {pct:>17.1f}%{flag}")

n_unique_predictions = len(top_match_counter)
print(f"\n{n_unique_predictions} distinct species appeared as the top-1 prediction "
      f"across {len(queries)} queries.")

top1_count = most_common[0][1] if most_common else 0
top1_share = 100 * top1_count / len(queries)

print("\n" + "-" * 78)
if top1_share > 15:
    print(f"DIAGNOSIS: ATTRACTOR PATTERN CONFIRMED.")
    print(f"  '{most_common[0][0]}' alone accounts for {top1_share:.1f}% of ALL top-1")
    print(f"  predictions across {len(true_genus_counter)} different true genera. A correctly")
    print(f"  functioning model should not return the same answer this often for")
    print(f"  this many genuinely different inputs -- this points to a structural")
    print(f"  embedding collapse at this fragment length, not biological confusion.")
elif n_unique_predictions < len(queries) * 0.3:
    print(f"DIAGNOSIS: PARTIAL COLLAPSE.")
    print(f"  Only {n_unique_predictions} distinct answers across {len(queries)} queries")
    print(f"  ({100*n_unique_predictions/len(queries):.1f}% unique) -- predictions are clustering")
    print(f"  onto a smaller set of outcomes than the input diversity would suggest.")
else:
    print(f"DIAGNOSIS: NO STRONG ATTRACTOR DETECTED at {FRAGMENT_LENGTH}bp.")
    print(f"  Predictions are reasonably spread ({n_unique_predictions} distinct answers across")
    print(f"  {len(queries)} queries) -- wrong answers here are more likely genuine")
    print(f"  (if mistaken) attempts rather than a structural collapse.")
print("-" * 78)