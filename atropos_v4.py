"""
Atropos v4 -- eDNA Taxonomic Embedding Suite
Single-file local script version. Run top to bottom:
    python atropos_v4.py

Before running: pip install pandas faiss-cpu umap-learn gradio torch matplotlib
"""


# ----------------------------------------------------------------------
# # Atropos v4 — eDNA Taxonomic Embedding Suite
# Runs locally on your laptop against the BOLD public data dump you downloaded
# (`BOLD_Public.22-May-2026.tsv`, ~31GB uncompressed, ~23M records) — no network
# fetch, the live BOLD API endpoints were unreliable/blocked when this was built.
#
# **What this does**
# 1. Streams the 31GB local TSV in chunks (never loaded fully into memory), filters to a broad marine taxonomic scope covering essentially every major marine-associated lineage, and reservoir-samples each group so you get an unbiased, capped-where-needed subset (~200k sequences target).
# 2. Trains a small Transformer/CNN hybrid encoder from scratch with contrastive learning in a compact 256-dim space (stable, fast, good clustering).
# 3. Learns a fixed up-projection 256 → 5000 dim for downstream/storage use (matches your target dimensionality without destabilizing training).
# 4. Builds a searchable reference index (FAISS) so a researcher can paste a raw eDNA sequence and get: nearest known sequences, predicted taxon/genus, confidence, and a 2D visualization of where it lands relative to known life.
# 5. Ships a simple Gradio UI cell so this is usable without touching code.
#
# **Before running:** this notebook assumes you have a CPU-or-GPU laptop, not Colab. If you don't have an NVIDIA GPU, training cells will fall back to CPU automatically — this will be slow (the markdown notes at the bottom of this notebook tell you what to shrink if so).
# ----------------------------------------------------------------------


print('\n=== 1. Install dependencies (local run — using your downloaded BOLD dump, no network fetch) ===')
# (install once beforehand: pip install pandas faiss-cpu umap-learn gradio torch matplotlib)

print('\n=== 2. Broad marine taxonomic scope (BOLD query terms) ===')
# Goal: maximum breadth across marine-associated life, not just the "popular" groups.
# Each entry is a BOLD taxon name that will be queried independently.
# Groups marked CAP get randomly subsampled if BOLD returns more than CAP_LIMIT records;
# everything else is pulled in full (subject to the global dataset target).

CAP_LIMIT = 12000  # per-group cap for the biggest groups only

MARINE_TAXA = {
    # ---- Fish ----
    "Actinopterygii": "CAP",      # bony fish — huge, capped
    "Chondrichthyes": None,        # sharks, rays, skates
    "Myxini": None,                # hagfish
    "Petromyzontiformes": None,    # lampreys (some marine)
    "Coelacanthiformes": None,     # coelacanth

    # ---- Marine mammals & reptiles ----
    "Cetacea": None,
    "Sirenia": None,
    "Pinnipedia": None,
    "Testudines": None,            # sea turtles (subset will include freshwater, fine)
    "Squamata": "CAP",             # includes sea snakes; capped since mostly terrestrial

    # ---- Mollusca ----
    "Gastropoda": "CAP",
    "Bivalvia": None,
    "Cephalopoda": None,
    "Polyplacophora": None,        # chitons
    "Aplacophora": None,
    "Monoplacophora": None,

    # ---- Arthropoda (marine) ----
    "Malacostraca": "CAP",         # crabs, shrimp, krill, isopods, amphipods
    "Maxillopoda": None,           # copepods, barnacles (Cirripedia included here in some taxonomies)
    "Cirripedia": None,
    "Branchiopoda": None,          # mostly freshwater but some marine
    "Pycnogonida": None,           # sea spiders
    "Ostracoda": None,

    # ---- Echinodermata ----
    "Asteroidea": None,
    "Echinoidea": None,
    "Ophiuroidea": None,
    "Holothuroidea": None,
    "Crinoidea": None,

    # ---- Cnidaria ----
    "Anthozoa": None,              # corals, anemones
    "Scyphozoa": None,             # true jellyfish
    "Hydrozoa": None,
    "Cubozoa": None,               # box jellyfish
    "Staurozoa": None,

    # ---- Other invertebrates often skipped ----
    "Porifera": None,              # sponges
    "Ctenophora": None,            # comb jellies
    "Polychaeta": None,            # marine worms
    "Bryozoa": None,
    "Brachiopoda": None,
    "Chaetognatha": None,          # arrow worms
    "Sipuncula": None,
    "Nemertea": None,              # ribbon worms
    "Phoronida": None,
    "Hemichordata": None,
    "Xenacoelomorpha": None,
    "Platyhelminthes": "CAP",      # mostly parasitic/terrestrial, capped
    "Rotifera": None,
    "Tardigrada": None,
    "Entoprocta": None,
    "Placozoa": None,

    # ---- Chordata (non-vertebrate) ----
    "Ascidiacea": None,            # tunicates / sea squirts
    "Appendicularia": None,
    "Thaliacea": None,             # salps

    # ---- Protists / algae (optional but broadens "where in the tree of life") ----
    "Foraminifera": None,
    "Phaeophyceae": None,          # brown algae (kelp etc.)
    "Rhodophyta": None,            # red algae
    "Chlorophyta": "CAP",          # green algae, includes many freshwater so capped
    "Dinophyceae": None,           # dinoflagellates
    "Bacillariophyta": None,       # diatoms

    # ---- Seabirds (eDNA from coastal/marine samples often catches these) ----
    "Sphenisciformes": None,       # penguins
    "Procellariiformes": None,     # albatrosses, petrels
    "Pelecaniformes": "CAP",
    "Charadriiformes": "CAP",      # shorebirds, capped (mostly non-marine subset)
}

print(f"Total taxonomic groups queried: {len(MARINE_TAXA)}")
print(f"Capped groups: {[k for k,v in MARINE_TAXA.items() if v=='CAP']}")
TARGET_TOTAL_SEQS = 200_000

print('\n=== 3a. Point this at your local files + inspect the schema first ===')
# IMPORTANT: update these two paths to wherever you put the downloaded dump.
DATAPACKAGE_PATH = "BOLD_Public.22-May-2026.datapackage.json"
TSV_PATH = "BOLD_Public.22-May-2026.tsv"   # the 31GB file

import json as _json
with open(DATAPACKAGE_PATH) as f:
    dp = _json.load(f)

# Print the field names so we can confirm the actual column names before the
# big 31GB streaming pass below -- BOLD's dump schema has changed across
# versions (e.g. 'nuc' vs 'nucleotides', 'class' vs 'class_name'), and getting
# this wrong silently would mean a multi-hour pass that fetches nothing.
resource = dp["resources"][0]
field_names = [f["name"] for f in resource["schema"]["fields"]]
print("Columns found in datapackage schema:")
for fn in field_names:
    print(" -", fn)

print('\n=== 3b. Confirm column name mapping (EDIT THIS if your column names differ from the printed list above) ===')
# Map our internal field names -> actual column names in your TSV, based on
# what cell 3a printed. Defaults below match the current BOLD public dump
# schema as of the 2026 dumps; adjust the right-hand side only if needed.
COLMAP = {
    "sequence":     "nuc",
    "phylum_name":  "phylum",
    "class_name":   "class",
    "order_name":   "order",
    "family_name":  "family",
    "genus_name":   "genus",
    "species_name": "species",
    "bin_uri":      "bin_uri",
    "processid":    "processid",
}

missing = [v for v in COLMAP.values() if v not in field_names]
if missing:
    print(f"WARNING: these expected columns are not in the schema: {missing}")
    print("Open the printed list from cell 3a and fix COLMAP above before continuing.")
else:
    print("Column mapping OK, all expected columns found.")

print('\n=== 3c. Stream the 31GB TSV in chunks, filter to marine taxa, reservoir-sample per group ===')
# We never load the full 31GB into memory. We read it in chunks with pandas,
# and for each marine taxon group we keep a running RANDOM reservoir sample
# (so "pull everything for small groups" really means "everything, up to a
# generous safety cap" -- a few small marine phyla could still have more
# records than fit comfortably in laptop RAM, so UNCAPPED_SAFETY_LIMIT exists
# purely as a memory guard, not a deliberate downsampling choice).

import pandas as pd
import random, time

random.seed(42)

UNCAPPED_SAFETY_LIMIT = 30_000   # memory guard for "uncapped" groups, not a deliberate cap
CHUNKSIZE = 200_000               # rows per chunk read from disk

# reservoir[taxon] -> {"seen": int, "sample": [row_dict, ...]}
reservoirs = {t: {"seen": 0, "sample": []} for t in MARINE_TAXA}

def group_limit(taxon):
    return CAP_LIMIT if MARINE_TAXA[taxon] == "CAP" else UNCAPPED_SAFETY_LIMIT

def reservoir_add(taxon, row):
    res = reservoirs[taxon]
    limit = group_limit(taxon)
    res["seen"] += 1
    if len(res["sample"]) < limit:
        res["sample"].append(row)
    else:
        # classic reservoir sampling: replace with decreasing probability
        j = random.randint(0, res["seen"] - 1)
        if j < limit:
            res["sample"][j] = row

usecols = list(COLMAP.values())
rename_back = {v: k for k, v in COLMAP.items()}

start = time.time()
rows_scanned = 0
reader = pd.read_csv(TSV_PATH, sep="\t", usecols=usecols, chunksize=CHUNKSIZE,
                     dtype=str, on_bad_lines="skip", low_memory=False)

for chunk in reader:
    chunk = chunk.rename(columns=rename_back)
    rows_scanned += len(chunk)

    # a record can match more than one MARINE_TAXA entry across different
    # taxonomic ranks (e.g. phylum AND order both listed) -- match on ANY rank
    for taxon in MARINE_TAXA:
        mask = (
            (chunk["phylum_name"] == taxon) | (chunk["class_name"] == taxon) |
            (chunk["order_name"] == taxon)  | (chunk["family_name"] == taxon) |
            (chunk["genus_name"] == taxon)
        )
        if not mask.any():
            continue
        sub = chunk.loc[mask]
        for _, r in sub.iterrows():
            seq = str(r.get("sequence", "")).strip().upper()
            if not seq or seq == "NAN" or len(seq) < 200:
                continue
            row = {
                "sequence": seq,
                "phylum_name": r.get("phylum_name", ""),
                "class_name": r.get("class_name", ""),
                "order_name": r.get("order_name", ""),
                "family_name": r.get("family_name", ""),
                "genus_name": r.get("genus_name", ""),
                "species_name": r.get("species_name", ""),
                "bin_uri": r.get("bin_uri", ""),
                "processid": r.get("processid", ""),
            }
            reservoir_add(taxon, row)

    if rows_scanned % 2_000_000 < CHUNKSIZE:
        elapsed = time.time() - start
        print(f"  scanned {rows_scanned:,} rows so far ({elapsed/60:.1f} min)...")

print(f"\nDone scanning. Total rows scanned: {rows_scanned:,} in {(time.time()-start)/60:.1f} min")
for taxon, res in reservoirs.items():
    print(f"  {taxon}: matched {res['seen']:,} rows, kept {len(res['sample']):,}")

print('\n=== 3d. Merge reservoirs, dedupe, trim to global target ===')
all_records = []
for taxon, res in reservoirs.items():
    all_records.extend(res["sample"])

print(f"Total records before dedup: {len(all_records):,}")

seen_seqs = set()
deduped = []
for r in all_records:
    if r["sequence"] not in seen_seqs:
        seen_seqs.add(r["sequence"])
        deduped.append(r)

if len(deduped) > TARGET_TOTAL_SEQS:
    deduped = random.sample(deduped, TARGET_TOTAL_SEQS)

print(f"Final dataset size: {len(deduped):,}")

import os, json
os.makedirs("./data", exist_ok=True)
with open("./data/sequences.json", "w") as f:
    json.dump(deduped, f)

print('\n=== 4. Tokenizer + dataset ===')
import torch
from torch.utils.data import Dataset
import json, random

with open("./data/sequences.json") as f:
    records = json.load(f)

VOCAB = {"A":0,"C":1,"G":2,"T":3,"N":4,"PAD":5}
MAX_LEN = 700  # COI barcodes are ~650bp; pad/truncate to this

def tokenize(seq, max_len=MAX_LEN):
    ids = [VOCAB.get(b, VOCAB["N"]) for b in seq[:max_len]]
    ids += [VOCAB["PAD"]] * (max_len - len(ids))
    return ids

# group records by genus for contrastive sampling (positives = same genus)
from collections import defaultdict
by_genus = defaultdict(list)
for i, r in enumerate(records):
    key = r.get("genus_name") or r.get("family_name") or "unknown"
    by_genus[key].append(i)

# drop singleton groups (no positive pair possible) for training, keep for index later
trainable_genera = {k: v for k, v in by_genus.items() if len(v) >= 2}
print(f"Genera/groups usable for contrastive training: {len(trainable_genera)}")

class ContrastivePairDataset(Dataset):
    def __init__(self, records, groups):
        self.records = records
        self.group_keys = list(groups.keys())
        self.groups = groups

    def __len__(self):
        return len(self.group_keys) * 4  # oversample epochs a bit

    def __getitem__(self, idx):
        key = random.choice(self.group_keys)
        idxs = self.groups[key]
        i, j = random.sample(idxs, 2) if len(idxs) >= 2 else (idxs[0], idxs[0])
        seq_a = tokenize(self.records[i]["sequence"])
        seq_b = tokenize(self.records[j]["sequence"])
        return torch.tensor(seq_a), torch.tensor(seq_b)

train_ds = ContrastivePairDataset(records, trainable_genera)
print(f"Training pairs per epoch: {len(train_ds)}")

print('\n=== 5. Model: small CNN+Transformer encoder (256-dim) + learned up-projection (5000-dim) ===')
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 256
UPPROJ_DIM = 5000

class DNAEncoder(nn.Module):
    def __init__(self, vocab_size=6, embed_dim=EMBED_DIM, max_len=MAX_LEN):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, 128, padding_idx=VOCAB["PAD"])
        self.conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=9, padding=4), nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=9, padding=4, stride=2), nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.pool_proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        h = self.token_embed(x)              # (B, L, 128)
        h = h.transpose(1, 2)                # (B, 128, L)
        h = self.conv(h)                     # (B, 256, L/2)
        h = h.transpose(1, 2)                # (B, L/2, 256)
        h = self.transformer(h)              # (B, L/2, 256)
        pooled = h.mean(dim=1)               # mean pool
        emb = self.pool_proj(pooled)         # (B, 256)
        return F.normalize(emb, dim=-1)

class UpProjector(nn.Module):
    # Fixed/learned linear map from the trained 256-dim space to 5000-dim.
    # Trained AFTER the encoder is frozen, just to preserve neighborhood structure
    # in a higher-dim space for downstream storage/compatibility.
    def __init__(self, in_dim=EMBED_DIM, out_dim=UPPROJ_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

encoder = DNAEncoder().cuda() if torch.cuda.is_available() else DNAEncoder()
print(encoder)

print('\n=== 6. Contrastive training loop (InfoNCE / SimCLR-style, in 256-dim space) ===')
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder.to(device)

EPOCHS = 8
BATCH_SIZE = 256
LR = 3e-4
TEMP = 0.07

loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)

def info_nce_loss(z_a, z_b, temp=TEMP):
    B = z_a.size(0)
    z = torch.cat([z_a, z_b], dim=0)              # (2B, D)
    sim = z @ z.T / temp                          # (2B, 2B)
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets])   # positive of i is i+B and vice versa
    return F.cross_entropy(sim, targets)

encoder.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for seq_a, seq_b in loader:
        seq_a, seq_b = seq_a.to(device), seq_b.to(device)
        z_a, z_b = encoder(seq_a), encoder(seq_b)
        loss = info_nce_loss(z_a, z_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/len(loader):.4f}")

torch.save(encoder.state_dict(), "./data/encoder.pt")

print('\n=== 7. Train the up-projection (256 -> 5000) to preserve neighborhood structure ===')
encoder.eval()
upproj = UpProjector().to(device)
up_optimizer = torch.optim.AdamW(upproj.parameters(), lr=1e-3)

# Self-supervised target: preserve pairwise cosine similarities of the 256-dim space
# inside the 5000-dim space (after L2-normalizing the projection output too).
UP_EPOCHS = 5
with torch.no_grad():
    all_tokens = torch.tensor([tokenize(r["sequence"]) for r in records])

up_loader = DataLoader(torch.arange(len(records)), batch_size=256, shuffle=True)

for epoch in range(UP_EPOCHS):
    total = 0.0
    for idx_batch in up_loader:
        toks = all_tokens[idx_batch].to(device)
        with torch.no_grad():
            z = encoder(toks)                      # (B, 256), already normalized
        z_up = F.normalize(upproj(z), dim=-1)       # (B, 5000)
        sim_low = z @ z.T
        sim_high = z_up @ z_up.T
        loss = F.mse_loss(sim_high, sim_low)
        up_optimizer.zero_grad()
        loss.backward()
        up_optimizer.step()
        total += loss.item()
    print(f"Up-proj epoch {epoch+1}/{UP_EPOCHS} - sim MSE: {total/len(up_loader):.6f}")

torch.save(upproj.state_dict(), "./data/upproj.pt")

print('\n=== 8. Build the searchable reference index (FAISS) + taxonomy lookup ===')
import faiss
import numpy as np

encoder.eval()
all_embeds = []
with torch.no_grad():
    for i in range(0, len(records), 512):
        batch = all_tokens[i:i+512].to(device)
        z = encoder(batch).cpu().numpy()
        all_embeds.append(z)
all_embeds = np.concatenate(all_embeds, axis=0).astype("float32")

index = faiss.IndexFlatIP(EMBED_DIM)  # cosine sim via inner product (vectors are normalized)
index.add(all_embeds)
faiss.write_index(index, "./data/reference.index")

taxonomy_lookup = [
    {k: r.get(k, "") for k in
     ["phylum_name","class_name","order_name","family_name","genus_name","species_name"]}
    for r in records
]
with open("./data/taxonomy_lookup.json", "w") as f:
    json.dump(taxonomy_lookup, f)

print(f"Index built with {index.ntotal} reference sequences.")

print('\n=== 9. Query function: raw sequence -> nearest taxa + confidence ===')
def predict_taxon(raw_sequence, k=5):
    raw_sequence = raw_sequence.strip().upper().replace("U","T")
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

    # simple consensus call: majority genus among top-k, weighted by similarity
    from collections import Counter
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

# quick smoke test (replace with a real raw eDNA read)
example_seq = records[0]["sequence"]
print(predict_taxon(example_seq))

print('\n=== 10. 2D map of the embedding space (UMAP) — see evolutionary/clustering structure ===')
import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
coords_2d = reducer.fit_transform(all_embeds)

phyla = [t.get("phylum_name") or "unknown" for t in taxonomy_lookup]
unique_phyla = list(set(phyla))
color_map = {p: i for i, p in enumerate(unique_phyla)}
colors = [color_map[p] for p in phyla]

plt.figure(figsize=(12, 9))
sc = plt.scatter(coords_2d[:,0], coords_2d[:,1], c=colors, cmap="tab20", s=3, alpha=0.6)
plt.title("Atropos v4 embedding space — colored by phylum")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.savefig("./data/embedding_map.png", dpi=150, bbox_inches="tight")
plt.show()

print('\n=== 11. Simple researcher-facing UI (Gradio) ===')
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
        lines.append(f"- {m['species']} (genus: {m['genus']}, family: {m['family']}, phylum: {m['phylum']}) — similarity {m['similarity']:.3f}")
    return "\n".join(lines)

demo = gr.Interface(
    fn=gradio_predict,
    inputs=[gr.Textbox(label="Paste raw eDNA sequence or FASTA", lines=6),
            gr.Slider(1, 20, value=5, step=1, label="Top-K matches")],
    outputs=gr.Markdown(label="Result"),
    title="Atropos v4 — eDNA Taxonomic Identifier",
    description="Paste a DNA sequence to find its closest known taxonomic match based on the trained embedding model."
)
demo.launch(share=False)  # local run -- opens http://127.0.0.1:7860, no public tunnel needed

# ----------------------------------------------------------------------
# ## Notes
# - **Why 256-dim for training, 5000-dim for storage:** contrastive losses degrade in very high-dim spaces (curse of dimensionality on cosine similarity), so the model learns a tight, well-separated 256-dim space, then a learned linear map preserves that structure at 5000-dim for downstream compatibility — gets you both stability and your target size.
# - **Taxonomic breadth:** the `MARINE_TAXA` dict in cell 2 is the single place controlling scope — add/remove taxon names there to widen or narrow coverage; matching happens against phylum/class/order/family/genus columns in your local TSV during the streaming pass.
# - **If you're on CPU only (no NVIDIA GPU):** in cell 6, drop `EPOCHS` to 3-4 and `BATCH_SIZE` to 64-128; in cell 4, consider sub-sampling `TARGET_TOTAL_SEQS` down to ~50k for a first end-to-end test run before committing to the full 200k. The streaming/filtering step (cell 3c) is CPU/disk-bound either way and will take a while on 31GB — that's expected, it's a one-time cost; the filtered ~200k-sequence subset gets cached to `./data/sequences.json` so you don't need to re-scan the TSV on subsequent runs.
# - **Scaling up later:** if this works well, the natural next step (off-laptop) is a larger transformer pretrained masked-language-model-style on raw sequence first, then fine-tuned with this same contrastive scheme — that's closer to genuine "DNA foundation model" semantics (e.g. JEPA-style) but needs a GPU cluster, not a laptop.
# ----------------------------------------------------------------------