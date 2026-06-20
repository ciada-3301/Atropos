"""
Atropos v5 -- eDNA Taxonomic Embedding Suite (CNN-only, CPU-optimized)
Single-file local script version. Run top to bottom:
    python atropos_v5.py

Before running: pip install pandas faiss-cpu umap-learn gradio torch matplotlib

=== CHANGES FROM v4 ===
1. Transformer encoder replaced with a dilated CNN encoder (no attention).
   On CPU, self-attention's O(L^2) cost and small-op overhead is a poor fit
   for hybrid P/E-core laptop chips. Dilated convs are fully parallel across
   the sequence dimension and give a large receptive field cheaply -- a much
   better fit for barcode-length (~650bp) sequences and this hardware.
2. torch.set_num_threads() / set_num_interop_threads() set explicitly at
   startup. PyTorch's default thread count on hybrid-core CPUs (e.g. Intel
   13th-gen U-series: 2 P-cores + 8 E-cores) tends to oversubscribe and hurt
   throughput -- worth benchmarking, see BENCH_THREADS below.
3. Sequences are tokenized ONCE up front into a single tensor, instead of
   inside Dataset.__getitem__ on every access. This removes repeated Python
   string indexing / dict lookups from the training hot loop.
4. DataLoader uses num_workers>0 + persistent_workers=True now that workers
   only do cheap tensor slicing (not string processing).
5. Batch size raised 64 -> 128 (CPU throughput is usually better with larger
   batches: more cache reuse, fewer Python-loop overheads per sample).
6. Memory fixes during the 31GB streaming pass (section 3c):
     - explicit dtype on read_csv kept narrow (str) -- unchanged, already
       memory-safe -- but chunk objects are now deleted explicitly each loop
       iteration and gc.collect() is called periodically, since long-running
       pandas chunked reads on Windows can let chunk fragments linger before
       the GC runs naturally.
     - sequence dedup in 3d now uses a generator instead of building two full
       lists in memory at once.
7. A BENCH_THREADS toggle is added near the top -- when True, the script
   times a few warmup batches at thread counts [2,4,6,8] and prints results
   before committing to training, so you can pick the fastest setting for
   your exact machine instead of guessing.
"""

import os
import gc
import json
import time
import random
import time as _time
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------
# WINDOWS MULTIPROCESSING FIX (vs the version that crashed):
# ContrastivePairDataset is defined here, unconditionally at module level,
# because DataLoader(..., num_workers=2) in Section 6 pickles instances of
# it to hand off to worker processes. On Windows (no fork()), workers are
# created by *re-importing this file* with __name__ == "__mp_main__". If
# this class were defined only inside the `if __name__ == "__main__":`
# guard below, the re-import would skip it (since __name__ != "__main__"
# in the worker), and unpickling the dataset in the worker would fail.
# Everything else -- all the actual script logic, side effects, and the
# DataLoader/training calls themselves -- lives inside the guard, so that
# re-importing this file in a worker process does not also try to spawn
# *more* workers (which is what caused the original RuntimeError).
# ----------------------------------------------------------------------

class ContrastivePairDataset(Dataset):
    """Returns index pairs into the pre-tokenized tensor (no string work here)."""

    def __init__(self, all_tokenized, groups):
        self.all_tokenized = all_tokenized
        self.group_keys = list(groups.keys())
        self.groups = groups

    def __len__(self):
        return len(self.group_keys) * 4  # oversample epochs a bit

    def __getitem__(self, idx):
        key = random.choice(self.group_keys)
        idxs = self.groups[key]
        i, j = random.sample(idxs, 2) if len(idxs) >= 2 else (idxs[0], idxs[0])
        return self.all_tokenized[i], self.all_tokenized[j]

if __name__ == "__main__":

    # ----------------------------------------------------------------------
    # SECTION 0: CPU thread tuning (do this BEFORE any heavy torch ops)
    # ----------------------------------------------------------------------
    # Default PyTorch thread settings often oversubscribe on hybrid P/E-core
    # CPUs (e.g. 13th-gen Intel U-series: 2 P-cores + 8 E-cores). Benchmark a
    # few options below; 4 is a reasonable starting guess but YOUR machine may
    # differ -- set BENCH_THREADS = True once to find the best value, then hardcode it.
    BENCH_THREADS = False
    DEFAULT_NUM_THREADS = 4
    DEFAULT_INTEROP_THREADS = 1

    torch.set_num_threads(DEFAULT_NUM_THREADS)
    torch.set_num_interop_threads(DEFAULT_INTEROP_THREADS)

    print(f"\n=== 0. Thread config: intra-op={torch.get_num_threads()}, "
          f"inter-op={torch.get_num_interop_threads()} ===")

    print('\n=== 1. Install dependencies (local run -- using your downloaded BOLD dump, no network fetch) ===')
    # (install once beforehand: pip install pandas faiss-cpu umap-learn gradio torch matplotlib)

    # ----------------------------------------------------------------------
    # SECTION 2: Broad marine taxonomic scope (BOLD query terms)
    # ----------------------------------------------------------------------
    print('\n=== 2. Broad marine taxonomic scope (BOLD query terms) ===')

    CAP_LIMIT = 12000  # per-group cap for the biggest groups only

    MARINE_TAXA = {
        # ---- Fish ----
        "Actinopterygii": "CAP",
        "Chondrichthyes": None,
        "Myxini": None,
        "Petromyzontiformes": None,
        "Coelacanthiformes": None,

        # ---- Marine mammals & reptiles ----
        "Cetacea": None,
        "Sirenia": None,
        "Pinnipedia": None,
        "Testudines": None,
        "Squamata": "CAP",

        # ---- Mollusca ----
        "Gastropoda": "CAP",
        "Bivalvia": None,
        "Cephalopoda": None,
        "Polyplacophora": None,
        "Aplacophora": None,
        "Monoplacophora": None,

        # ---- Arthropoda (marine) ----
        "Malacostraca": "CAP",
        "Maxillopoda": None,
        "Cirripedia": None,
        "Branchiopoda": None,
        "Pycnogonida": None,
        "Ostracoda": None,

        # ---- Echinodermata ----
        "Asteroidea": None,
        "Echinoidea": None,
        "Ophiuroidea": None,
        "Holothuroidea": None,
        "Crinoidea": None,

        # ---- Cnidaria ----
        "Anthozoa": None,
        "Scyphozoa": None,
        "Hydrozoa": None,
        "Cubozoa": None,
        "Staurozoa": None,

        # ---- Other invertebrates often skipped ----
        "Porifera": None,
        "Ctenophora": None,
        "Polychaeta": None,
        "Bryozoa": None,
        "Brachiopoda": None,
        "Chaetognatha": None,
        "Sipuncula": None,
        "Nemertea": None,
        "Phoronida": None,
        "Hemichordata": None,
        "Xenacoelomorpha": None,
        "Platyhelminthes": "CAP",
        "Rotifera": None,
        "Tardigrada": None,
        "Entoprocta": None,
        "Placozoa": None,

        # ---- Chordata (non-vertebrate) ----
        "Ascidiacea": None,
        "Appendicularia": None,
        "Thaliacea": None,

        # ---- Protists / algae ----
        "Foraminifera": None,
        "Phaeophyceae": None,
        "Rhodophyta": None,
        "Chlorophyta": "CAP",
        "Dinophyceae": None,
        "Bacillariophyta": None,

        # ---- Seabirds ----
        "Sphenisciformes": None,
        "Procellariiformes": None,
        "Pelecaniformes": "CAP",
        "Charadriiformes": "CAP",
    }

    print(f"Total taxonomic groups queried: {len(MARINE_TAXA)}")
    print(f"Capped groups: {[k for k, v in MARINE_TAXA.items() if v == 'CAP']}")
    TARGET_TOTAL_SEQS = 200_000

    # ----------------------------------------------------------------------
    # SECTION 3a: Point this at your local files + inspect the schema first
    # ----------------------------------------------------------------------
    print('\n=== 3a. Point this at your local files + inspect the schema first ===')
    DATAPACKAGE_PATH = "data/BOLD_Public.22-May-2026.datapackage.json"
    TSV_PATH = "data/BOLD_Public.22-May-2026.tsv"   # the 31GB file

    with open(DATAPACKAGE_PATH) as f:
        dp = json.load(f)

    resource = dp["resources"][0]
    field_names = [f["name"] for f in resource["schema"]["fields"]]
    print("Columns found in datapackage schema:")
    for fn in field_names:
        print(" -", fn)

    # ----------------------------------------------------------------------
    # SECTION 3b: Confirm column name mapping
    # ----------------------------------------------------------------------
    print('\n=== 3b. Confirm column name mapping (EDIT THIS if your column names differ from the printed list above) ===')
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
        print("Open the printed list from section 3a and fix COLMAP above before continuing.")
    else:
        print("Column mapping OK, all expected columns found.")

    # ----------------------------------------------------------------------
    # SECTION 3c: Stream the 31GB TSV in chunks, filter, reservoir-sample
    # ----------------------------------------------------------------------
    print('\n=== 3c. Stream the 31GB TSV in chunks, filter to marine taxa, reservoir-sample per group ===')

    import pandas as pd

    def clean_tax(v):
        """Normalize a taxonomy field so a missing value can never come back
        as truthy NaN later on. Missing TSV cells arrive from pandas as
        float('nan') even when read with dtype=str -- and `x or fallback`
        does NOT fall through for it, because bool(float('nan')) is True.
        NaN is also unsafe as a dict key once it crosses a pickle boundary
        (e.g. DataLoader workers): NaN != NaN, and CPython's pickler doesn't
        preserve float object identity, so a key that matched before
        pickling can silently stop matching after -- this is what caused
        the `KeyError: nan` in the contrastive-pair dataset. Always reduce
        to a clean string (possibly empty) instead of ever passing NaN on."""
        if v is None:
            return ""
        if isinstance(v, float):  # NaN, including ones round-tripped through JSON
            return ""
        v = str(v).strip()
        return "" if v.lower() == "nan" else v

    random.seed(42)

    UNCAPPED_SAFETY_LIMIT = 30_000   # memory guard for "uncapped" groups, not a deliberate cap
    CHUNKSIZE = 200_000               # rows per chunk read from disk
    GC_EVERY_N_CHUNKS = 10            # periodic gc.collect() during the long streaming pass

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
            j = random.randint(0, res["seen"] - 1)
            if j < limit:
                res["sample"][j] = row


    usecols = list(COLMAP.values())
    rename_back = {v: k for k, v in COLMAP.items()}

    start = time.time()
    rows_scanned = 0
    reader = pd.read_csv(TSV_PATH, sep="\t", usecols=usecols, chunksize=CHUNKSIZE,
                          dtype=str, on_bad_lines="skip", low_memory=False)

    for chunk_idx, chunk in enumerate(reader):
        chunk = chunk.rename(columns=rename_back)
        rows_scanned += len(chunk)

        for taxon in MARINE_TAXA:
            mask = (
                (chunk["phylum_name"] == taxon) | (chunk["class_name"] == taxon) |
                (chunk["order_name"] == taxon) | (chunk["family_name"] == taxon) |
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
                    "phylum_name": clean_tax(r.get("phylum_name", "")),
                    "class_name": clean_tax(r.get("class_name", "")),
                    "order_name": clean_tax(r.get("order_name", "")),
                    "family_name": clean_tax(r.get("family_name", "")),
                    "genus_name": clean_tax(r.get("genus_name", "")),
                    "species_name": clean_tax(r.get("species_name", "")),
                    "bin_uri": clean_tax(r.get("bin_uri", "")),
                    "processid": clean_tax(r.get("processid", "")),
                }
                reservoir_add(taxon, row)

        # MEMORY FIX: explicitly drop the chunk reference before the next
        # iteration, and periodically force a gc pass. On long Windows runs
        # over a 31GB file, letting pandas chunk fragments pile up before a
        # natural gc cycle can bloat working-set memory noticeably.
        # NOTE: `sub`/`mask` are inner-loop locals and are NOT guaranteed to
        # exist here (a chunk where every taxon's mask.any() is False never
        # assigns them), so only `chunk` is safe to del at this scope -- `sub`
        # and `mask` get garbage collected naturally when overwritten next chunk.
        del chunk
        if chunk_idx % GC_EVERY_N_CHUNKS == 0:
            gc.collect()

        if rows_scanned % 2_000_000 < CHUNKSIZE:
            elapsed = time.time() - start
            print(f"  scanned {rows_scanned:,} rows so far ({elapsed/60:.1f} min)...")

    print(f"\nDone scanning. Total rows scanned: {rows_scanned:,} in {(time.time()-start)/60:.1f} min")
    for taxon, res in reservoirs.items():
        print(f"  {taxon}: matched {res['seen']:,} rows, kept {len(res['sample']):,}")

    # ----------------------------------------------------------------------
    # SECTION 3d: Merge reservoirs, dedupe, trim to global target
    # ----------------------------------------------------------------------
    print('\n=== 3d. Merge reservoirs, dedupe, trim to global target ===')

    all_records = []
    for taxon, res in reservoirs.items():
        all_records.extend(res["sample"])
        res["sample"] = None  # MEMORY FIX: release reservoir storage once merged
    gc.collect()

    print(f"Total records before dedup: {len(all_records):,}")

    # MEMORY FIX: dedupe via generator instead of building a second full list
    # alongside the seen-set and the original list simultaneously.
    seen_seqs = set()


    def _dedup_gen(records_iterable):
        for r in records_iterable:
            s = r["sequence"]
            if s not in seen_seqs:
                seen_seqs.add(s)
                yield r


    deduped = list(_dedup_gen(all_records))
    all_records = None
    gc.collect()

    if len(deduped) > TARGET_TOTAL_SEQS:
        deduped = random.sample(deduped, TARGET_TOTAL_SEQS)

    print(f"Final dataset size: {len(deduped):,}")

    os.makedirs("./data", exist_ok=True)
    with open("./data/sequences.json", "w") as f:
        json.dump(deduped, f)

    # ----------------------------------------------------------------------
    # SECTION 4: Tokenizer + dataset (tokenize ONCE up front)
    # ----------------------------------------------------------------------
    print('\n=== 4. Tokenizer + dataset ===')

    with open("./data/sequences.json") as f:
        records = json.load(f)

    VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "PAD": 5}
    MAX_LEN = 700  # COI barcodes are ~650bp; pad/truncate to this


    def tokenize(seq, max_len=MAX_LEN):
        ids = [VOCAB.get(b, VOCAB["N"]) for b in seq[:max_len]]
        ids += [VOCAB["PAD"]] * (max_len - len(ids))
        return ids


    # SPEED FIX: tokenize every record once, up front, into a single tensor.
    # v4 tokenized inside Dataset.__getitem__, repeating string indexing + dict
    # lookups on every single sample access across every epoch -- pure overhead.
    print("Pre-tokenizing all sequences (one-time cost)...")
    _tok_start = time.time()
    all_tokenized = torch.tensor([tokenize(r["sequence"]) for r in records], dtype=torch.long)
    print(f"  done in {time.time()-_tok_start:.1f}s -- tensor shape {tuple(all_tokenized.shape)}")

    by_genus = defaultdict(list)
    for i, r in enumerate(records):
        # clean_tax guards against NaN here too: even though section 3c now
        # writes clean strings for *new* scans, your existing cached
        # ./data/sequences.json was written before that fix and still has
        # raw NaN in it for records missing genus/family. Without this, a
        # record's `or` chain treats NaN as truthy and never falls through
        # to "unknown", which is what caused the KeyError: nan in training.
        key = clean_tax(r.get("genus_name")) or clean_tax(r.get("family_name")) or "unknown"
        by_genus[key].append(i)

    trainable_genera = {k: v for k, v in by_genus.items() if len(v) >= 2}
    print(f"Genera/groups usable for contrastive training: {len(trainable_genera)}")



    # (ContrastivePairDataset is now defined above, right after the imports -- see note at top of file.)

    train_ds = ContrastivePairDataset(all_tokenized, trainable_genera)
    print(f"Training pairs per epoch: {len(train_ds)}")

    # ----------------------------------------------------------------------
    # SECTION 5: Model -- dilated CNN encoder (no Transformer) + up-projection
    # ----------------------------------------------------------------------
    print('\n=== 5. Model: dilated CNN encoder (256-dim, no attention) + learned up-projection (5000-dim) ===')

    EMBED_DIM = 256
    UPPROJ_DIM = 5000


    class DNAEncoderCNN(nn.Module):
        """
        Pure CNN encoder, no self-attention.

        Why: self-attention is O(L^2) and made of many small ops -- a poor fit
        for hybrid P/E-core CPUs where per-op dispatch overhead is relatively
        high. Dilated convolutions are fully parallel across the sequence
        dimension and give a large effective receptive field cheaply, which
        suits short, fixed-length barcode sequences (~650bp) well. This is the
        same family of architecture used by DeepBind-style genomic sequence
        models.

        Receptive field math (kernel=9 each layer):
          layer1 (dilation=1): +8   -> RF=9
          layer2 (dilation=2): +16  -> RF=25
          layer3 (dilation=4): +32  -> RF=57
          layer4 (dilation=1, stride=2, downsample): RF effectively covers
            the full local motif neighborhood feeding into the final pooled
            representation.
        """

        def __init__(self, vocab_size=6, embed_dim=EMBED_DIM):
            super().__init__()
            self.token_embed = nn.Embedding(vocab_size, 128, padding_idx=VOCAB["PAD"])

            self.conv = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=9, padding=4, dilation=1),   # RF=9
                nn.GELU(),
                nn.Conv1d(256, 256, kernel_size=9, padding=8, dilation=2),   # RF=25
                nn.GELU(),
                nn.Conv1d(256, 256, kernel_size=9, padding=16, dilation=4),  # RF=57
                nn.GELU(),
                nn.Conv1d(256, 256, kernel_size=9, padding=4, stride=2),     # downsample L -> L/2
                nn.GELU(),
            )
            self.pool_proj = nn.Linear(256, embed_dim)

        def forward(self, x):
            h = self.token_embed(x)              # (B, L, 128)
            h = h.transpose(1, 2)                # (B, 128, L)
            h = self.conv(h)                     # (B, 256, L/2)
            pooled = h.mean(dim=-1)              # global average pool -> (B, 256)
            emb = self.pool_proj(pooled)         # (B, embed_dim)
            return F.normalize(emb, dim=-1)


    class UpProjector(nn.Module):
        """Fixed/learned linear map from the trained 256-dim space to 5000-dim."""

        def __init__(self, in_dim=EMBED_DIM, out_dim=UPPROJ_DIM):
            super().__init__()
            self.proj = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.proj(x)


    encoder = DNAEncoderCNN().cuda() if torch.cuda.is_available() else DNAEncoderCNN()
    print(encoder)

    # ----------------------------------------------------------------------
    # SECTION 5b: Optional thread-count benchmark
    # ----------------------------------------------------------------------
    if BENCH_THREADS:
        print('\n=== 5b. Benchmarking thread counts on a few warmup batches ===')
        bench_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
        bench_batches = []
        for i, b in enumerate(bench_loader):
            bench_batches.append(b)
            if i >= 4:
                break

        for nthreads in [2, 4, 6, 8]:
            torch.set_num_threads(nthreads)
            enc_bench = DNAEncoderCNN()
            enc_bench.train()
            t0 = time.time()
            for seq_a, seq_b in bench_batches:
                za, zb = enc_bench(seq_a), enc_bench(seq_b)
                (za.sum() + zb.sum()).backward()
            dt = time.time() - t0
            print(f"  threads={nthreads}: {dt:.2f}s for {len(bench_batches)} batches "
                  f"({dt/len(bench_batches):.2f}s/batch)")

        torch.set_num_threads(DEFAULT_NUM_THREADS)  # restore
        print(f"Restored to {DEFAULT_NUM_THREADS} threads. "
              f"Edit DEFAULT_NUM_THREADS above with the fastest value and rerun with BENCH_THREADS=False.")

    # ----------------------------------------------------------------------
    # SECTION 6: Contrastive training loop (InfoNCE / SimCLR-style)
    # ----------------------------------------------------------------------
    print('\n=== 6. Contrastive training loop (InfoNCE / SimCLR-style, in 256-dim space) ===')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)
    print(f"Training on device: {device}")

    EPOCHS = 4
    BATCH_SIZE = 128       # raised from 64 -- larger batches are typically more
                            # CPU-efficient (better cache reuse, fewer per-sample
                            # Python overheads), and InfoNCE also benefits from
                            # more negatives per batch.
    LR = 3e-4
    TEMP = 0.07
    PRINT_EVERY = 5
    NUM_WORKERS = 2         # safe now that __getitem__ only slices tensors

    loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0),
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)

    # Short linear warmup helps smooth out the very noisy early-training loss
    # (small batches + random init can otherwise look like nothing is learning
    # for the first several dozen steps).
    WARMUP_STEPS = 50
    total_steps = EPOCHS * len(loader)


    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        return 1.0


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    def info_nce_loss(z_a, z_b, temp=TEMP):
        B = z_a.size(0)
        z = torch.cat([z_a, z_b], dim=0)              # (2B, D)
        sim = z @ z.T / temp                          # (2B, 2B)
        sim.fill_diagonal_(-1e9)
        targets = torch.arange(B, device=z.device)
        targets = torch.cat([targets + B, targets])
        return F.cross_entropy(sim, targets)


    encoder.train()
    print(f"Batches per epoch: {len(loader)}")
    global_step = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        epoch_start = _time.time()
        for batch_idx, (seq_a, seq_b) in enumerate(loader):
            batch_start = _time.time()
            seq_a, seq_b = seq_a.to(device), seq_b.to(device)
            z_a, z_b = encoder(seq_a), encoder(seq_b)
            loss = info_nce_loss(z_a, z_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            total_loss += loss.item()
            if batch_idx % PRINT_EVERY == 0:
                print(f"  epoch {epoch+1} batch {batch_idx+1}/{len(loader)} "
                      f"- loss {loss.item():.4f} - lr {scheduler.get_last_lr()[0]:.2e} "
                      f"- {_time.time()-batch_start:.2f}s/batch")
        print(f"Epoch {epoch+1}/{EPOCHS} - avg loss: {total_loss/len(loader):.4f} "
              f"- {(_time.time()-epoch_start)/60:.1f} min")

    torch.save(encoder.state_dict(), "./data/encoder.pt")

    # ----------------------------------------------------------------------
    # SECTION 7: Train the up-projection (256 -> 5000)
    # ----------------------------------------------------------------------
    print('\n=== 7. Train the up-projection (256 -> 5000) to preserve neighborhood structure ===')

    encoder.eval()
    upproj = UpProjector().to(device)
    up_optimizer = torch.optim.AdamW(upproj.parameters(), lr=1e-3)

    UP_EPOCHS = 5
    up_loader = DataLoader(torch.arange(len(records)), batch_size=256, shuffle=True)

    print(f"Up-proj batches per epoch: {len(up_loader)}")
    for epoch in range(UP_EPOCHS):
        total = 0.0
        epoch_start = _time.time()
        for batch_idx, idx_batch in enumerate(up_loader):
            toks = all_tokenized[idx_batch].to(device)
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

    # ----------------------------------------------------------------------
    # SECTION 8: Build the searchable reference index (FAISS) + taxonomy lookup
    # ----------------------------------------------------------------------
    print('\n=== 8. Build the searchable reference index (FAISS) + taxonomy lookup ===')

    import faiss
    import numpy as np

    encoder.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(records), 512):
            batch = all_tokenized[i:i + 512].to(device)
            z = encoder(batch).cpu().numpy()
            all_embeds.append(z)
    all_embeds = np.concatenate(all_embeds, axis=0).astype("float32")

    index = faiss.IndexFlatIP(EMBED_DIM)  # cosine sim via inner product (vectors are normalized)
    index.add(all_embeds)
    faiss.write_index(index, "./data/reference.index")

    taxonomy_lookup = [
        {k: clean_tax(r.get(k, "")) for k in
         ["phylum_name", "class_name", "order_name", "family_name", "genus_name", "species_name"]}
        for r in records
    ]
    with open("./data/taxonomy_lookup.json", "w") as f:
        json.dump(taxonomy_lookup, f)

    print(f"Index built with {index.ntotal} reference sequences.")

    # ----------------------------------------------------------------------
    # SECTION 9: Query function -- raw sequence -> nearest taxa + confidence
    # ----------------------------------------------------------------------
    print('\n=== 9. Query function: raw sequence -> nearest taxa + confidence ===')


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
    print('\n=== 10. 2D map of the embedding space (UMAP) -- see evolutionary/clustering structure ===')

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
    demo.launch(share=False)  # local run -- opens http://127. 0.0.1:7860

    # ----------------------------------------------------------------------
    # ## Notes
    # - **Why CNN instead of Transformer:** self-attention is O(L^2) and made up
    #   of many small ops -- a poor fit for hybrid P/E-core CPUs (e.g. 13th-gen
    #   Intel U-series) where per-op dispatch overhead matters relatively more.
    #   Dilated convolutions are fully parallel across the sequence dimension
    #   and give a large receptive field cheaply, which is a good match for
    #   short, fixed-length barcode sequences (~650bp).
    # - **Why 256-dim for training, 5000-dim for storage:** unchanged from v4 --
    #   contrastive losses degrade in very high-dim spaces, so the model learns
    #   a tight 256-dim space, then a learned linear map preserves that
    #   structure at 5000-dim for downstream compatibility.
    # - **Thread tuning:** set BENCH_THREADS = True once to find the fastest
    #   torch.set_num_threads() value for your exact CPU, then hardcode it via
    #   DEFAULT_NUM_THREADS and set BENCH_THREADS back to False.
    # - **If still too slow:** drop TARGET_TOTAL_SEQS in section 2 (e.g. to
    #   50,000) for a faster end-to-end test run before committing to the full
    #   200k, or drop EPOCHS to 2-3 for a first pass.
    # - **Memory:** section 3c now explicitly deletes chunk references and
    #   periodically calls gc.collect() during the long streaming pass; section
    #   3d dedupes via a generator instead of holding two full record lists in
    #   memory simultaneously. The cached ./data/sequences.json means you don't
    #   need to re-scan the 31GB TSV on subsequent runs -- if you're only
    #   iterating on the model (sections 4 onward), you can comment out section
    #   3c/3d entirely and just load the cached file in section 4.
    # ----------------------------------------------------------------------