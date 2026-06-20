"""
Atropos Explorer -- 3D layout precompute (run this ONCE, offline)
====================================================================
Computes a 3D UMAP projection of the reference embedding cloud and saves
it to ./data/coords_3d.json. app.py just loads this file -- it never runs
UMAP itself, so the Flask server starts instantly every time.

WHY THE ORIGINAL VERSION HUNG FOR 30+ MINUTES
-----------------------------------------------
UMAP's default initialization is `init="spectral"`, which computes a
spectral embedding via an eigendecomposition (scipy's `eigsh`) of the
fuzzy graph Laplacian as a smart starting layout before the main
optimization. On ~200k points this eigensolver step can take a very long
time on CPU and, worse, doesn't print any progress -- it just looks hung.
That's exactly the call you Ctrl+C'd out of.

THE FIX
-------
- `init="random"` skips the spectral step entirely. UMAP's gradient-descent
  optimization still converges to a good layout from a random start, just
  with slightly less "global structure" fidelity than spectral init would
  give -- a fine tradeoff for an exploratory visualization. This alone
  should take the runtime from "30+ min and counting" to a few minutes.
- `low_memory=True` trades a bit of speed for staying well under your RAM
  ceiling on 200k x 256-dim input.
- `n_jobs` is left unset/-1 intentionally: passing a fixed `random_state`
  forces UMAP to single-thread (the warning you saw: "n_jobs value 1
  overridden... Use no seed for parallelism"). For this one-time offline
  run, reproducibility doesn't matter as much as speed, so we drop
  `random_state` and let it parallelize across your CPU cores. If you
  want a perfectly reproducible layout, add random_state=42 back and
  accept the single-threaded cost.
- Progress is printed via `verbose=True` so you can see it's alive instead
  of staring at a silent terminal.

Run with:
    python precompute_3d.py

Takes maybe 3-8 minutes on a 1315U for 200k points with these settings --
still not instant, but it runs ONCE, ever, and produces a file app.py
just reads from then on.
"""

import json
import os
import time

import numpy as np
import faiss

DATA_DIR = "./data"
INDEX_PATH = os.path.join(DATA_DIR, "reference.index")
COORDS_3D_CACHE = os.path.join(DATA_DIR, "coords_3d.json")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
all_embeds = index.reconstruct_n(0, index.ntotal).astype("float32")
print(f"  {index.ntotal:,} vectors, dim={all_embeds.shape[1]}")

print("\nFitting 3D UMAP (init='random' -- skips the slow spectral eigensolver step)...")
print("This will print progress as it runs. Expect several minutes, not 30+.")

import umap

t0 = time.time()
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=3,
    metric="cosine",
    init="random",      # <-- the actual fix: skips the eigsh hang
    low_memory=True,
    verbose=True,
)
coords = reducer.fit_transform(all_embeds).astype("float32")
print(f"\nDone in {(time.time()-t0)/60:.1f} min")

with open(COORDS_3D_CACHE, "w") as f:
    json.dump({"coords": coords.tolist()}, f)

print(f"Saved 3D coordinates to {COORDS_3D_CACHE}")
print("You can now run app.py -- it will load this file directly and start instantly.")