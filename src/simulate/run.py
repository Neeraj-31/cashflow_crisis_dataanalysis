"""Run the simulation study (all training lengths, train + held-out firms) in parallel.

    python -m src.simulate.run [--firms N] [--parametric-B B] [--workers W]

Writes data/processed/sim_runs.pkl.gz (per-firm thresholds + monitored CUSUM paths).
Defaults come from config: simulation.firms_per_cell firms per (set, n).
--parametric-B: refits per firm for the parametric-bootstrap fix (default 200; see decision log).
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import time
from multiprocessing import Pool

from src.config import Paths, load_config
from src.simulate.experiment import run_firm


def safe_run(task):
    try:
        return run_firm(task)
    except Exception as e:  # recorded, excluded and counted in the analysis; never silently dropped
        seed, s, n, f, _ = task
        return {"set": s, "n": n, "firm": f, "error": f"{type(e).__name__}: {e}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--firms", type=int, default=None)
    ap.add_argument("--parametric-B", type=int, default=200)
    ap.add_argument("--workers", type=int, default=min(9, max(1, (os.cpu_count() or 2) - 2)))  # 20 workers ran out of memory
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(); cfg["_parametric_B"] = a.parametric_B
    sc = cfg["simulation"]
    F = a.firms or sc["firms_per_cell"]
    tasks = [(cfg["seed"], s, n, f, cfg) for s in ("heldout", "train") for n in sc["training_lengths"] for f in range(F)]
    paths = Paths.from_config(cfg).ensure()
    out = a.out or str(paths.processed / "sim_runs.pkl.gz")
    ckpt = out + ".partial"
    res = []
    try:
        with gzip.open(ckpt, "rb") as f:
            res = pickle.load(f)
    except (FileNotFoundError, EOFError):
        pass
    done = {(r["set"], r["n"], r["firm"]) for r in res}
    todo = [t for t in tasks if (t[1], t[2], t[3]) not in done]
    print(f"resuming with {len(res)} done, {len(todo)} to run", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(safe_run, todo, chunksize=4), 1):
            res.append(r)
            if i % 250 == 0 or i == len(todo):
                with gzip.open(ckpt, "wb") as f:
                    pickle.dump(res, f)
                print(f"{len(res)}/{len(tasks)} firms  {time.time() - t0:,.0f}s", flush=True)
    res.sort(key=lambda r: (r["set"], r["n"], r["firm"]))
    with gzip.open(out, "wb") as f:
        pickle.dump({"meta": {"firms_per_cell": F, "parametric_B": a.parametric_B, "seed": cfg["seed"]}, "runs": res}, f)
    print("saved", out)


if __name__ == "__main__":
    main()
