
"""
Multi-GPU Pandas vs cuDF Benchmark on ~115M Rows (NYC Yellow Taxi)
-----------------------------------------------------------------
* Downloads 5 months of Yellow Taxi Parquet (~115M rows total) if missing
* Spins up a LocalCUDACluster across all GPUs
* Compares Pandas (CPU) vs Dask-cuDF (multi-GPU) for daily mean total_fare → max
"""
import os
import time
import glob
import urllib.request
from pathlib import Path

import pandas as pd
import cudf
import dask_cudf
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

# ── CONFIG ─────────────────────────────────────────────
MONTHS     = ["2023-01","2023-02","2023-03","2023-04","2023-05"]  # 5 months ≈ 115M rows
data_dir   = Path("data")
REPEATS    = 3
TS_COL     = "tpep_pickup_datetime"
VAL_COL    = "total_amount"
BASE_URL   = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
# ───────────────────────────────────────────────────────

def ensure_data():
    data_dir.mkdir(exist_ok=True)
    files = []
    for m in MONTHS:
        fname = f"yellow_tripdata_{m}.parquet"
        path = data_dir / fname
        if not path.exists():
            url = BASE_URL + fname
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, path)
            print(f"Saved to {path}")
        files.append(str(path))
    return files

# CPU path: load + prepare + aggregate
def pandas_workflow(files):
    # load all into one DataFrame
    dfs = [pd.read_parquet(f) for f in files]
    pdf = pd.concat(dfs, ignore_index=True)
    pdf["day"] = pd.to_datetime(pdf[TS_COL]).dt.date
    # group & aggregate
    return pdf.groupby("day")[VAL_COL].mean().max()

# GPU path: Dask-cuDF multi-GPU
def cudf_workflow(files):
    cluster = LocalCUDACluster()
    client = Client(cluster)
    n_gpus = len(client.ncores())
    print(f"Using {n_gpus} GPU(s) for Dask-cuDF")

    ddf = dask_cudf.read_parquet(files)
    ddf["day"] = ddf[TS_COL].dt.date
    max_series = ddf.groupby("day")[VAL_COL].mean()
    result = max_series.max().compute()

    client.close(); cluster.close()
    return result

# Utility timer
def timeit(func, *args, **kwargs):
    t0 = time.time()
    out = func(*args, **kwargs)
    return out, time.time() - t0

# Main benchmark
def main():
    files, _ = timeit(lambda: ensure_data())
    print(f"\nFiles ready: {len(files)} months → ~{len(files)*23}M rows\n{'-'*60}")

    # Pandas
    print("→ pandas (CPU)")
    _, t_load = timeit(pandas_workflow, files)
    print(f"  full load+agg: {t_load:.2f}s")

    # Dask-cuDF
    print("\n→ Dask-cuDF (multi-GPU)")
    _, t_gpu = timeit(cudf_workflow, files)
    print(f"  full load+agg: {t_gpu:.2f}s")

    print("\nSpeed-up (CPU ÷ GPU):", f"{t_load/t_gpu:.2f}×")

if __name__ == "__main__":
    main()
