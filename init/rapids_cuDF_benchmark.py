import time
import pandas as pd
import cudf

# ——— CONFIG ———————————————————————————————————————————————————
# Path to the downloaded 1brc dataset CSV
DATA_FILE = "../data/20210830-WeatherData.csv"

# Column that holds the date (or station/day grouping key)
DATE_COL = "Date"

# Column that holds the daily mean temperature
# (in the 1brc CSV this is called something like "MeanTemp" or "TAVG" – check your header)
TEMP_COL = "MeanTemp"

# Number of times to repeat the group+agg to smooth out noise
REPEATS = 3
# ——————————————————————————————————————————————————————————

def timeit(func, *args, **kwargs):
    t0 = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - t0

def pandas_max_mean_temp(df):
    # group by DATE_COL, compute avg of TEMP_COL, then take the overall max
    return df.groupby(DATE_COL)[TEMP_COL].mean().max()

def cudf_max_mean_temp(df):
    # same in cuDF
    return df.groupby(DATE_COL)[TEMP_COL].mean().max().item()

def benchmark():
    print(f"\nLoading & computing on {DATA_FILE!r}\n" + "-"*60)
    # pandas
    print("→ pandas:")
    df_pd, t_load_pd = timeit(pd.read_csv, DATA_FILE)
    print(f"   load: {t_load_pd:.2f}s")

    # warm-up / repeat the group-by a few times
    t_agg_pd = 0.0
    max_mean_pd = None
    for _ in range(REPEATS):
        val, dt = timeit(pandas_max_mean_temp, df_pd)
        t_agg_pd += dt
        max_mean_pd = val
    t_agg_pd /= REPEATS
    print(f"   avg(group→max)  over {REPEATS} runs: {t_agg_pd:.2f}s → max_mean={max_mean_pd:.2f}")

    # cuDF
    print("\n→ cuDF (GPU):")
    df_gpu, t_load_gpu = timeit(cudf.read_csv, DATA_FILE)
    print(f"   load: {t_load_gpu:.2f}s")

    t_agg_gpu = 0.0
    max_mean_gpu = None
    for _ in range(REPEATS):
        val, dt = timeit(cudf_max_mean_temp, df_gpu)
        t_agg_gpu += dt
        max_mean_gpu = val
    t_agg_gpu /= REPEATS
    print(f"   avg(group→max)  over {REPEATS} runs: {t_agg_gpu:.2f}s → max_mean={max_mean_gpu:.2f}")

    # speedups
    print("\nSpeedups:")
    print(f"  load speedup : {t_load_pd / t_load_gpu:.1f}×")
    print(f"  agg speedup  : {t_agg_pd  / t_agg_gpu:.1f}×")

if __name__ == "__main__":
    benchmark()