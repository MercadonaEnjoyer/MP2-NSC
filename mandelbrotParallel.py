import numpy as np
from dask import delayed
import dask
from dask.distributed import Client, LocalCluster
from numba import njit
from multiprocessing import Pool
import time, os, statistics, matplotlib.pyplot as plt
from pathlib import Path

@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0: 
            return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit
def mandelbrot_chunk(row_start, row_end, N,
    x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

# --- MP2 M2: add below M1 in mandelbrot_parallel.py ---


def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4, n_chunks=None, pool=None):
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    if pool is not None: 
        return np.vstack(pool.map(_worker, chunks))
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny)
        parts = p.map(_worker, chunks)
    return np.vstack(parts)

def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=12):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == "__main__":
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    p = 8  

    cluster = LocalCluster(n_workers=p, threads_per_worker=1)
    client = Client(cluster)

    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    n_chunks_list = list(range(1, 33))
    times = []

    # with Pool(processes=p) as pool:
    #     result = mandelbrot_parallel(
    #         N, X_MIN, X_MAX, Y_MIN, Y_MAX,
    #         max_iter=max_iter,
    #         pool=pool
    #     )

    # for n_chunks in n_chunks_list:
    #     t = []
    #     for _ in range(3):
    #         t0 = time.perf_counter()
    #         mandelbrot_dask(
    #             N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks
    #         )
    #         t.append(time.perf_counter() - t0)
    #     times.append(statistics.median(t))

    # T1 = times[0]

    # vs1x = [t / T1 for t in times]
    # speedup = [T1 / t for t in times]
    # LIF = [p * (t / T1) - 1 for t in times]  

    # print("\nn_chunks | time (s) | vs 1x | speedup | LIF")
    # print("-" * 55)
    # for n, t, v, s, l in zip(n_chunks_list, times, vs1x, speedup, LIF):
    #     print(f"{n:8d} | {t:8.3f} | {v:6.2f} | {s:7.2f} | {l:8.3f}")

    # t_min = min(times)
    # idx_opt = times.index(t_min)
    # n_opt = n_chunks_list[idx_opt]
    # LIF_min = min(LIF)

    # print("\n--- Summary ---")
    # print(f"n_chunks optimal : {n_opt}")
    # print(f"t_min            : {t_min:.3f} s")
    # print(f"LIF_min          : {LIF_min:.3f}")

    # plt.figure()
    # plt.plot(n_chunks_list, times, marker='o')
    # plt.xscale("log")
    # plt.xlabel("n_chunks (log scale)")
    # plt.ylabel("Wall time (s)")
    # plt.title(f"Dask Chunk Sweep (p={p})")
    # plt.grid(True)

    # plt.savefig("dask chunk sweep.png", dpi=300)
    # plt.close()
    for _ in range(3): 
        t0 = time.perf_counter()
        result = mandelbrot_dask(1024, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    print(f"Dask local(n_chunks=12):{statistics.median(times):.3f}s")
    client.close()
    cluster.close()