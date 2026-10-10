"""
Q3.3 (e): time single CKKS operations (100 runs each) vs. NumPy on plaintext.

TenSEAL's CKKSVector has no public rotate(). A rotation is therefore measured
indirectly through sum(), which is "log2(n) x (rotate + add)":
    t_sum(n=2)    = 1 rotation  + 1 add
    t_sum(n=4096) = 12 rotations + 12 adds
so   rotation ~= (t_sum(4096) - t_sum(2)) / 11 - t_add   (and t_sum(2) - t_add as a cross-check)
"""
import json
import time
from pathlib import Path

import numpy as np
import tenseal as ts

from encryRetrieval import make_client_context

ROOT = Path(__file__).resolve().parent.parent
RUNS = 100
perf = time.perf_counter


def avg_ms(fn, runs=RUNS):
    fn()
    t = perf()
    for _ in range(runs):
        fn()
    return 1000 * (perf() - t) / runs


def main():
    ctx = make_client_context()
    rng = np.random.default_rng(0)
    slots = 4096
    x, y = rng.random(slots), rng.random(slots)
    a, b = ts.ckks_vector(ctx, x.tolist()), ts.ckks_vector(ctx, y.tolist())
    a2 = ts.ckks_vector(ctx, [0.5, 0.25])
    y_list = y.tolist()

    he = {
        "ct_ct_multiplication": avg_ms(lambda: a * b),      # incl. relinearize + rescale
        "ct_pt_multiplication": avg_ms(lambda: a * y_list), # incl. encoding the plaintext
        "addition": avg_ms(lambda: a + b),
    }
    t_sum_small, t_sum_full = avg_ms(lambda: a2.sum()), avg_ms(lambda: a.sum())
    he["rotation"] = (t_sum_full - t_sum_small) / 11 - he["addition"]
    he["rotation_crosscheck_from_sum2"] = t_sum_small - he["addition"]

    x64, y64 = x[:64], y[:64]
    np_ms = { # NumPy, 4096 and 64 elements
        "ct_ct_multiplication": (avg_ms(lambda: x * y), avg_ms(lambda: x64 * y64)),
        "ct_pt_multiplication": (avg_ms(lambda: x * y), avg_ms(lambda: x64 * y64)),
        "addition": (avg_ms(lambda: x + y), avg_ms(lambda: x64 + y64)),
        "rotation": (avg_ms(lambda: np.roll(x, 1)), avg_ms(lambda: np.roll(x64, 1))),
    }

    print(f"{'operation':<24}{'CKKS (ms)':>12}{'NumPy 4096 (ms)':>18}{'NumPy 64 (ms)':>16}{'slowdown vs 4096':>20}")
    for k, (n4096, n64) in np_ms.items():
        print(f"{k:<24}{he[k]:>12.4f}{n4096:>18.5f}{n64:>16.5f}{he[k] / n4096:>19,.0f}x")
    print(f"(rotation cross-check from sum on 2 elements: {he['rotation_crosscheck_from_sum2']:.4f} ms)")

    out = ROOT / "results" / "q3_3"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"runs": RUNS, "ckks_ms": he,
               "numpy_ms_4096": {k: v[0] for k, v in np_ms.items()},
               "numpy_ms_64": {k: v[1] for k, v in np_ms.items()}},
              open(out / "single_ops.json", "w"), indent=1)


if __name__ == "__main__":
    main()