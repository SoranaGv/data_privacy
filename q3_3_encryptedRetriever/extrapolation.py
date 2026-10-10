"""
Q3.3 (f): extrapolate query latency and server storage from 1,000 to 1,000,000 documents.
Reads results/q3_3/encrypted_eval.json (run eval_encrypted.py first) and single_ops.json.
Everything in the simple design (1 ciphertext per document) scales linearly with the number of docs.
"""
import json
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "q3_3"
N_FROM, N_TO = 1_000, 1_000_000
SLOTS, DIM = 4096, 64


def human_bytes(x):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1000 or u == "TB":
            return f"{x:,.1f} {u}"
        x /= 1000


def human_time(ms):
    s = ms / 1000
    return f"{s:,.1f} s" if s < 120 else f"{s / 60:,.1f} min" if s < 7200 else f"{s / 3600:,.1f} h"


def main():
    ev = json.load(open(R / "encrypted_eval.json"))
    ops = json.load(open(R / "single_ops.json"))["ckks_ms"]
    lat, sto = ev["latency_ms"]["q3_3_encrypted"], ev["storage_bytes"]
    f = N_TO / N_FROM

    print("== Simple design: 1 ciphertext per document (what we implemented) ==")
    for stage in ["similarity", "decrypt_scores"]:
        print(f"{stage:<16} {human_time(lat[stage] * f)}")
    per_doc = sto["one_vector_ciphertext"] + sto["enc_texts"] / N_FROM
    print(f"server storage   {human_bytes(sto['one_vector_ciphertext'] * N_TO)} (vectors) "
          f"+ {human_bytes(sto['enc_texts'] * f)} (texts) + {human_bytes(sto['public_context_with_galois_keys'])} (keys)")
    print(f"returned scores  {human_bytes(sto['all_score_ciphertexts_per_query'] * f)} per query (one ciphertext per document!)")

    print("\n== Packed design: floor(4096/64)=64 documents per ciphertext (estimate from op timings) ==")
    docs_per_ct = SLOTS // DIM
    n_ct = ceil(N_TO / docs_per_ct)
    rots = 6  # block-sum over 64 slots = log2(64) rotations
    t_ct = ops["ct_ct_multiplication"] + rots * (ops["rotation"] + ops["addition"])
    print(f"ciphertexts      {n_ct:,}")
    print(f"similarity       {human_time(n_ct * t_ct)}  ({t_ct:.2f} ms per ciphertext: 1 mult + {rots} rot + {rots} add)")
    print(f"vector storage   {human_bytes(n_ct * sto['one_vector_ciphertext'])}")
    print(f"returned scores  {human_bytes(n_ct * sto['one_vector_ciphertext'])} per query (still large)")
    json.dump({"note": "see printed output"}, open(R / "extrapolation.json", "w"))


if __name__ == "__main__":
    main()