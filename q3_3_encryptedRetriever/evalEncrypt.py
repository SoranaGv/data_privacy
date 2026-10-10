"""
Q3.3 (c) + (d): correctness and per-stage latency of the encrypted retriever,
compared with the plaintext retriever of Q3.1.
"""
import argparse
import json
import platform
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent/ "q3_1_plaintext"))

import numpy as np

from encode import truncate_and_normalize
from encryRetrieval import DIM, EncryptedClient, EncryptedServer
from similarity import cosine_similarity, top_k

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "q3_3"
K_VALUES = [1, 5, 10]
perf = time.perf_counter


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--live-encode", action="store_true")
    args = ap.parse_args()

    docs = json.load(open(ROOT / "data" / "docs.json"))
    queries = json.load(open(ROOT / "data" / "queries.json"))[: args.n_queries]
    plain31 = json.load(open(ROOT / "results" / "q3_1" / "plain_eval.json"))
    top10_31 = {r["qid"]: r["top10"] for r in json.load(open(ROOT / "results" / "q3_1" / "plain_top10.json"))}

    doc_768 = np.load(ROOT / "embeddings" / "doc_768.npy")
    query_768 = np.load(ROOT / "embeddings" / "query_768.npy")
    D = truncate_and_normalize(doc_768, DIM) # 64-dim, length 1
    Q = truncate_and_normalize(query_768, DIM)
    texts = [d["text"] for d in docs]

    # one-time setup
    t = perf(); client = EncryptedClient(); t_keygen = perf() - t
    pub = client.public_context()
    server = EncryptedServer(pub)
    t = perf(); enc_vecs, blobs = client.encrypt_documents(D, texts); t_encdocs = perf() - t
    server.upload(enc_vecs, blobs)
    print(f"setup: keygen {t_keygen:.2f}s, encrypt 1000 docs {t_encdocs:.2f}s")

    #  per-query loop
    st = ["encode", "encrypt_query", "similarity", "decrypt_scores", "top_k", "fetch"]
    he = {s: 0.0 for s in st}
    pl = {s: 0.0 for s in st} # plaintext retriever, also at 64 dims
    hits = {k: 0 for k in K_VALUES}
    hits_p64 = {k: 0 for k in K_VALUES}
    max_diff, exact64, set64, overlap31 = 0.0, 0, 0, []
    score_bytes = 0

    def warm(): # one untimed query (caches, allocator)
        s = server.search(client.encrypt_query(Q[0])); client.decrypt_scores(s)
    warm()

    live = None
    if args.live_encode:
        from encode import encode_full
        live = encode_full

    for i, qd in enumerate(queries):
        # encode (same model for both systems)
        if live:
            t = perf(); q = truncate_and_normalize(live([qd["question"]], is_query=True), DIM)[0]
            e = perf() - t
        else:
            q, e = Q[i], plain31["avg_latency_ms"]["encode"] / 1000
        he["encode"] += e; pl["encode"] += e

        # encrypted (3.3)
        t = perf(); enc_q = client.encrypt_query(q); he["encrypt_query"] += perf() - t
        t = perf(); enc_scores = server.search(enc_q); he["similarity"] += perf() - t
        t = perf(); scores = client.decrypt_scores(enc_scores); he["decrypt_scores"] += perf() - t
        t = perf(); best = top_k(scores, 10); he["top_k"] += perf() - t
        t = perf(); got = client.decrypt_texts(best, server.fetch(best)); he["fetch"] += perf() - t
        assert got == [texts[j] for j in best]            # texts decrypt correctly
        score_bytes = sum(len(b) for b in enc_scores)

        # plaintext at 64 dims (same code as 3.1, fair baseline)
        t = perf(); ps = cosine_similarity(q, D); pl["similarity"] += perf() - t
        t = perf(); pbest = top_k(ps, 10); pl["top_k"] += perf() - t
        t = perf(); _ = [{"id": j, "text": texts[j]} for j in pbest]; pl["fetch"] += perf() - t

        # correctness
        max_diff = max(max_diff, float(np.max(np.abs(scores - ps))))
        for k in K_VALUES:
            hits[k] += qd["gold_doc_id"] in best[:k]
            hits_p64[k] += qd["gold_doc_id"] in pbest[:k]
        exact64 += best == pbest
        set64 += set(best) == set(pbest)
        overlap31.append(len(set(best) & set(top10_31[qd["qid"]])) / 10)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)} queries done")

    n = len(queries)
    ms = lambda d: {s: 1000 * v / n for s, v in d.items()}
    he_ms, pl64_ms = ms(he), ms(pl)
    pl31_ms = {"encode": plain31["avg_latency_ms"]["encode"], "encrypt_query": 0.0,
               "similarity": plain31["avg_latency_ms"]["similarity"], "decrypt_scores": 0.0,
               "top_k": plain31["avg_latency_ms"]["top_k"], "fetch": plain31["avg_latency_ms"]["fetch"]}
    for d in (he_ms, pl64_ms, pl31_ms):
        d["total"] = sum(d.values())

    plain_storage = D.nbytes + sum(len(t.encode()) for t in texts)
    result = {
        "cpu": cpu_name(), "n_queries": n, "live_encode": bool(live),
        "recall_encrypted": {f"Recall@{k}": hits[k] / n for k in K_VALUES},
        "recall_plain_64d": {f"Recall@{k}": hits_p64[k] / n for k in K_VALUES},
        "max_abs_score_diff": max_diff,
        "top10_exact_match_vs_plain64": exact64 / n,
        "top10_same_set_vs_plain64": set64 / n,
        "top10_mean_overlap_vs_q31_256d": float(np.mean(overlap31)),
        "latency_ms": {"q3_1_plain_256d": pl31_ms, "plain_64d": pl64_ms, "q3_3_encrypted": he_ms},
        "setup": {"keygen_s": t_keygen, "encrypt_1000_docs_s": t_encdocs},
        "storage_bytes": {"enc_vectors": server.vector_bytes, "enc_texts": server.blob_bytes,
                          "public_context_with_galois_keys": server.ctx_bytes,
                          "plaintext_vectors_64d_float32+texts": int(plain_storage),
                          "one_vector_ciphertext": server.vector_bytes // 1000,
                          "enc_query": len(enc_q), "all_score_ciphertexts_per_query": score_bytes},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(OUT / "encrypted_eval.json", "w"), indent=1)

    #overhead table
    names = {"encode": "Encode question", "encrypt_query": "Encrypt query", "similarity": "Server similarity",
             "decrypt_scores": "Decrypt scores", "top_k": "Top-k selection",
             "fetch": "Fetch + decrypt docs", "total": "Total"}
    f = lambda x: f"{x:.3f}"
    lines = ["| Stage | Q3.1 plain 256d (ms) | plain 64d (ms) | Q3.3 encrypted 64d (ms) | Overhead vs 3.1 | Overhead vs 64d |",
             "|---|---|---|---|---|---|"]
    for s, name in names.items():
        a, b, c = pl31_ms[s], pl64_ms[s], he_ms[s]
        o1 = f"{c / a:,.1f}x" if a > 0 else "n/a (new)"
        o2 = f"{c / b:,.1f}x" if b > 0 else "n/a (new)"
        lines.append(f"| {name} | {f(a)} | {f(b)} | {f(c)} | {o1} | {o2} |")
    (OUT / "overhead_table.md").write_text("\n".join(lines) + "\n")

    print(f"\n{n} queries on {cpu_name()}")
    print("Recall encrypted:", result["recall_encrypted"], "| plain 64d:", result["recall_plain_64d"])
    print(f"max |enc - plain| score diff: {max_diff:.3e}")
    print(f"top-10 identical (order) to plain 64d: {exact64 / n:.3f}, same set: {set64 / n:.3f}")
    print(f"mean top-10 overlap with Q3.1 (256d): {np.mean(overlap31):.3f}")
    print("\n" + "\n".join(lines))
    print("\nstorage (MB):", {k: round(v / 1e6, 3) for k, v in result["storage_bytes"].items()})


if __name__ == "__main__":
    main()