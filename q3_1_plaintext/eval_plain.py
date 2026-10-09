"""
3.1(d) - Recall@1/5/10 and average query latency of the plaintext retriever.

"""
import json
import platform
import time
from pathlib import Path

from encode import encode
from server import load_server

K_VALUES = [1, 5, 10]
WARMUP = 5  # first few queries are slower (caches, model start-up), so we don't time them

ROOT = Path(__file__).resolve().parent.parent


def cpu_name():
    """Name of the CPU, so the report says which machine the timings come from."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    with open(ROOT / "data" / "queries.json") as f:
        queries = json.load(f)
    server = load_server()

    # Warm-up not timed
    for q in queries[:WARMUP]:
        server.search(encode([q["question"]], is_query=True)[0], 10)

    # Run all 200 questions, one at a time like a real client.
    # Same steps as Client.ask(), but we time the encoding separately.
    stages = ["encode", "similarity", "top_k", "fetch"]
    total_time = {s: 0.0 for s in stages}
    hits = {k: 0 for k in K_VALUES}
    top10_per_query = []

    for q in queries:
        t0 = time.perf_counter()
        query_vector = encode([q["question"]], is_query=True)[0]   # client side
        total_time["encode"] += time.perf_counter() - t0

        results = server.search(query_vector, 10)                  # server side
        for stage, seconds in server.last_timings.items():
            total_time[stage] += seconds

        top10 = [r["id"] for r in results]
        top10_per_query.append({"qid": q["qid"], "top10": top10})
        for k in K_VALUES:
            if q["gold_doc_id"] in top10[:k]:
                hits[k] += 1

    # Averages
    n = len(queries)
    recall = {f"Recall@{k}": hits[k] / n for k in K_VALUES}
    avg_ms = {s: 1000 * total_time[s] / n for s in stages}
    avg_ms["total"] = sum(avg_ms.values())

    print(f"{n} questions on {cpu_name()}")
    for name, value in recall.items():
        print(f"{name}: {value:.3f}")
    print("Average time per question (ms):")
    for stage, ms in avg_ms.items():
        print(f"  {stage:<10} {ms:8.3f}")

    # Save for the report 
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "plain_eval.json", "w") as f:
        json.dump({"n_queries": n, "warmup": WARMUP, "cpu": cpu_name(),
                   "recall": recall, "avg_latency_ms": avg_ms}, f, indent=1)
    with open(ROOT / "results" / "plain_top10.json", "w") as f:
        json.dump(top10_per_query, f, indent=1)
    print("Saved to results/plain_eval.json and results/plain_top10.json")


if __name__ == "__main__":
    main()