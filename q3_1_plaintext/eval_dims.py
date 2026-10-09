"""
Recall@10 at 64, 128, 256 and 768 dimensions.


"""
import json
from pathlib import Path

import numpy as np

from encode import truncate_and_normalize
from similarity import cosine_similarity, top_k

DIMS = [64, 128, 256, 768]
K = 10

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "q3_1"


def recall_at_k(doc_vectors, query_vectors, correct_docs, k):
    """Fraction of questions whose own document is in their top k results."""
    hits = 0
    for query, correct in zip(query_vectors, correct_docs):
        scores = cosine_similarity(query, doc_vectors)
        if correct in top_k(scores, k):
            hits += 1
    return hits / len(correct_docs)


def main():
    # Load the saved vectors and the correct document of each question
    doc_768 = np.load(ROOT / "embeddings" / "doc_768.npy")
    query_768 = np.load(ROOT / "embeddings" / "query_768.npy")
    with open(ROOT / "data" / "queries.json") as f:
        correct_docs = [q["gold_doc_id"] for q in json.load(f)]

    # For each size, cut the vectors, re-normalize, and measure Recall@10
    results = {}
    print("Dimensions  Recall@10")
    for dim in DIMS:
        docs = truncate_and_normalize(doc_768, dim)
        queries = truncate_and_normalize(query_768, dim)
        recall = recall_at_k(docs, queries, correct_docs, K)
        results[dim] = recall
        print(f"{dim:>10}  {recall:.3f}")

    # Save the numbers for the report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "recall_by_dim.json", "w") as f:
        json.dump(results, f, indent=1)
    print("Saved to results/q3_1/recall_by_dim.json")


if __name__ == "__main__":
    main()