"""
The server and the client of the plaintext retriever.

Server: stores the 1,000 documents and their vectors.
Client: holds the encoder and talks to the server ONLY through search(query_vector, k).


"""
import json
import time
from pathlib import Path

import numpy as np

from encode import encode, truncate_and_normalize
from similarity import cosine_similarity, top_k

ROOT = Path(__file__).resolve().parent.parent


class Server:
    """The remote document database. It only gets the question's vector, never its text."""

    def __init__(self, docs, doc_vectors):
        # The underscore means "private": the client never reads these directly.
        self._docs = docs            
        self._vectors = doc_vectors    
        self.last_timings = {}         # filled by search(), used to measure latency in 3.1(d)

    def search(self, query_vector, k):
        """Compare the query with every document and return the k most similar."""
        t0 = time.perf_counter()
        scores = cosine_similarity(query_vector, self._vectors)  # Similarity
        t1 = time.perf_counter()
        best = top_k(scores, k)                                   # Pick the top k
        t2 = time.perf_counter()
        results = [                                               # Fetch the documents
            {"id": self._docs[i]["id"], "score": float(scores[i]), "text": self._docs[i]["text"]}
            for i in best
        ]
        t3 = time.perf_counter()

        self.last_timings = {"similarity": t1 - t0, "top_k": t2 - t1, "fetch": t3 - t2}
        return results


class Client:
    """The user's device. It turns the question into a vector and calls server.search()."""

    def __init__(self, server):
        self.server = server

    def ask(self, question, k=10):
        query_vector = encode([question], is_query=True)[0]  # 256-dim, length 1
        return self.server.search(query_vector, k)


def load_server(dim=256):
    """Build the server from the saved documents and 768-dim vectors."""
    with open(ROOT / "data" / "docs.json") as f:
        docs = json.load(f)
    doc_768 = np.load(ROOT / "embeddings" / "doc_768.npy")
    return Server(docs, truncate_and_normalize(doc_768, dim))


if __name__ == "__main__":
    # Try one question from our query set
    with open(ROOT / "data" / "queries.json") as f:
        example = json.load(f)[0]

    client = Client(load_server())
    results = client.ask(example["question"], k=3)

    print("Question:", example["question"])
    print("Correct document:", example["gold_doc_id"])
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. doc {r['id']}  score {r['score']:.3f}  {r['text'][:70]}...")