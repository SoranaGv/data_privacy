"""
Our own cosine similarity and top-k selection.

"""
import numpy as np


def cosine_similarity(query, doc_vectors):
    """Cosine similarity between one query and every document.
    
    """
    dot_products = doc_vectors @ query
    lengths = np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query)
    return dot_products / lengths


def top_k(scores, k):
    """Return the positions of the k highest scores, best first.

    We go over the scores once and keep a short list of the k best seen so far,
    sorted from best to worst.
    """
    best = []  # (score, position) pairs, best first, at most k of them
    for position, score in enumerate(scores.tolist()):
        # Skip scores that are worse than the current k-th best
        if len(best) == k and score <= best[-1][0]:
            continue

        # Find where this score goes in the sorted list, and insert it there
        place = 0
        while place < len(best) and best[place][0] >= score:
            place += 1
        best.insert(place, (score, position))

        # Keep only the k best
        if len(best) > k:
            best.pop()

    return [position for _, position in best]


if __name__ == "__main__":
    # Test on random data. np.argsort is used ONLY here, to check our answers.
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(1000, 256))
    query = rng.normal(size=256)

    scores = cosine_similarity(query, docs)
    expected = (docs @ query) / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query))
    assert np.allclose(scores, expected)

    for k in (1, 5, 10):
        assert top_k(scores, k) == list(np.argsort(-scores)[:k])

    print("All tests passed")