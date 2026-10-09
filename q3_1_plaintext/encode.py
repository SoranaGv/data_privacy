"""
Turn texts into vectors with the Snowflake model.

"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # from the model card
DIM = 256          
BATCH_SIZE = 32   

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = ROOT / "embeddings"

# The model is loaded the first time we need it, not when this file is imported.
_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model
    if _model is None:
        print("Loading the Snowflake model ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModel.from_pretrained(MODEL_ID, add_pooling_layer=False)
        _model.eval()  # we only use the model, we don't train it
    return _tokenizer, _model


def truncate_and_normalize(vectors, dim=DIM):
    """Keep the first `dim` numbers of each vector, then scale it back to length 1."""
    short = vectors[:, :dim]
    lengths = np.linalg.norm(short, axis=1, keepdims=True)
    return short / lengths


def encode_full(texts, is_query=False):
    """Return one 768-dim [CLS] vector per text before truncating."""
    tokenizer, model = load_model()

    # Questions get the prefix, documents don't.
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]

    all_vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        # Split the texts into tokens the model reads at most 512.
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")

        # Run the model. It gives one 768-dim vector per token.
        with torch.inference_mode():
            output = model(**tokens)

        # Keep only the [CLS] vector, which is the first token.
        cls_vectors = output.last_hidden_state[:, 0]
        all_vectors.append(cls_vectors.numpy())

    return np.concatenate(all_vectors).astype(np.float32)


def encode(texts, is_query=False):
    # Keep the first 256 numbers and re-normalize.
    return truncate_and_normalize(encode_full(texts, is_query), DIM)


def main():
    with open(DATA_DIR / "docs.json") as f:
        docs = json.load(f)
    with open(DATA_DIR / "queries.json") as f:
        queries = json.load(f)

    doc_texts = [d["text"] for d in docs]
    query_texts = [q["question"] for q in queries]

    # Encode everything once and save the full 768-dim vectors.
    # Needs 64/128/256/768, and all of them can be cut from these.
    print(f"Encoding {len(doc_texts)} documents.")
    doc_vectors = encode_full(doc_texts, is_query=False)
    print(f"Encoding {len(query_texts)} questions.")
    query_vectors = encode_full(query_texts, is_query=True)

    EMB_DIR.mkdir(exist_ok=True)
    np.save(EMB_DIR / "doc_768.npy", doc_vectors)
    np.save(EMB_DIR / "query_768.npy", query_vectors)
    print("Saved", doc_vectors.shape, "and", query_vectors.shape, "to", EMB_DIR)

    # Sanity check 
    d = truncate_and_normalize(doc_vectors)
    q = truncate_and_normalize(query_vectors)
    top_result = np.argmax(q @ d.T, axis=1)
    correct_doc = np.array([x["gold_doc_id"] for x in queries])
    print(f"Quick check: {np.mean(top_result == correct_doc):.0%} of questions find their own document first.")


if __name__ == "__main__":
    main()