"""
Run from the repo root:
    python src/data.py

Creates:
    data/docs.json     1,000 documents  
    data/queries.json  200 questions   
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

N_DOCS = 1000      # size of the document database
N_QUERIES = 200    # number of test questions
SEED = 6          # our group number

# The data/ folder in the repo root
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    # Download SQuAD (validation split).
    # Each row is one question together with its paragraph.
    squad = load_dataset("rajpurkar/squad", split="validation")

    # Build the document list.
    # Many questions share the same paragraph, so we keep each paragraph only once.
    doc_id_of_text = {} 
    for row in squad:
        text = row["context"]
        if text not in doc_id_of_text:
            doc_id_of_text[text] = len(doc_id_of_text)
        if len(doc_id_of_text) == N_DOCS:
            break

    docs = [{"id": doc_id, "text": text} for text, doc_id in doc_id_of_text.items()]

    # Collect the questions that belong to each document.
    # We go over SQuAD again and keep only questions whose paragraph is one of our 1,000 documents.
    questions_per_doc = {} 
    for row in squad:
        doc_id = doc_id_of_text.get(row["context"]) 
        if doc_id is None:
            continue
        if doc_id not in questions_per_doc:
            questions_per_doc[doc_id] = []
        questions_per_doc[doc_id].append({"qid": row["id"], "question": row["question"]})

    # Pick 200 questions, each from another document.
    rng = random.Random(SEED)
    chosen_doc_ids = rng.sample(sorted(questions_per_doc), N_QUERIES)

    queries = []
    for doc_id in chosen_doc_ids:
        q = rng.choice(questions_per_doc[doc_id])  # one random question for this document
        queries.append({"qid": q["qid"], "question": q["question"], "gold_doc_id": doc_id})

    # Save both lists, so we use the same data.
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "docs.json", "w") as f:
        json.dump(docs, f, indent=1)
    with open(DATA_DIR / "queries.json", "w") as f:
        json.dump(queries, f, indent=1)

    # Sanity check
    print(f"Saved {len(docs)} documents and {len(queries)} queries to {DATA_DIR}")
    example = queries[0]
    print("Example question:", example["question"])
    print("Its document starts with:", docs[example["gold_doc_id"]]["text"][:100], "...")


if __name__ == "__main__":
    main()