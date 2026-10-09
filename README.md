# Data Privacy Assignment 2, Group 6

This README contains the code for the whole assignment. It explains how to set up the code and how to get every number in our PDF report. The answers and explanations are in the report.

Members: Sorana Gavril, Bianca Vraci, Maria Voicu


## Setup

You need Python 3.11 or newer, because TenSEAL does not work on older versions. We used Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Linux / WSL only: install the CPU version of PyTorch first 
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

The first run downloads the SQuAD dataset and the Snowflake model (about 440 MB) from Hugging Face. 

## What is in this repository

| Part | Folder | Code? |
|---|---|---|
| Q1 Secret sharing | – | No, only in the report |
| Q2 Garbled circuits | – | No, only in the report |
| 3.1 Plaintext retriever | `q3_1_plaintext/` | Yes |
| 3.2 What the server learns | – | No, only in the report |
| 3.3 Encrypted retrieval | `TO DO add the name you pick here` | Yes |
| 3.4 Cost estimation | `TO DO add the name you pick here` | Yes (timings for 3.3(e)) |

Shared folders, used by all parts:

| Folder | What is in it |
|---|---|
| `data/` | The 1,000 documents and 200 questions we use |
| `embeddings/` | The vectors of the documents and questions (768 numbers each) |
| `results/` | The numbers we report |

## 3.1 Plaintext retriever

Run these commands from the main folder of the repository, in this order:


```bash
python q3_1_plaintext/data.py        # pick the 1,000 documents and 200 questions
python q3_1_plaintext/encode.py      # turn them into vectors with the Snowflake model 
python q3_1_plaintext/similarity.py  # test our own cosine similarity and top-k
python q3_1_plaintext/eval_dims.py   # 3.1(a): Recall@10 at 64, 128, 256 and 768 dimensions
python q3_1_plaintext/eval_plain.py  # 3.1(d): Recall@1/5/10 and latency
```

The first two steps are optional, because `data/` and `embeddings/` are already in the repository. If you run them again, you get the same documents and questions (we use a fixed seed). The vectors can differ in the last decimals on another computer, but this should not change the results.


### What each file does

| File | Part | What it does | Output |
|---|---|---|---|
| `data.py` | Data | Takes the first 1,000 unique paragraphs of the SQuAD validation split, then 200 random questions, one per paragraph (seed 6) | `data/docs.json`, `data/queries.json` |
| `encode.py` | 3.1(a) | `encode(texts)`: the [CLS] vector of the model, cut to 256 numbers and normalized. Questions get the query prefix, documents do not | `embeddings/doc_768.npy`, `embeddings/query_768.npy` |
| `similarity.py` | 3.1(c) | Our own `cosine_similarity` and `top_k`, using only NumPy | – |
| `server.py` | 3.1(b) | The `Server` class stores the documents and vectors. The `Client` only talks to it through `search(query_vector, k)` | – |
| `eval_dims.py` | 3.1(a) | Recall@10 for each vector size | `results/recall_by_dim.json` |
| `eval_plain.py` | 3.1(d) | Asks all 200 questions one by one and times each step | `results/plain_eval.json`, `results/plain_top10.json` |

We do not use faiss, sklearn or `torch.topk`. The only place we use `np.argsort` is the test at the bottom of `similarity.py`, to check that our own `top_k` gives the right answer.

### Our results

Recall@10 for each vector size (3.1(a)):

| Dimensions | 64 | 128 | 256 | 768 |
|---|---|---|---|---|
| Recall@10 | 0.820 | 0.940 | 0.960 | 0.975 |

Plaintext retriever at 256 dimensions (3.1(d)):

| Recall@1 | Recall@5 | Recall@10 | Average time per question |
|---|---|---|---|
| 0.755 | 0.935 | 0.960 | 30.7 ms |

The time was measured on an AMD Ryzen 7 8745HX laptop CPU. On another computer the time will be different, but the recall should be the same.

## 3.3 Encrypted retrieval

TODO

## 3.4 Cost estimation

TODO
