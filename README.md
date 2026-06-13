# AILA_Project
# ⚖️ AILA Legal Retrieval — Unified Pipeline

> An end-to-end system for retrieving relevant **prior cases** and **statutes** from Indian legal documents, powered by a progressively fine-tuned bi-encoder trained with hard negative mining on an A100 GPU.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 0 — Installation & Setup](#stage-0--installation--setup)
  - [Stage 1–3.5 — Case Preprocessing](#stage-13-5--case-preprocessing)
  - [Stage 4 — Query Preprocessing](#stage-4--query-preprocessing)
  - [Stage 5 — Statute Preprocessing](#stage-5--statute-preprocessing)
  - [Stage 6 — Compact Representations](#stage-6--compact-representations)
  - [Stage 7 — BM25 Hard Negative Mining](#stage-7--bm25-hard-negative-mining)
  - [Stage 8 — Bi-Encoder v1](#stage-8--bi-encoder-v1)
  - [Stage 9 — Bi-Encoder v2](#stage-9--bi-encoder-v2)
  - [Stage 10 — Semantic Hard Negative Mining](#stage-10--semantic-hard-negative-mining)
  - [Stage 11 — Bi-Encoder v3 (A100)](#stage-11--bi-encoder-v3-a100)
  - [Stage 12 — Evaluation](#stage-12--evaluation)
- [Supplementary Code](#supplementary-code)
- [Evaluation Metrics](#evaluation-metrics)
- [Dependencies](#dependencies)
- [Hardware](#hardware)

---

## Overview

This project implements the **AILA (AI for Legal Assistance)** legal document retrieval task. Given a query case document, the system retrieves:

1. **Relevant prior cases** — earlier Indian court decisions that the query case should cite
2. **Relevant statutes** — sections of Indian law applicable to the query case

The pipeline follows a **curriculum learning** strategy — starting with lexical hard negatives, then bootstrapping to semantic hard negatives using the model trained in the previous round:

```
Raw Documents
     ↓
Legal NLP Preprocessing (role classification, entity extraction, chunking)
     ↓
Compact Structured Representations
     ↓
BM25 Hard Negative Mining  →  Train Bi-Encoder v1  →  Train Bi-Encoder v2
                                                               ↓
                                          Semantic Hard Negative Mining
                                                               ↓
                                              Fine-tune Bi-Encoder v3 (A100)
                                                               ↓
                                          FAISS Dense Retrieval + Evaluation
```

---

## Pipeline Stages

### Stage 0 — Installation & Setup

Installs required libraries (`sentence-transformers`, `faiss-cpu`, `rank-bm25`, `accelerate`) and fixes global random seeds (`42`) across `random`, `numpy`, and `torch` for reproducibility. The compute device is auto-detected (`cuda` or `cpu`).

---

### Stage 1–3.5 — Case Preprocessing

Each raw `.txt` case file is processed through a **fused multi-stage pipeline** running in parallel via `ThreadPoolExecutor` (up to 2× CPU core count).

#### Text Cleaning
Normalises whitespace, removes form-feed characters and repeated separators, and canonicalises legal references — `Section 302` becomes `SECTION_302`, `Article 14` becomes `ARTICLE_14` — so that section citations are treated as single tokens downstream.

#### Overlapping Chunker
Splits text into sentences on punctuation boundaries, then applies a **sliding window of 6 sentences with a stride of 3** to produce overlapping passage chunks. Overlapping ensures that context spanning sentence boundaries is not lost.

#### Weighted Role Classifier
Each passage is scored against 10 legal semantic roles using a lexicon of weighted trigger phrases. The primary role and a normalised score dictionary are assigned per passage:

| Role | Signal Phrases | Importance Score |
|---|---|---|
| `HEADER` | "Supreme Court", "bench", "judgment was delivered" | 0.10 |
| `FACT` | "background", "filed", "agreement", "entered into" | 0.55 |
| `ISSUE` | "whether", "question", "issue", "controversy" | 0.80 |
| `STATUTE` | `SECTION_*`, `ARTICLE_*`, "Constitution", "Code" | 0.75 |
| `PRECEDENT` | " v. ", "SCC", "AIR", "versus" | 0.70 |
| `ARGUMENT` | "contended", "argued", "learned counsel", "submitted" | 0.50 |
| `PROCEDURE` | "petition", "appeal", "writ", "magistrate" | 0.60 |
| `REASONING` | "therefore", "in our view", "we find", "considering" | 0.90 |
| `HOLDING` | "we hold", "we conclude", "held that", "find no merit" | 0.98 |
| `ORDER` | "appeal dismissed", "appeal allowed", "set aside" | 1.00 |

#### Entity Extraction
For each case document, the pipeline extracts:
- **Acts cited** — IPC, CrPC, CPC, Constitution, etc. via regex
- **Sections cited** — canonicalised `SECTION_*` / `ARTICLE_*` tokens (up to 100)
- **Cases cited** — `Party A v. Party B` pattern matches (up to 100)
- **Metadata** — years mentioned, monetary amounts, courts named, section and citation counts
- **Keywords** — top-25 non-stopword terms by frequency
- **Legal phrases** — top-12 bigrams/trigrams containing domain keywords (act, court, issue, evidence, etc.)

---

### Stage 4 — Query Preprocessing

Applies the identical pipeline (clean → chunk → classify → extract) to every query from `Query_doc.txt`, which uses a `qid || text` format. Each query becomes a structured JSON object with passages, entities, and keywords — the same schema as a processed case.

---

### Stage 5 — Statute Preprocessing

Statutes arrive in a different raw format (`Title:` / `Desc:` fields) and are processed separately:

- **Act detection** — heuristic keyword rules identify the parent act: IPC, CrPC, Constitution, Industrial Disputes Act, Land Acquisition Act, Arbitration Act, or General Law
- **Domain classification** — 7 legal domains detected: constitutional, criminal, procedure, labour, property, arbitration, civil
- **Section reference extraction** — regex pulls the specific Article/Section/Subsection from the statute title
- **Word-level chunking** — overlapping 120-word chunks with a stride of 60 words
- **Keywords & concepts** — top legal terms and bigram/trigram legal phrases

Processing runs in parallel via `ThreadPoolExecutor` across all statute files.

---

### Stage 6 — Compact Representations

Transforms the rich JSON structures into **flat structured text** strings optimised for embedding. Each document becomes a labelled field string:

```
ACTS: IPC, Constitution of India
SECTIONS: SECTION_302, ARTICLE_21
CASES: State of Maharashtra v. Mayer Hans George, ...
KEYWORDS: conviction, evidence, appeal, burden, right, ...
ISSUE: Whether the accused was liable under Section 302 ...
HOLDING: We hold that the conviction is upheld ...
ORDER: Appeal dismissed.
```

For cases, passages are first **sorted by importance score** (descending). High-importance roles (`ISSUE`, `STATUTE`, `PRECEDENT`, `REASONING`, `HOLDING`, `ORDER`) are deduplicated — only the top-scoring passage per role is kept — and up to 6 passages are selected. If high-importance passages are sparse, secondary roles (`FACT`, `ARGUMENT`, `PROCEDURE`) are added to fill the budget.

For statutes, the compact string leads with structured metadata (act, section, domain) followed by the first chunk of text.

Three compact maps are produced: `{qid → text}`, `{doc_id → text}`, `{sid → text}`.

---

### Stage 7 — BM25 Hard Negative Mining

Uses `BM25Okapi` to mine **lexically confusable but irrelevant** documents as training negatives:

1. All documents are regex-tokenised and indexed with BM25
2. For each query with ground-truth relevant documents:
   - Top-150 BM25-ranked documents are retrieved
   - Ground-truth positives are excluded
   - Up to **25 hard negatives** are kept (high BM25 score, label = 0)
3. Two data structures are generated per retrieval type:
   - **Triplets** `(query, positive, hard_negative)` — for `MultipleNegativesRankingLoss`
   - **Labelled pairs** `(query, doc, label∈{0,1})` — for pair classifiers or analysis

Run independently for **cases** and **statutes**, producing 4 JSON files total.

---

### Stage 8 — Bi-Encoder v1

Fine-tunes [`law-ai/InLegalBERT`](https://huggingface.co/law-ai/InLegalBERT) — a BERT model pre-trained on Indian legal corpora — on the BM25 triplets using `MultipleNegativesRankingLoss`.

| Hyperparameter | Value |
|---|---|
| Base model | `law-ai/InLegalBERT` |
| Loss | `MultipleNegativesRankingLoss` (MNRL) |
| Batch size | 16 |
| Epochs | 5 |
| Learning rate | 2e-5 |
| Warmup | 10% of total steps |
| Max sequence length | 512 tokens |
| Mixed precision | AMP (fp16) |
| Checkpoints | Every 200 steps |

Training examples are **deduplicated** on the first 200 characters of (query, positive) pairs before shuffling, to avoid repeated pulls of the same document with different negatives inflating the training set.

---

### Stage 9 — Bi-Encoder v2

Continues training from `law-ai/InLegalBERT` (same base, not fine-tuned v1) on the full BM25 triplet dataset — no deduplication filter — with an increased batch size for stronger contrastive signal:

| Change from v1 | Value |
|---|---|
| Batch size | 32 (doubled) |
| DataLoader workers | 4 with `persistent_workers=True` |
| Deduplication | Disabled (full dataset) |

Larger in-batch negatives in MNRL improve the quality of the contrastive objective. This model is then used as the encoder for semantic hard negative mining in Stage 10.

---

### Stage 10 — Semantic Hard Negative Mining

Uses **bi_encoder_v2** to find documents that are semantically close to the query but not ground-truth relevant — substantially harder negatives than BM25 can produce:

1. All cases and statutes are encoded with bi_encoder_v2 (batch size 128, L2-normalised to unit sphere)
2. A `FAISS IndexFlatIP` (exact cosine similarity) index is built for each corpus
3. All queries are encoded and searched against each index (top-100 neighbors)
4. Ground-truth positives are excluded; up to **15 semantic hard negatives** are kept per query
5. Triplets are generated with up to **5 negatives per positive**

This two-corpus mining (cases + statutes) produces 4 JSON files of semantic triplets and pairs.

---

### Stage 11 — Bi-Encoder v3 (A100)

The final model, fine-tuned from **bi_encoder_v2** on the **semantic hard negatives** using `CachedMultipleNegativesRankingLoss` — a gradient-cached variant of MNRL that enables very large effective batch sizes on a single GPU without OOM.

| Hyperparameter | Value |
|---|---|
| Base model | `bi_encoder_v2` (warm start) |
| Loss | `CachedMultipleNegativesRankingLoss` |
| Effective batch size | 192 |
| Gradient cache mini-batch | 48 |
| Epochs | 4 |
| Learning rate | 1e-5 |
| Warmup | 10% of total steps |
| Workers | 8, `persistent_workers=True` |
| Mixed precision | AMP (fp16) |
| Checkpoints | Every 25 steps |

The large batch size (192) is essential — MNRL treats all other items in the batch as implicit negatives, so a batch of 192 provides 191 negatives per anchor. `CachedMNRL` achieves this via gradient checkpointing across mini-batches, making it feasible on a single A100 80 GB.

---

### Stage 12 — Evaluation

Evaluates **bi_encoder_v3_a100** on the official AILA relevance judgments using FAISS dense retrieval at k=10. The process mirrors the mining setup: encode all documents, build `IndexFlatIP`, encode all queries, retrieve top-10 per query.

All 6 metrics are computed per query and then **macro-averaged** across all queries that have at least one ground-truth relevant document. Evaluation is run separately for **Case Retrieval** and **Statute Retrieval**.

---

## Supplementary Code

### Section A — Loss Curve Tracking

Monkey-patches the loss function's `forward()` method to intercept and log per-step loss values during training without modifying the `sentence-transformers` library. Loss logs are saved as `loss_log.json` alongside each model. A plotting utility reads these logs and renders smoothed loss curves (rolling-window average) for v1, v2, and v3 side-by-side.

### Section B — Baseline Comparison Evaluation

Runs a full **4-way comparison** across both retrieval tracks:

- **BM25** — lexical baseline using `BM25Okapi`
- **bi_encoder_v1** — MNRL, BM25 negatives, bs=16
- **bi_encoder_v2** — MNRL, BM25 negatives, bs=32
- **bi_encoder_v3** — CachedMNRL, semantic negatives, bs=192

All 6 metrics are computed for each model on both Case and Statute retrieval. Two bar charts are generated: a full 6-metric comparison and a ranking-focused chart (MAP, MRR, nDCG@10) for reporting.

---

## Results

All models evaluated on 50 queries against 2,914 cases and 197 statutes using FAISS dense retrieval at k=10. Metrics are macro-averaged across all queries.

### Case Retrieval

| Model | Precision@10 | Recall@10 | F1@10 | MAP | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| BM25 (baseline) | 0.022 | 0.0787 | 0.0290 | 0.0382 | 0.0886 | 0.0592 |
| Bi-Encoder v1 | 0.098 | 0.3095 | 0.1358 | 0.1774 | 0.3087 | 0.2497 |
| Bi-Encoder v2 | 0.308 | 0.9243 | 0.4149 | 0.8603 | 0.8719 | 0.8943 |
| **Bi-Encoder v3** | **0.312** | **0.9301** | **0.4196** | **0.8694** | **0.8922** | **0.9036** |

### Statute Retrieval

| Model | Precision@10 | Recall@10 | F1@10 | MAP | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| BM25 (baseline) | 0.044 | 0.1013 | 0.0610 | 0.0428 | 0.1608 | 0.0874 |
| Bi-Encoder v1 | 0.298 | 0.6747 | 0.4110 | 0.4282 | 0.7050 | 0.5863 |
| Bi-Encoder v2 | 0.428 | 0.9647 | 0.5900 | 0.8801 | 0.9467 | 0.9300 |
| **Bi-Encoder v3** | **0.430** | **0.9687** | **0.5927** | **0.8880** | **0.9550** | **0.9360** |

### Key Takeaways

- **BM25 → v1**: The jump from lexical to dense retrieval is dramatic — nDCG@10 improves from 0.059 to 0.250 on cases and 0.087 to 0.586 on statutes, purely from switching to InLegalBERT fine-tuned on BM25 hard negatives.
- **v1 → v2**: Doubling the batch size (16 → 32) delivers the largest single gain in the pipeline — MAP on cases leaps from 0.177 to 0.860, confirming that larger in-batch negatives are the primary driver of MNRL quality.
- **v2 → v3**: Semantic hard negative mining with CachedMNRL provides consistent incremental gains across all metrics, pushing nDCG@10 to **0.9036 (cases)** and **0.9360 (statutes)** — the ceiling of what this architecture and corpus size support.

---

## Evaluation Metrics

All metrics are computed at **k=10** and macro-averaged across queries.

| Metric | Formula | What it measures |
|---|---|---|
| **Precision@10** | `\|retrieved ∩ relevant\| / 10` | How many of the top 10 retrieved are relevant |
| **Recall@10** | `\|retrieved ∩ relevant\| / \|relevant\|` | What fraction of all relevant docs appear in the top 10 |
| **F1@10** | `2 × P × R / (P + R)` | Harmonic balance between precision and recall |
| **MAP** | Mean of per-query Average Precision | Rewards ranking relevant docs higher across all recall levels |
| **MRR** | Mean of `1 / rank_of_first_relevant` | How early the first relevant result appears |
| **nDCG@10** | `DCG@10 / IDCG@10` | Graded relevance with logarithmic position discount |

---

## Dependencies

```
sentence-transformers    # Bi-encoder training & inference
faiss-cpu                # Exact nearest neighbor search (IndexFlatIP)
rank-bm25                # BM25Okapi lexical retrieval
accelerate               # Mixed-precision training support
torch                    # PyTorch backend
numpy
matplotlib
tqdm
```

Install via the first notebook cell, or manually:
```bash
pip install sentence-transformers accelerate faiss-cpu rank-bm25
```

---

## Hardware

| Component | Specification |
|---|---|
| **GPU** | NVIDIA A100 (80 GB VRAM) |
| **v1 / v2 training** | MNRL, AMP fp16, bs 16–32 |
| **v3 training** | CachedMNRL, AMP fp16, bs 192, mini-batch 48 |
| **Encoding** | Batch size 128, L2-normalised embeddings |
| **Retrieval index** | `faiss.IndexFlatIP` (exact cosine search) |

> The pipeline is device-agnostic and falls back to CPU automatically, though training at these batch sizes will be significantly slower without an A100-class GPU.

---

*AILA Legal Retrieval Pipeline — Built for the AI for Legal Assistance shared task on Indian legal document retrieval.*
