---
title: Hybrid Movie Recommender
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "6.0.0"
app_file: app.py
pinned: false
---

<h1 align="center">Hybrid Movie Recommendation System</h1>

<p align="center">
  <i>A Multi-Stage Pipeline Combining Collaborative Filtering, Sequential Models,<br/>and Large Language Models via LambdaRank Meta-Learning</i>
</p>

<br/>

<table align="center">
  <tr>
    <th colspan="4">Ranking Accuracy</th>
    <th colspan="4">Beyond Accuracy</th>
  </tr>
  <tr>
    <td align="center"><b>HR@10</b><br/><code>0.797</code></td>
    <td align="center"><b>NDCG@10</b><br/><code>0.636</code></td>
    <td align="center"><b>HR@20</b><br/><code>0.915</code></td>
    <td align="center"><b>NDCG@20</b><br/><code>0.665</code></td>
    <td align="center"><b>Personalisation</b><br/><code>0.99</code></td>
    <td align="center"><b>Coverage</b><br/><code>78%</code></td>
    <td align="center"><b>Diversity</b><br/><code>11.6</code></td>
    <td align="center"><b>Novelty</b><br/><code>10.93</code></td>
  </tr>
</table>

<br/>

<p align="center">
  <a href="https://huggingface.co/spaces/Nishant7p/Hybrid-Movie-Recommender"><b>Live Demo</b></a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;<a href="./Report.pdf"><b>Technical Report (27 pages)</b></a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;<a href="https://www.youtube.com/watch?v=NHvm4Z_XI8Q"><b>Demo Video</b></a>
</p>

---

<p align="center">
  <img src="https://i.ibb.co/sJNgbqdP/ui-main.jpg" alt="Hybrid Recommender UI" width="100%"/>
</p>

## Overview

This project implements a complete, end-to-end hybrid recommendation pipeline trained on [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) (1M ratings, 6,040 users, 3,952 movies), enriched with [Kaggle 45K movie metadata](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

The system is architected as a **three-stage cascade**:

1. **Candidate Generation** -- Eight independent CF and sequential models produce diverse candidate scores per user.
2. **LLM Semantic Augmentation** -- A LoRA-adapted Gemma model scores each (user, candidate) pair using enriched natural-language metadata, bridging the semantic gap that co-occurrence signals alone cannot capture.
3. **Meta-Learning (Reranking)** -- A LightGBM LambdaRank model trained with Optuna HPO combines all signals into a 13-dimensional feature vector and directly optimises NDCG to produce final ranked lists.

The result is a system that simultaneously achieves **strong ranking accuracy** (HR@10 = 0.797), **near-perfect personalisation** (0.99), and **78% catalog coverage** -- a combination that single-model approaches consistently fail to deliver.

---

## Technical Report

> **The full 27-page technical report is included as [`Report.pdf`](./Report.pdf).**
> It covers the complete pipeline architecture, mathematical formulations for all 8 models,
> feature engineering methodology, LambdaRank design, Optuna HPO, evaluation protocol,
> beyond-accuracy metrics, UI walkthrough, and discussion of results.

<details>
<summary><b>Report Table of Contents</b></summary>

| # | Section | Page |
|---|---------|------|
| 1 | Introduction | 4 |
| 2 | Dataset and Preprocessing | 5 |
| 3 | Pipeline Architecture | 7 |
| 4 | Candidate Generation Models (8 models) | 9 |
| 5 | LLM Feature Generation (LoRA Fine-Tuning) | 11 |
| 6 | Feature Engineering (13-D Feature Vector) | 13 |
| 7 | Meta-Learner: LightGBM LambdaRank | 15 |
| 8 | Evaluation Metrics: Detailed Formulations | 16 |
| 9 | User Interface | 20 |
| 10 | Configuration, Scripts, and Reproducibility | 22 |
| 11 | System Dependencies | 25 |
| 12 | Discussion and Analysis | 25 |
| 13 | Conclusion | 26 |
| | References | 27 |

</details>

---

## Pipeline Architecture

<p align="center">
  <img src="https://i.ibb.co/m5dNSkZw/pipeline-flowchart.png" alt="End-to-end Pipeline Architecture" width="85%"/>
</p>

<p align="center"><i>End-to-end pipeline. Data flows top-to-bottom through six major stages.<br/>Full details with mathematical formulations in <a href="./Report.pdf">Report.pdf</a> (Section 3).</i></p>

### Stage 1 -- Data Preparation

Raw MovieLens ratings are converted to **implicit binary feedback** (positive if rating >= 4, otherwise discarded). A **leave-one-out** temporal split creates train/val/test sets. Each user is evaluated against a frozen pool of **1 positive + 99 randomly sampled negatives**.

### Stage 2 -- Candidate Generation (8 Models)

| Model | Type | Key Idea |
|---|---|---|
| **Popularity** | Non-personalised baseline | Global interaction frequency ranking |
| **ItemKNN** | Neighbourhood CF | Cosine similarity over binary user-item vectors |
| **EASE^R** | Shallow autoencoder | Closed-form item-to-item weight matrix |
| **BPR** | Matrix factorisation | Pairwise ranking loss maximising `P(pos > neg)` |
| **LightGCN** | Graph neural network | Multi-layer graph convolution on bipartite user-item graph |
| **DCN v2** | Deep & Cross Network | Explicit bounded-degree feature interactions + deep layers |
| **NeuMF** | Neural MF | GMF branch + MLP branch concatenated before output |
| **SASRec** | Sequential transformer | Causal self-attention over user interaction history |

### Stage 3 -- LLM Semantic Feature (Optional)

A **Gemma** model is fine-tuned using **LoRA** (only low-rank A/B matrices trained, base weights frozen). The softmax over ("Yes", "No") logits yields a scalar **p_yes(u,i)** -- a semantic compatibility score **orthogonal** to all CF signals.

### Stage 4 -- Feature Engineering

Raw CF scores are **rank-normalised** to a uniform [0, 1] percentile scale. The complete **13-dimensional feature vector**:

| # | Feature | Source |
|---|---------|--------|
| 1-8 | `{bpr, itemknn, ease, pop, lightgcn, dcn, neumf, sasrec}_rank_norm` | Rank-normalised CF scores |
| 9 | `user_interaction_count` | User training history length |
| 10 | `user_avg_item_popularity` | Mean popularity of user's history |
| 11 | `item_interaction_count` | Item global interaction count |
| 12 | `item_popularity_rank` | Item popularity rank in catalog |
| 13 | `llm_yes_prob` | LoRA-Gemma semantic score |

### Stage 5 -- LightGBM LambdaRank Meta-Learner

LambdaRank computes pseudo-gradients that **directly optimise NDCG**. Hyperparameters tuned with Optuna (TPE sampler, 72+ trials, SQLite-persisted).

---

## Results

### Baseline Comparison

| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| Popularity | 0.498 | 0.277 |
| ItemKNN | 0.681 | 0.399 |
| BPR-MF | 0.619 | 0.348 |
| EASE^R | 0.695 | 0.430 |
| LightGCN | 0.498 | 0.276 |
| DCN-v2 | 0.638 | 0.359 |
| NeuMF | 0.684 | 0.394 |
| Hybrid (CF only) | 0.718 | 0.433 |
| **Hybrid (LambdaRank + Optuna)** | **0.797** | **0.636** |

### Beyond-Accuracy Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Genre Diversity (avg @20) | **11.6 genres/list** | High intra-list diversity from ensemble diversity |
| Novelty (avg @20) | **10.93** | Balances mainstream and long-tail content |
| Catalog Coverage (@20) | **78%** | 2,970 of 3,807 items appear in at least one user's top-20 |
| Personalisation (@20) | **0.99** | Near-zero inter-user list overlap across 6,000+ users |

<p align="center">
  <img src="https://i.ibb.co/GgyXQ1M/ui-metrics.jpg" alt="Metrics Dashboard" width="90%"/>
</p>

### Ablation Study

| Dropped Feature | HR@10 | NDCG@10 | Delta |
|----------------|-------|---------|-------|
| None (full) | 0.718 | 0.433 | -- |
| ease | 0.703 | 0.411 | **-0.021** |
| neumf | 0.711 | 0.433 | +0.000 |
| knn | 0.714 | 0.434 | +0.001 |
| dcn | 0.718 | 0.434 | +0.001 |
| lgcn | 0.721 | 0.435 | +0.002 |
| bpr | 0.719 | 0.435 | +0.002 |
| pop | 0.721 | 0.436 | +0.003 |
| llm_yes_prob | 0.720 | 0.436 | +0.004 |

---

## Why This Pipeline Works

**1. Ensemble diversity drives coverage.** BPR and ItemKNN recommend genre-similar items. SASRec and LightGCN surface structurally different candidates. The meta-learner combines these complementary signals.

**2. Rank normalisation is critical.** A BPR dot-product of 1.2 and an EASE prediction of 0.04 are not comparable. Rank normalisation puts all signals on equal footing.

**3. LLM signals are orthogonal.** CF models know *who* likes *what* but not *why*. The LoRA-Gemma score operates on content (plot, genre, metadata) -- the feature with the lowest collinearity to all CF signals.

**4. LambdaRank closes the train/eval gap.** Training a classifier then evaluating with NDCG is misaligned. LambdaRank computes gradients that directly improve NDCG.

---

## Interactive Demo

<p align="center">
  <a href="https://huggingface.co/spaces/Nishant7p/Hybrid-Movie-Recommender"><b>Try the Live Demo</b></a>
</p>

The Gradio app precomputes ranked lists for all 6,035 users. Three panels:

| Panel | Content |
|-------|---------|
| **Left** | User selector + scrollable watch history with genre tags |
| **Centre** | Top-20 recommendations with relevance bars, diversity/novelty cards |
| **Right** | Per-candidate prediction details: score, rank, ground-truth label |

```bash
# Run locally
python app.py
# -> http://localhost:7860
```

---

## Quickstart

```bash
git clone https://github.com/Nishant7p/Hybrid-Movie-Recommender.git
cd Hybrid-Movie-Recommender
uv sync            # or: pip install -r requirements.txt
```

Place raw data under `data/raw/ml-1m/` (`ratings.dat`, `movies.dat`, `users.dat`) and Kaggle metadata under `data/raw/kaggle/`.

### Reproducing Results

```bash
python scripts/prepare_data.py                 # Binarise, split, freeze negatives
CUDA_VISIBLE_DEVICES=0 python scripts/dump_scores.py  # Train CF models + dump scores
python scripts/build_lora_dataset.py           # (Optional) Build LoRA dataset
python scripts/lora_train.py                   # (Optional) Fine-tune Gemma
python scripts/build_llm_features_lora.py      # (Optional) LLM features
python scripts/tune_meta_learner.py            # Optuna HPO for LambdaRank
python scripts/train_final_model.py            # Train final model
python scripts/run_pipeline.py                 # Evaluate
python app.py                                  # Launch Gradio UI
```

Single-command: `bash scripts/rebuild_all.sh` | Fast recovery: `bash scripts/fast_recover.sh`

---

## Repository Structure

```
Hybrid-Movie-Recommender/
|-- Report.pdf                     # 27-page technical report
|-- app.py                         # Gradio UI
|-- src/cf_pipeline/
|   |-- data/                      # Loaders, binarization, splits, negatives
|   |-- models/                    # Popularity, ItemKNN, EASE, BPR, LightGCN, DCN, NeuMF, SASRec
|   |-- llm/                       # LLM server, cold-start, RAG, decision
|   |-- eval/                      # HR@K, NDCG@K, evaluation protocol
|   |-- features.py                # 9-dim feature matrix (CF + LLM)
|   +-- features_enhanced.py       # 13-dim enhanced feature matrix
|-- scripts/                       # All pipeline stages + orchestrators
|-- configs/                       # Hydra YAML configs
|-- results/                       # JSON/markdown result tables + figures
|-- tests/                         # Unit + integration tests
|-- data/processed/                # Pipeline artifacts (parquet, json)
+-- checkpoints/                   # Model checkpoints
```

---

## Evaluation Protocol

All metrics computed over frozen `eval_negatives.json` (1 positive + 99 negatives per user). Standard NCF leave-one-out protocol [He et al., 2017].

| Metric | Description |
|--------|-------------|
| **HR@K** | Did the positive item appear in top-K? |
| **NDCG@K** | Position-sensitive: rewards higher placement |
| **Coverage@K** | Fraction of catalog recommended to anyone |
| **Personalisation@K** | How different are lists across users? |
| **Genre Diversity@K** | Unique genres per recommendation list |
| **Novelty@K** | Inverse popularity of recommended items |

---

## Testing

```bash
pytest                    # Full suite
pytest tests/data/        # Loader + binarisation
pytest tests/eval/        # HR, NDCG, protocol
pytest tests/models/      # Component tests
pytest tests/llm/         # LLM integration tests
```

---

## Tech Stack

| Package | Role |
|---------|------|
| `torch` | Neural CF models, SASRec |
| `lightgbm` | LambdaRank meta-learner |
| `transformers` / `peft` | Gemma + LoRA fine-tuning |
| `optuna` | Hyperparameter optimisation |
| `hydra-core` | Configuration management |
| `pandas` / `pyarrow` | Data manipulation + Parquet I/O |
| `gradio` / `plotly` | Interactive UI |
| `sentence-transformers` / `faiss-cpu` | Dense retrieval |

---

## References

<details>
<summary>BibTeX citations</summary>

```bibtex
@inproceedings{he2017ncf, title={Neural Collaborative Filtering}, author={He et al.}, booktitle={WWW}, year={2017}}
@inproceedings{he2020lightgcn, title={LightGCN}, author={He et al.}, booktitle={SIGIR}, year={2020}}
@inproceedings{rendle2009bpr, title={BPR: Bayesian Personalized Ranking}, author={Rendle et al.}, booktitle={UAI}, year={2009}}
@inproceedings{steck2019ease, title={Embarrassingly Shallow Autoencoders}, author={Steck}, booktitle={WWW}, year={2019}}
@inproceedings{kang2018sasrec, title={Self-Attentive Sequential Recommendation}, author={Kang and McAuley}, booktitle={ICDM}, year={2018}}
@inproceedings{wang2021dcnv2, title={DCN V2}, author={Wang et al.}, booktitle={WWW}, year={2021}}
@techreport{burges2010lambdamart, title={From RankNet to LambdaRank to LambdaMART}, author={Burges}, institution={MSR}, year={2010}}
@inproceedings{hu2022lora, title={LoRA}, author={Hu et al.}, booktitle={ICLR}, year={2022}}
@inproceedings{akiba2019optuna, title={Optuna}, author={Akiba et al.}, booktitle={KDD}, year={2019}}
@article{harper2015movielens, title={The MovieLens Datasets}, author={Harper and Konstan}, journal={ACM TIIS}, year={2015}}
```

</details>

---

## Team

| Name | Roll No. |
|------|----------|
| Nishant Tomer | 2023355 |
| Vinayak Koli | 2023597 |
| Yashasvi | 2023611 |

<p align="center">Built at IIIT-Delhi</p>
