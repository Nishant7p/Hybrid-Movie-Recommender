<p align="center">
  <img src="https://i.ibb.co/sJNgbqdP/ui-main.jpg" alt="Hybrid Movie Recommender System" width="100%"/>
</p>

<h1 align="center">Hybrid Movie Recommendation System</h1>

<h3 align="center">A Multi-Stage Pipeline Combining Collaborative Filtering, Sequential Models,<br/>and Large Language Models via LambdaRank Meta-Learning</h3>

<p align="center">
  <img src="https://img.shields.io/badge/HR%4010-0.797-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NDCG%4010-0.636-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/HR%4020-0.915-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NDCG%4020-0.665-green?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Personalisation-0.99-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Catalog%20Coverage-78%25-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Genre%20Diversity-11.6-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Novelty-10.93-teal?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-%E2%89%A53.11-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Models-8%20CF%20%2B%20LLM-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square"/>
</p>

<p align="center">
  <a href="./Report.pdf"><img src="https://img.shields.io/badge/Technical%20Report-27%20Page%20PDF-blue?style=for-the-badge&logo=adobeacrobatreader&logoColor=white"/></a>
  &nbsp;&nbsp;
  <a href="https://www.youtube.com/watch?v=NHvm4Z_XI8Q"><img src="https://img.shields.io/badge/Demo%20Video-YouTube-red?style=for-the-badge&logo=youtube&logoColor=white"/></a>
</p>

---

## Technical Report

> **The full 27-page technical report is included in this repository as [`Report.pdf`](./Report.pdf).**
> It covers the complete pipeline architecture, mathematical formulations for all 8 models,
> feature engineering methodology, LambdaRank meta-learner design, Optuna HPO configuration,
> evaluation protocol, beyond-accuracy metrics, UI walkthrough, and discussion of results.

<details>
<summary><b>Report Table of Contents (click to expand)</b></summary>

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

## Abstract

This project presents a comprehensive, end-to-end **Hybrid Movie Recommendation System** built on the [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) benchmark (1M ratings, 6,040 users, 3,952 movies) and enriched with the [Kaggle 45K movie metadata](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) dataset. Rather than relying on a single model, we design a **multi-stage pipeline** that:

1. **Preprocesses** raw explicit ratings into implicit binary feedback via binarization
2. **Generates** diverse candidate lists using **8 independent CF and sequential models**
3. **Augments** candidates with semantic signals from a **LoRA-fine-tuned Gemma LLM**
4. **Engineers** a **13-dimensional feature vector** per (user, candidate) pair with rank normalisation
5. **Trains** a **LightGBM LambdaRank meta-learner** -- tuned with Optuna -- to produce final rankings

The system achieves **HR@10 = 0.797** and **NDCG@10 = 0.636** under the standard NCF leave-one-out protocol, while simultaneously delivering **78% catalog coverage** and **0.99 inter-user personalisation** -- a combination that single-model approaches consistently fail to deliver.

---

## Pipeline Architecture

<p align="center">
  <img src="https://i.ibb.co/m5dNSkZw/pipeline-flowchart.png" alt="End-to-end Pipeline Architecture" width="85%"/>
</p>

<p align="center"><i>Figure 1: End-to-end pipeline architecture. Data flows top-to-bottom through six major stages.<br/>Full architectural details with mathematical formulations are in <a href="./Report.pdf">Report.pdf</a> (Section 3, p.7).</i></p>

### Stage 1 -- Data Preparation

Raw MovieLens ratings are converted to **implicit binary feedback** (positive if rating >= 4, otherwise discarded). A **leave-one-out** temporal split creates train/val/test sets. Each user is evaluated against a frozen pool of **1 positive + 99 randomly sampled negatives** for fully reproducible metric computation.

### Stage 2 -- Candidate Generation (8 Models)

All models run independently and produce per-user candidate scores. No model has visibility into another's output at this stage.

| Model | Type | Key Idea |
|---|---|---|
| **Popularity** | Non-personalised baseline | Global interaction frequency ranking |
| **ItemKNN** | Neighbourhood CF | Cosine similarity over binary user-item vectors |
| **EASE^R** | Shallow autoencoder | Closed-form item-to-item weight matrix `B = I - P * diag(1/diag(P))` |
| **BPR** | Matrix factorisation | Pairwise ranking loss maximising `P(positive > negative)` |
| **LightGCN** | Graph neural network | Multi-layer graph convolution on bipartite user-item graph |
| **DCN v2** | Deep & Cross Network | Explicit bounded-degree feature interactions + deep layers |
| **NeuMF** | Neural MF | GMF branch + MLP branch concatenated before output |
| **SASRec** | Sequential transformer | Causal self-attention over user interaction history |

### Stage 3 -- LLM Semantic Feature Extraction (Optional)

A **Gemma** base model is fine-tuned using **LoRA** (Low-Rank Adaptation -- only A/B decomposition matrices trained, base weights frozen). During inference, the softmax over ("Yes", "No") logits yields a scalar **p_yes(u,i)** -- a semantic compatibility score that is **orthogonal** to all co-occurrence-based CF signals.

### Stage 4 -- Feature Engineering & Rank Normalisation

Raw CF scores live on incomparable scales. We use **rank normalisation** to map each model's output to a uniform percentile scale:

```
s_norm(u, i) = (|C_u| - rank(u, i) + 1) / |C_u|   in (0, 1]
```

The complete **13-dimensional feature vector** per (user, candidate) pair:

| # | Feature | Source |
|---|---------|--------|
| 1-8 | `{bpr, itemknn, ease, pop, lightgcn, dcn, neumf, sasrec}_rank_norm` | Rank-normalised CF scores |
| 9 | `user_interaction_count` | User training history length |
| 10 | `user_avg_item_popularity` | Mean popularity of user's watched items |
| 11 | `item_interaction_count` | Item's global training interaction count |
| 12 | `item_popularity_rank` | Item's rank in global popularity ordering |
| 13 | `llm_yes_prob` | LoRA-Gemma semantic compatibility score |

### Stage 5 -- LightGBM LambdaRank Meta-Learner

**LambdaRank** computes pseudo-gradients that directly optimise NDCG:

```
lambda_ij = |delta_NDCG_ij| / (1 + exp(y_hat_ui - y_hat_uj))
```

**Hyperparameter optimisation** uses Optuna with TPE over 8 parameters. The study is persisted in SQLite and can be paused/resumed without re-running completed trials.

---

## Results

### Ranking Accuracy (Tuned LambdaRank Pipeline)

| Metric | @1 | @5 | @10 | @20 |
|--------|------|------|---------|------|
| **HR** | 0.530 | 0.618 | **0.797** | 0.915 |
| **NDCG** | 0.530 | 0.574 | **0.636** | 0.665 |

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
| Genre Diversity (avg @20) | **11.6 unique genres/list** | High intra-list diversity from ensemble model diversity |
| Novelty (avg @20) | **10.93** | Balances mainstream and long-tail; rank normalisation prevents popularity dominance |
| Catalog Coverage (@20) | **78%** | 2,970 of 3,807 catalog items appear in at least one user's top-20 |
| Personalisation (@20) | **0.99** | Near-zero inter-user list overlap across 6,000+ users |

<p align="center">
  <img src="https://i.ibb.co/GgyXQ1M/ui-metrics.jpg" alt="Metrics Dashboard" width="90%"/>
</p>

<p align="center"><i>Figure 2: Live metrics dashboard in the Gradio UI. Diversity, novelty, coverage, and personalisation cards alongside the HR/NDCG@K curve.</i></p>

### Ablation Study -- Feature Contribution

| Dropped Feature | HR@10 | NDCG@10 | Delta NDCG@10 |
|----------------|-------|---------|---------------|
| None (full model) | 0.718 | 0.433 | -- |
| ease | 0.703 | 0.411 | **-0.021** |
| neumf | 0.711 | 0.433 | +0.000 |
| knn | 0.714 | 0.434 | +0.001 |
| dcn | 0.718 | 0.434 | +0.001 |
| lgcn | 0.721 | 0.435 | +0.002 |
| bpr | 0.719 | 0.435 | +0.002 |
| pop | 0.721 | 0.436 | +0.003 |
| llm_yes_prob | 0.720 | 0.436 | +0.004 |

> EASE^R is the single most impactful feature -- removing it causes the largest NDCG@10 drop (-2.1%).

---

## Why This Pipeline Works

**1. Ensemble diversity is the primary driver of coverage.**
BPR and ItemKNN tend to recommend genre-similar items. SASRec and LightGCN surface structurally different candidates based on sequential patterns and graph topology. The meta-learner learns to combine these complementary signals.

**2. Rank normalisation is non-negotiable.**
A BPR dot-product of 1.2 and an EASE prediction of 0.04 are not comparable. Normalising by rank ensures all CF signals contribute on equal footing regardless of score magnitude.

**3. LLM signals are orthogonal, not redundant.**
CF models exploit co-occurrence patterns -- they know *who* tends to like *what*, but not *why*. The LoRA-Gemma semantic score operates on item content (plot, genre, metadata) and is the feature with the lowest collinearity with all CF signals.

**4. LambdaRank directly optimises what you measure.**
Training a classifier and then evaluating with NDCG creates a train/eval gap. LambdaRank eliminates this by computing gradients that maximally improve NDCG.

---

## Interactive Demo (Gradio UI)

<p align="center">
  <img src="https://i.ibb.co/sJNgbqdP/ui-main.jpg" alt="Gradio UI" width="90%"/>
</p>

The interactive demo (`app.py`) precomputes and caches ranked lists for all 6,035 users at startup. It exposes three panels:

| Panel | Content |
|-------|---------|
| **Left** | User selector dropdown + scrollable watch history with year and genre tags |
| **Centre** | Top-20 recommendations with relevance score bars, genre tags, and live diversity/novelty cards |
| **Right** | Per-(user, candidate) prediction details: relevance score, rank, feature breakdown, ground-truth label |

An HR/NDCG@K line chart updates dynamically and shows the accuracy/position tradeoff across cutoffs K = {1, 5, 10, 20}.

```bash
python app.py
# -> http://localhost:7860
```

---

## Quickstart

### Requirements

```
Python >= 3.11
torch, lightgbm, transformers, peft, accelerate
optuna, hydra-core, omegaconf
pandas, pyarrow, numpy, scipy, scikit-learn
sentence-transformers, faiss-cpu, rank-bm25
gradio, plotly, matplotlib
pytest
```

Install with [uv](https://github.com/astral-sh/uv) (recommended):

```bash
git clone https://github.com/Nishant7p/Hybrid-Movie-Recommender.git
cd Hybrid-Movie-Recommender
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

### Data Setup

Place the raw MovieLens-1M files under `data/raw/ml-1m/`:

```
data/raw/ml-1m/
  |- ratings.dat
  |- movies.dat
  |- users.dat
```

The Kaggle 45K enrichment dataset (`movies_metadata.csv`) goes under `data/raw/kaggle/`.

---

## Reproducing Results

All steps are orchestrated through individual scripts. Each stage writes canonical artifacts to `data/processed/`, so any stage can be rerun independently.

```bash
# 1. Binarise ratings, leave-one-out split, freeze 99 negatives per user
python scripts/prepare_data.py

# 2. Train all CF models and dump candidate scores
CUDA_VISIBLE_DEVICES=0 python scripts/dump_scores.py

# 3. (Optional) Build LoRA fine-tuning dataset from user histories + metadata
python scripts/build_lora_dataset.py

# 4. (Optional) Fine-tune Gemma adapter
python scripts/lora_train.py

# 5. (Optional) Generate LLM yes-probability features
python scripts/build_llm_features_lora.py

# 6. Hyperparameter optimisation for LightGBM LambdaRank (resumable)
python scripts/tune_meta_learner.py

# 7. Train final model on train + val with best Optuna parameters
python scripts/train_final_model.py

# 8. Evaluate and write results/{hybrid_pipeline,ablation}.json
python scripts/run_pipeline.py

# 9. Launch the interactive Gradio UI on port 7860
python app.py
```

For a single-command rebuild from scratch:

```bash
bash scripts/rebuild_all.sh
```

For fast recovery (skipping already-complete stages based on artifact presence):

```bash
bash scripts/fast_recover.sh
```

---

## Repository Structure

```
Hybrid-Movie-Recommender/
|
|-- Report.pdf                        # 27-page technical report (full methodology + results)
|-- app.py                            # Gradio UI entry point
|
|-- src/cf_pipeline/
|   |-- data/
|   |   |-- loaders.py                # MovieLens + Kaggle data loaders
|   |   |-- binarize.py               # binarize_ratings(df, threshold=4)
|   |   |-- splits.py                 # Leave-one-out temporal splitting
|   |   |-- join_tmdb.py              # TMDB metadata enrichment
|   |   +-- negatives.py              # Frozen negative sampling
|   |-- models/
|   |   |-- baselines.py              # Popularity, ItemKNN
|   |   |-- ease.py                   # EASE^R (closed-form)
|   |   |-- bpr_mf.py                 # BPR matrix factorisation
|   |   |-- lightgcn.py               # LightGCN (graph convolution)
|   |   |-- dcn.py                    # DCN v2 (deep & cross)
|   |   |-- neumf.py                  # Neural Matrix Factorisation
|   |   +-- sasrec.py                 # Self-Attentive Sequential Rec
|   |-- llm/
|   |   |-- server.py                 # LLM inference wrapper
|   |   |-- cold_start.py             # Cold-start user profiling
|   |   |-- rag.py                    # FAISS + BM25 retrieval
|   |   +-- decision.py              # YES/NO decision + logprob extraction
|   |-- eval/
|   |   |-- metrics.py                # HR@K, NDCG@K (vectorised)
|   |   +-- protocol.py              # Leave-one-out evaluation protocol
|   |-- features.py                   # 9-dim feature matrix (CF + LLM)
|   +-- features_enhanced.py          # 13-dim enhanced feature matrix
|
|-- scripts/
|   |-- prepare_data.py               # Stage 1: preprocessing
|   |-- dump_scores.py                # Stage 2: all CF model scores
|   |-- build_lora_dataset.py         # Stage 3a: LoRA training data
|   |-- lora_train.py                 # Stage 3b: Gemma LoRA adapter
|   |-- build_llm_features_lora.py    # Stage 3c: LLM inference -> parquet
|   |-- tune_meta_learner.py          # Stage 4a: Optuna HPO
|   |-- train_final_model.py          # Stage 4b: final LightGBM model
|   |-- train_meta_learner.py         # Meta-learner training
|   |-- run_pipeline.py               # Stage 5: evaluation
|   |-- eval.py                       # Hydra-based evaluation
|   |-- generate_tables.py            # Auto-generate result tables
|   |-- ablation_runner.py            # Ablation study automation
|   |-- cold_user_eval.py             # Cold-user sub-population eval
|   |-- metrics_postproc.py           # Diversity, novelty, coverage computation
|   |-- rebuild_all.sh                # Full pipeline orchestrator
|   +-- fast_recover.sh               # Skip completed stages
|
|-- configs/
|   |-- config.yaml                   # Hydra master config
|   |-- data/ml1m.yaml                # Binarisation threshold, split params
|   |-- eval/ncf_protocol.yaml        # Leave-one-out, K cutoffs, 99 negatives
|   |-- model/*.yaml                  # Per-model hyperparameters
|   +-- experiment/*.yaml             # Experiment configurations
|
|-- results/
|   |-- hybrid_pipeline.json          # Final HR/NDCG results
|   |-- tuned_pipeline.json           # Optuna-tuned pipeline results
|   |-- ablation.json                 # Ablation study results
|   |-- best_params.json              # Best Optuna hyperparameters
|   |-- table1_baselines.md           # Baseline comparison table
|   |-- table4_cold_users.md          # Cold-user analysis
|   |-- table5_ablation.md            # Ablation table
|   +-- figures/                      # Diversity vs NDCG plots, novelty histograms
|
|-- tests/
|   |-- data/                         # Loader + binarisation + split tests
|   |-- eval/                         # HR, NDCG, protocol tests
|   |-- models/                       # NeuMF, SASRec, LightGCN, EASE, BPR tests
|   +-- llm/                          # LLM server, RAG, decision tests
|
|-- data/
|   |-- raw/                          # Input data (not tracked by git)
|   +-- processed/                    # Pipeline artifacts (parquet, json)
|
|-- checkpoints/                      # Model checkpoints (gitignored)
+-- docs/superpowers/                 # Design specifications and build plans
```

---

## Processed Artifacts

| File | Description |
|------|-------------|
| `train.parquet` | Implicit positive interactions for training |
| `val.parquet` | Second-to-last interaction per user (validation) |
| `test.parquet` | Last interaction per user (held-out test) |
| `items_metadata.parquet` | Title, genres, TMDB links, Kaggle enrichment |
| `eval_negatives.json` | Frozen 99 negatives + 1 positive per user per split |
| `cf_scores_*.parquet` | Per-model candidate scores (one file per model) |
| `llm_features.parquet` | LLM `yes_prob` per (user, candidate) pair |
| `lora_train.jsonl` | LoRA fine-tuning prompt/response pairs |

---

## Evaluation Protocol

All metrics are computed over the frozen `eval_negatives.json` pool (1 positive + 99 negatives per user). This is the standard NCF leave-one-out protocol from [He et al., 2017].

| Metric | Formula | Description |
|--------|---------|-------------|
| **HR@K** | `mean_u [ i+_u in Top-K ]` | Binary: did the positive item appear in top-K? |
| **NDCG@K** | `mean_u [ 1/log2(rank(i+) + 1) ]` | Position-sensitive: rewards higher placement |
| **Catalog Coverage@K** | `|union_u Top-K(u)| / |I|` | Fraction of catalog recommended to anyone |
| **Personalisation@K** | `1 - avg pairwise list overlap` | How different are lists across users? |
| **Genre Diversity@K** | `mean_u |union genres in L_u|` | Unique genres per recommendation list |
| **Novelty@K** | `mean_u mean_i [-log2 p(i)]` | Inverse popularity of recommended items |

---

## Optuna Search Space

| Parameter | Range | Distribution |
|-----------|-------|-------------|
| `learning_rate` | [0.005, 0.3] | log-uniform |
| `num_leaves` | [20, 300] | integer |
| `max_depth` | [-1, 12] | integer |
| `min_child_samples` | [5, 100] | integer |
| `lambda_l1` | [0, 5] | uniform |
| `lambda_l2` | [0, 5] | uniform |
| `feature_fraction` | [0.5, 1.0] | uniform |
| `bagging_fraction` | [0.5, 1.0] | uniform |

Best trial: **#72** with validation NDCG = **0.718**. Study persisted in `checkpoints/optuna_study.db` (SQLite).

---

## Testing

```bash
pytest                       # Full suite
pytest tests/data/           # Loader + binarisation
pytest tests/eval/           # HR, NDCG, protocol
pytest tests/models/         # Component tests
pytest tests/llm/            # LLM integration tests
```

---

## Tech Stack

| Package | Role | Pipeline Stage |
|---------|------|----------------|
| `torch` | Neural CF models, SASRec | Candidate generation |
| `lightgbm` | LambdaRank meta-learner | Meta-learner |
| `transformers` | Gemma base model | LLM features |
| `peft` | LoRA fine-tuning | LLM features |
| `accelerate` | Distributed training | LLM features |
| `optuna` | Hyperparameter optimisation | Meta-learner tuning |
| `hydra-core` / `omegaconf` | Configuration management | All stages |
| `sentence-transformers` | Dense retrieval embeddings | LLM/RAG |
| `faiss-cpu` | Approximate nearest neighbours | LLM/RAG |
| `pandas` / `pyarrow` | Data manipulation + Parquet I/O | All stages |
| `gradio` / `plotly` | Interactive UI + charts | UI |
| `pytest` | Testing | Tests |

---

## References

```bibtex
@inproceedings{he2017ncf,
  title     = {Neural Collaborative Filtering},
  author    = {He, Xiangnan and Liao, Lizi and Zhang, Hanwang and Nie, Liqiang and Hu, Xia and Chua, Tat-Seng},
  booktitle = {WWW}, year = {2017}
}
@inproceedings{he2020lightgcn,
  title     = {LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation},
  author    = {He, Xiangnan and Deng, Kuan and Wang, Xiang and Li, Yan and Zhang, Yongdong and Wang, Meng},
  booktitle = {SIGIR}, year = {2020}
}
@inproceedings{rendle2009bpr,
  title     = {BPR: Bayesian Personalized Ranking from Implicit Feedback},
  author    = {Rendle, Steffen and Freudenthaler, Christoph and Gantner, Zeno and Schmidt-Thieme, Lars},
  booktitle = {UAI}, year = {2009}
}
@inproceedings{steck2019ease,
  title     = {Embarrassingly Shallow Autoencoders for Sparse Data},
  author    = {Steck, Harald},
  booktitle = {WWW}, year = {2019}
}
@inproceedings{kang2018sasrec,
  title     = {Self-Attentive Sequential Recommendation},
  author    = {Kang, Wang-Cheng and McAuley, Julian},
  booktitle = {ICDM}, year = {2018}
}
@inproceedings{wang2021dcnv2,
  title     = {DCN V2: Improved Deep \& Cross Network},
  author    = {Wang, Ruoxi and Shivanna, Rakesh and Cheng, Derek and Jain, Sagar and Lin, Dong and Hong, Lichan and Chi, Ed},
  booktitle = {WWW}, year = {2021}
}
@techreport{burges2010lambdamart,
  title       = {From RankNet to LambdaRank to LambdaMART: An Overview},
  author      = {Burges, Christopher J.C.},
  institution = {Microsoft Research}, year = {2010}
}
@inproceedings{hu2022lora,
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Chen, Weizhu},
  booktitle = {ICLR}, year = {2022}
}
@inproceedings{akiba2019optuna,
  title     = {Optuna: A Next-generation Hyperparameter Optimization Framework},
  author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle = {KDD}, year = {2019}
}
@article{harper2015movielens,
  title   = {The MovieLens Datasets: History and Context},
  author  = {Harper, F. Maxwell and Konstan, Joseph A.},
  journal = {ACM Transactions on Interactive Intelligent Systems},
  volume  = {5}, number = {4}, year = {2015}
}
```

---

## Team

| Name | Roll No. |
|------|----------|
| Nishant Tomer | 2023355 |
| Vinayak Koli | 2023597 |
| Yashasvi | 2023611 |

<p align="center">
  Built at IIIT-Delhi
</p>
