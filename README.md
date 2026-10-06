# GeoTrust-RAG: Trustworthy Retrieval-Augmented Vision-Language Reasoning for Interactive Disaster Response in Earth Observation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![IEEE J-STARS](https://img.shields.io/badge/IEEE-J--STARS-blue.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4609443)

Official implementation of **GeoTrust-RAG**, a trustworthy retrieval-augmented vision-language framework designed for evidence-grounded remote-sensing reasoning and emergency disaster response.

---

## 📌 Overview

Contemporary Vision-Language Models (VLMs) deployed in Earth Observation (EO) suffer from parametric hallucinations, lack access to authoritative geospatial ontologies, and display pathological overconfidence even under optical occlusions, sensor noise, or unanswerable queries. In safety-critical disaster relief (e.g., FEMA flood evacuation or urban search-and-rescue), an overconfident false declaration can jeopardize human lives.

**GeoTrust-RAG** solves this crisis through three core innovations:
1. **Context-Aware Geospatial RAG (Geo-RAG)**: Dynamically retrieves authoritative domain knowledge from structured geospatial specifications (FEMA damage criteria, FAO LCCS ontologies, USGS sensor standards, OpenStreetMap schemas) using dense vector representations modulated by continuous Ground Sampling Distance (GSD) scale embeddings.
2. **Epistemic Trustworthiness Engine**: Quantifies predictive uncertainty via length-normalized multi-sample semantic entropy and evidence-conformity scoring, establishing a calibrated selective prediction mechanism with an explicit **reject option** ($\varnothing_{\text{abstain}}$).
3. **Dual-Modal Grounded Attribution**: Simultaneously generates localized spatial bounding coordinates $[y_1, x_1, y_2, x_2]$ and verifiable document citation tags $[\text{Doc-}k]$ justifying every deduction.

---

## 🏗️ System Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                    INPUT MULTIMODAL QUERY                                          |
|  [Remote Sensing Image I (Satellite / UAV Drone)]  +  [Instruction Q]  +  [Spatial Scale Metadata M] |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                           STAGE 1: MULTIMODAL CONTEXT PROJECTION                                    |
|  q_mm = alpha * e_Q + (1 - alpha) * W_v * v_global + W_m * m_GSD                                   |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                        STAGE 2: CONTEXT-AWARE GEOSPATIAL RETRIEVAL (Geo-RAG)                       |
|  - FAISS Dense Vector Search over Curated Geospatial Corpus (FEMA, FAO LCCS, USGS, OSM)            |
|  - Cross-Encoder Re-Ranking & Reciprocal Rank Fusion (RRF) -> Top-k Evidence Chunks {D_1, ..., D_k} |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                       STAGE 3: VISION-LANGUAGE REASONING CORE                                      |
|  - Topic-Guided Dynamic Prompt Synthesis: P = [ Q || "Topic: " || Topic(D_1) ]                     |
|  - Monte Carlo Multi-Sample Generation: Y = {y^(1), ..., y^(M)}                                     |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                         STAGE 4: EPISTEMIC TRUSTWORTHINESS ENGINE                                  |
|  - Semantic Equivalence Clustering: y^(a) ~ y^(b) <=> Bidirectional Entailment                     |
|  - Semantic Entropy: SE(x) = - \sum p(c_k | x) log p(c_k | x)                                       |
|  - Evidence Conformity Score: CS(y*) = max_k [ P_NLI(Entail) - P_NLI(Contradict) ]                 |
|  - Unified Calibrated Confidence: C(x) = (1 - SE / log M)^beta * ((1 + CS) / 2)^(1 - beta)         |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                         STAGE 5: CALIBRATED SELECTIVE DECISION HEAD                                |
|                                                                                                    |
|            /--- C(x) < tau*  ---> [REJECT OPTION: "ABSTAIN: Insufficient Evidence to Decide"]      |
|  DECISION:                                                                                         |
|            \--- C(x) >= tau* ---> [GROUNDED ANSWER + VISUAL BOXES + CITATION TAGS]                 |
+----------------------------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```
GeoTrust-RAG/
├── configs/
│   └── config_local.yaml              # Hyperparameters, paths, and retrieval settings
├── data/
│   ├── knowledge_base/
│   │   ├── geospatial_corpus.json     # Curated FEMA, FAO, USGS, and OSM knowledge rules
│   │   └── faiss_index.bin            # Indexed dense vector database (FAISS HNSW)
│   ├── real_combined_benchmark.json   # 118 real evaluation items across 7 geospatial tasks
│   ├── real_floodnet/                 # High-resolution UAV drone imagery (Hurricane Harvey)
│   └── real_rsvqa/                    # Multispectral satellite imagery (RSVQA-LR)
├── scripts/
│   ├── 01_build_knowledge_base.py     # Embeds and indexes knowledge corpus with FAISS
│   ├── 02_run_benchmark.py            # Executes benchmark evaluations
│   ├── 03_plot_results.py             # Visualizes evaluation curves
│   ├── 04_download_real_datasets.py   # Downloader for RSVQA and FloodNet
│   ├── plot_real_experiments.py       # Plots real experimental figures
│   └── run_all_real_experiments.py    # Master runner executing EXP-01 to EXP-07
├── src/
│   ├── metrics.py                     # ECE, AURC, Brier Score, and Citation Recall
│   ├── model_engine.py                # Multimodal perception and autoregressive generation
│   ├── retrieval.py                   # Context-aware vector retrieval & re-ranking
│   └── trustworthiness.py             # Semantic entropy & calibrated selective prediction
├── requirements.txt                   # Environment dependencies
├── LICENSE                            # MIT License
└── README.md                          # Project documentation
```

---

## ⚡ Quickstart & Installation

### 1. Environment Setup

Clone this repository and create a Python virtual environment:

```bash
git clone https://github.com/farhansajid/GeoTrust-RAG.git
cd GeoTrust-RAG

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build the Geospatial Knowledge Index

Ingest the authoritative geospatial corpus (FEMA, FAO LCCS, USGS, OSM) and build the FAISS index:

```bash
python scripts/01_build_knowledge_base.py
```

### 3. Run Real Experiments

Execute the full suite of reproducible experiments:

```bash
python scripts/run_all_real_experiments.py
```

---

## 📊 Core Empirical Findings

All evaluations originate strictly from executed inference passes on laptop-class hardware (Intel Core Ultra 7 CPU, NVIDIA GeForce RTX 5060 Laptop GPU):

* **Reasoning Accuracy**: Context-Aware Geo-RAG achieves **37.29%** accuracy vs. **24.58%** for the base unaugmented model (**+12.71% absolute gain**).
* **Calibration**: Expected Calibration Error (ECE) is reduced by **50.5%** (**0.2115** vs. **0.4274**).
* **Trustworthy Abstention**: Full GeoTrust achieves **100% correct abstention** on optical ambiguity probes (e.g., subterranean infrastructure where optical sensors cannot see).
* **Hardware Efficiency**: Peak GPU memory footprint is only **1,438.6 MB** (~1.44 GB), with sub-10 ms CPU vector retrieval latency.
* **Statistical Significance**: McNemar's paired test confirms significance over the baseline ($\chi^2 = 17.05, p = 0.000036$).

---

## 📜 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{ullah2026geotrust,
  title={{GeoTrust-RAG}: Trustworthy Retrieval-Augmented Vision-Language Reasoning for Interactive Disaster Response in Earth Observation},
  author={Ullah, Farhan and Ahmad, Muhammad and Collaborators},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
