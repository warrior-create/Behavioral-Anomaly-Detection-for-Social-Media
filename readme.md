# Detection of Coordinated Inauthentic Behavior and Bot Networks on Social Media

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
![NLP](https://img.shields.io/badge/NLP-BERT-orange)
![Graph Analytics](https://img.shields.io/badge/Graph%20Theory-NetworkX-purple)

## Overview

This project presents an end-to-end framework for detecting Coordinated Inauthentic Behavior (CIB) and malicious bot networks on social media platforms.

Unlike traditional spam detection systems that primarily rely on posting frequency, this framework combines behavioral analytics, natural language processing, graph theory, and machine learning to identify coordinated campaigns that mimic legitimate user activity.

The system integrates:

* Social Network Analysis
* Stylometric Natural Language Processing
* Temporal Behavioral Analysis
* Semantic Similarity Detection
* Supervised and Unsupervised Machine Learning
* Interactive Visualization and Monitoring

---

## Key Features

### Social Network Analysis

* User interaction graph construction
* Community detection
* Centrality analysis
* Clique identification
* Coordinated behavior discovery

### Stylometric Analysis

* Writing style fingerprinting
* Vocabulary richness analysis
* Punctuation behavior analysis
* Sentence structure analysis

### Temporal Analysis

* Inter-post interval analysis
* Burst detection
* Posting regularity measurement
* Automated scheduling detection

### Semantic Analysis

* BERT Embeddings (384 dimensions)
* Content similarity measurement
* Coordinated messaging detection

### Machine Learning Models

**Supervised**

* XGBoost
* Random Forest
* Logistic Regression

**Unsupervised**

* DBSCAN
* Isolation Forest
* K-Means Clustering

---

## System Architecture

```text
Raw Social Media Data
        │
        ▼
Data Ingestion
        │
        ▼
Preprocessing
        │
        ▼
Feature Engineering
(120+ Features)
        │
        ▼
Graph Construction
        │
        ▼
Model Training
        │
        ▼
Evaluation
        │
        ▼
Visualization Dashboard
```

---

## Feature Engineering

The framework extracts more than 120 behavioral features.

### Stylometric Features

* Average sentence length
* Vocabulary richness
* Character distribution
* Punctuation patterns
* Capitalization ratios

### Temporal Features

* Average inter-post interval
* Posting frequency
* Burstiness score
* Activity variance

### Semantic Features

* BERT embeddings
* Content similarity scores
* Topic consistency metrics

### Graph Features

* Degree Centrality
* Betweenness Centrality
* Clustering Coefficient
* Community Membership

---

## Project Structure

```text
Behavioral-Anomaly-Detection-for-Social-Media
│
├── data/
├── preprocessing/
├── features/
├── graph/
├── models/
├── evaluation/
├── visualization/
│
├── config.yaml
├── main_pipeline.py
├── requirements.txt
└── README.md
```

---

## Model Training

The framework employs both supervised and unsupervised learning approaches.

| Model               | Purpose                 |
| ------------------- | ----------------------- |
| XGBoost             | Classification          |
| Random Forest       | Ensemble Learning       |
| Logistic Regression | Explainable Baseline    |
| DBSCAN              | Cluster Discovery       |
| Isolation Forest    | Anomaly Detection       |
| K-Means             | Behavioral Segmentation |

To prevent shortcut learning, simple indicators such as total post count are removed during training, forcing models to learn deeper behavioral patterns.

---

## Evaluation Metrics

Performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Silhouette Score

The evaluation pipeline also handles feature dimensionality mismatches between training and inference automatically.

---

## Results

### Detection Performance

* Achieved approximately 78–90% classification accuracy
* Successfully distinguishes coordinated accounts from genuine users
* Detects behavioral automation rather than simple spam activity

### Important Predictive Features

* Average Inter-Post Time
* Burstiness Score
* Stylometric Consistency
* Semantic Similarity

### Clustering Performance

DBSCAN effectively identifies:

* Coordinated bot communities
* Behavioral outliers
* Previously unseen suspicious accounts

---

## Installation

```bash
git clone https://github.com/warrior-create/Behavioral-Anomaly-Detection-for-Social-Media.git

cd Behavioral-Anomaly-Detection-for-Social-Media

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
python main_pipeline.py --step ingest
python main_pipeline.py --step preprocess
python main_pipeline.py --step features
python main_pipeline.py --step graph
python main_pipeline.py --step train
python main_pipeline.py --step evaluate
python main_pipeline.py --step visualize
```

Dashboard:

```text
http://127.0.0.1:8050/
```

---

## Technology Stack

* Python
* Scikit-Learn
* XGBoost
* NetworkX
* BERT Embeddings
* Plotly Dash
* Pandas
* NumPy
* Matplotlib

---

## Applications

* Election Integrity Monitoring
* Bot Network Detection
* Misinformation Campaign Analysis
* Coordinated Influence Operations Detection
* Social Media Threat Intelligence
* Platform Trust and Safety

---

## Author

Vansh Goel
Indian Institute of Technology Mandi
