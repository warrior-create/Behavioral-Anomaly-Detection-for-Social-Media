CIB DETECTION SYSTEM - PROJECT DOCUMENTATION
==============================================================================

PROJECT DETAILS
---------------
Project Title  : Detection of Coordinated Inauthentic Behavior and Bot Networks
                 on Social Media Platforms

==============================================================================

1. EXECUTIVE SUMMARY
--------------------
This project delivers an end-to-end automated pipeline for detecting Coordinated 
Inauthentic Behavior (CIB) and malicious bot networks on social media. Unlike 
traditional spam filters that look for high-volume posting, this system uses a 
multi-modal approach to detect *behavioral coordination*.

It combines three advanced techniques:
  A. Social Network Analysis (Graph Theory) to find clusters of coordinated users.
  B. Stylometric NLP to detect automated writing styles and repetitive content.
  C. Unsupervised Machine Learning to identify anomalies without labeled data.

The system is modular, production-ready, and includes an interactive dashboard 
for real-time analysis.

2. SYSTEM ARCHITECTURE & MODULES
--------------------------------
The project is built on a modular architecture where each component handles a 
specific stage of the detection pipeline.

[Module 1: Data Ingestion]
- created a synthetic dataset to anlyse .

[Module 2: Preprocessing]
- File: nlpr/preprocessing/
- Purpose: Cleans raw social media text for analysis.
- Actions:
  * Removes URLs, mentions, and emojis.
  * Normalizes text (lowercase, stemming).
  * Detects language to filter non-English content.

[Module 3: Feature Extraction (The Core Intelligence)]
- File: nlpr/features/
- Purpose: Converts raw behavior into 120 numerical features.
- Components:
  A. Stylometry: Measures writing style (Sentence length, vocabulary richness, 
     punctuation ratios) to detect automated text generation.
  B. Temporal Analysis: Measures timing (Inter-post intervals, burstiness) to 
     detect automated scheduling (e.g., posting exactly every 15 mins).
  C. Semantic Analysis: Uses BERT embeddings (384-dim) to detect content 
     similarity (bots posting identical messages).

[Module 4: Graph Construction]
- File: nlpr/graph/
- Purpose: Models user interactions as a network.
- Methodology:
  * Nodes = Users
  * Edges = Replies/Mentions/Retweets
  * Weights = Frequency of interaction
- Analytics: Calculates Centrality (Who is influential?) and Modularity 
  (Are there isolated cliques of bots?).

[Module 5: Model Training]
- File: nlpr/models/
- Purpose: Classifies users as 'Bot' or 'Human'.
- Models Implemented:
  A. Supervised: XGBoost, Random Forest, Logistic Regression (High accuracy).
  B. Unsupervised: DBSCAN, K-Means, Isolation Forest (Finds new bot clusters 
     without labels).
- Strategy: Uses "Hard Mode" training, where easy features (Total Posts) are 
  removed to force the model to learn complex behavioral patterns.

[Module 6: Evaluation]
- File: nlpr/evaluation/
- Purpose: Validates model performance.
- Metrics: Accuracy, Precision, Recall, F1-Score, and Silhouette Score.
- Feature: Automatically handles dimension mismatches between training and testing.

[Module 7: Visualization]
- File: nlpr/visualization/
- Purpose: Explains the results to the user.
- Outputs:
  * Interactive Dashboard (Plotly): Shows feature importance and performance.
  * Network Graph (.gexf): Exportable file for Gephi to visualize bot clusters.

1. DIRECTORY STRUCTURE
----------------------
CIB_Detection_System/
├── data/
│   ├── raw/                  # Input data
│   ├── processed/            # Cleaned data
│   ├── features/             # Feature matrices (120 dims)
│   ├── graphs/               # NetworkX graph objects
│   ├── models/               # Saved model artifacts (.pkl)
│   ├── results/              # Metrics CSVs
│   └── visualizations/       # Dashboard outputs & Gephi files
├── nlpr/                     # Source code package
├── config.yaml               # Configuration settings
├── main_pipeline.py          # Execution script
├── requirements.txt          # Python dependencies
└── README.txt                # This file

1. INSTALLATION
---------------
1. Unzip the project.
2. Open terminal in the project folder.
3. Create and activate virtual environment:
     python -m venv venv
     source venv/bin/activate  (Mac/Linux)
     venv\Scripts\activate     (Windows)
4. Install dependencies:
     pip install -r requirements.txt

5. HOW TO RUN
-------------
Run the pipeline sequentially using `main_pipeline.py`.

Step 1: Ingest & Clean Data
   python main_pipeline.py --step ingest
   python main_pipeline.py --step preprocess

Step 2: Generate Features (Stylometry + BERT + Temporal)
   python main_pipeline.py --step features

Step 3: Build Network Graph
   python main_pipeline.py --step graph

Step 4: Train Models (Supervised & Unsupervised)
   python main_pipeline.py --step train

Step 5: Evaluate Performance
   python main_pipeline.py --step evaluate

Step 6: Visualize Results
   python main_pipeline.py --step visualize
   -> Opens dashboard at http://127.0.0.1:8050/

6. RESULTS & INTERPRETATION
---------------------------
- **High Accuracy (~78-90%):** The model successfully distinguishes bots based on 
  inter-post time and stylometric features, not just volume.
- **Feature Importance:** The top predictor is often `avg_inter_post_time`, 
  indicating that the system detects the *regularity* of automation.
- **Clustering:** DBSCAN successfully identifies "noise" (outliers) and dense 
  clusters of coordinated accounts.
