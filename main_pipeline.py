"""
Main Pipeline for CIB Detection System
Coordinates all modules for end-to-end detection of coordinated inauthentic behavior
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# Import project modules
from preprocessing.text_cleaner import TextCleaner
from preprocessing.stylometry import StylometricAnalyzer
from preprocessing.text_normalizer import TextNormalizer
from features.bert_embeddings import BERTEmbedder
from features.temporal_features import TemporalFeatureExtractor
from features.tfidf_features import TFIDFAnalyzer
from features.ner_features import NERFeatureExtractor
from features.topic_modeling import TopicModeler
from features.ngram_perplexity import PerplexityFeatureExtractor
from features.pmi_cooccurrence import PMICooccurrenceAnalyzer
from graph.graph_builder import GraphBuilder
from graph.community_detection import CommunityDetector
from graph.graph_feature_extractor import extract_all_graphs
from models.train_supervised import SupervisedDetector
from models.train_unsupervised import UnsupervisedDetector
from models.predict import BotPredictor
from evaluation.metrics import ModelEvaluator
from visualization.plotly_dashboard import DashboardGenerator
from visualization.export_gephi import GephiExporter
from combine_features import combine_feature_files
from create_labels import create_robust_labels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CIBDetectionPipeline:
    """Main pipeline orchestrator for CIB detection"""
    
    def __init__(self, config_path):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.setup_directories()
        logger.info("CIB Detection Pipeline initialized")
    
    def _load_config(self, config_path):
        """Load YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'data/raw',
            'data/processed',
            'data/features',
            'data/graphs',
            'data/models',
            'data/results',
            'data/visualizations'
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        logger.info("Directory structure created")
    
    def run_data_collection(self):
        """Step 1: Collect Twitter data"""
        logger.info("=" * 80)
        logger.info("STEP 1: DATA COLLECTION (TWITTER ONLY)")
        logger.info("=" * 80)
    
        collection_mode = self.config.get('data_collection', {}).get('mode', 'file')
    
        if collection_mode == 'api':
            # Twitter collection only
            if self.config.get('twitter', {}).get('enabled', False):
                from ingestion.twitter_collector import TwitterCollector
            
                twitter = TwitterCollector(self.config['twitter'])
            
                # Get duration from config
                duration = self.config.get('data_collection', {}).get('duration_minutes', 60)
            
                # Collect data
                twitter_data = twitter.collect_real_time(duration_minutes=duration)
            
                if not twitter_data.empty:
                    # Save with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = f'data/raw/twitter_data_{timestamp}.csv'
                    twitter.save_data(twitter_data, output_path)
                
                    # Show statistics
                    stats = twitter.get_collection_stats(twitter_data)
                    logger.info(f"Collection Statistics:")
                    for key, value in stats.items():
                        logger.info(f"  {key}: {value}")
                else:
                    logger.warning("No Twitter data collected")
            else:
                logger.error("Twitter is not enabled in config.yaml")
        else:
            logger.info("Using existing data files from data/raw/")

    def run_preprocessing(self):
        """Step 2: Clean and preprocess text data"""
        logger.info("=" * 80)
        logger.info("STEP 2: TEXT PREPROCESSING")
        logger.info("=" * 80)
        
        cleaner = TextCleaner(self.config.get('preprocessing', {}))
        
        # Process all CSV files in raw directory
        raw_files = list(Path('data/raw').glob('*.csv'))
        for file_path in raw_files:
            logger.info(f"Processing {file_path.name}")
            cleaned_data = cleaner.process_file(str(file_path))
            output_path = f"data/processed/{file_path.stem}_cleaned.csv"
            cleaned_data.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned data to {output_path}")
    
    def run_feature_extraction(self):
        """Step 3: Extract all features — stylometric, embedding, and NLP."""
        logger.info("=" * 80)
        logger.info("STEP 3: FEATURE EXTRACTION (Stylometry + BERT + NLP Suite)")
        logger.info("=" * 80)

        processed_files = list(Path('data/processed').glob('*_cleaned.csv'))

        # --- 3a. Stylometric features ---
        stylo = None
        for file_path in processed_files:
            output_path = f"data/features/{file_path.stem}_stylo.csv"
            if Path(output_path).exists():
                logger.info(f"Stylometric features already exist at {output_path}, skipping.")
            else:
                if stylo is None:
                    stylo = StylometricAnalyzer()
                logger.info(f"Extracting stylometric features from {file_path.name}")
                features = stylo.extract_features(str(file_path))
                features.to_csv(output_path, index=False)

        # --- 3b. BERT semantic embeddings ---
        for file_path in processed_files:
            output_path = f"data/features/{file_path.stem}_embeddings.npy"
            if Path(output_path).exists():
                logger.info(f"BERT embeddings already exist at {output_path}, skipping recomputation.")
            else:
                logger.info(f"Generating BERT embeddings for {file_path.name}")
                embedder = BERTEmbedder(self.config.get('bert', {}))
                embeddings = embedder.generate_embeddings(str(file_path))
                embedder.save_embeddings(embeddings, output_path)

        # --- 3c. Temporal behaviour features ---
        temporal = None
        for file_path in processed_files:
            output_path = f"data/features/{file_path.stem}_temporal.csv"
            if Path(output_path).exists():
                logger.info(f"Temporal features already exist at {output_path}, skipping.")
            else:
                if temporal is None:
                    temporal = TemporalFeatureExtractor()
                logger.info(f"Extracting temporal features from {file_path.name}")
                temp_features = temporal.extract_features(str(file_path))
                temp_features.to_csv(output_path, index=False)

        # --- 3d. TF-IDF content similarity features [NLP] ---
        tfidf_analyzer = None
        for file_path in processed_files:
            output_path = f"data/features/{file_path.stem}_tfidf.csv"
            if Path(output_path).exists():
                logger.info(f"TF-IDF features already exist at {output_path}, skipping.")
            else:
                if tfidf_analyzer is None:
                    tfidf_analyzer = TFIDFAnalyzer(self.config.get('tfidf', {}))
                logger.info(f"Extracting TF-IDF features from {file_path.name}")
                tfidf_features = tfidf_analyzer.extract_features(str(file_path))
                tfidf_features.to_csv(output_path, index=False)

        # --- 3e. Named Entity Recognition features [NLP] ---
        ner_extractor = None
        for file_path in processed_files:
            output_path = f"data/features/{file_path.stem}_ner.csv"
            if Path(output_path).exists():
                logger.info(f"NER features already exist at {output_path}, skipping.")
            else:
                try:
                    if ner_extractor is None:
                        ner_extractor = NERFeatureExtractor(self.config.get('ner', {}))
                    logger.info(f"Extracting NER features from {file_path.name}")
                    ner_features = ner_extractor.extract_features(str(file_path))
                    ner_features.to_csv(output_path, index=False)
                except Exception as e:
                    logger.warning(f"NER skipped: {e}")

        # --- 3f. LDA Topic Modelling features [NLP] ---
        topic_modeler = None
        for file_path in processed_files:
            output_path = f"data/features/{file_path.stem}_topics.csv"
            if Path(output_path).exists():
                logger.info(f"Topic features already exist at {output_path}, skipping.")
            else:
                if topic_modeler is None:
                    topic_modeler = TopicModeler(self.config.get('topic_modeling', {}))
                logger.info(f"Running LDA topic modelling on {file_path.name}")
                topic_features = topic_modeler.extract_features(str(file_path))
                topic_features.to_csv(output_path, index=False)
                topic_modeler.save_model('data/models/')

        # --- 3g. N-gram Perplexity features [NLP] ---
        perplexity_extractor = None
        for file_path in processed_files:
            output_path = f"data/features/{file_path.stem}_perplexity.csv"
            if Path(output_path).exists():
                logger.info(f"Perplexity features already exist at {output_path}, skipping.")
            else:
                if perplexity_extractor is None:
                    perplexity_extractor = PerplexityFeatureExtractor(
                        self.config.get('perplexity', {})
                    )
                logger.info(f"Computing perplexity features from {file_path.name}")
                perplexity_features = perplexity_extractor.extract_features(str(file_path))
                perplexity_features.to_csv(output_path, index=False)

        # --- 3h. PMI Keyword Co-occurrence features [NLP] ---
        pmi_analyzer = None
        for file_path in processed_files:
            output_path = f"data/features/{file_path.stem}_pmi.csv"
            if Path(output_path).exists():
                logger.info(f"PMI features already exist at {output_path}, skipping.")
            else:
                if pmi_analyzer is None:
                    pmi_analyzer = PMICooccurrenceAnalyzer(self.config.get('pmi', {}))
                logger.info(f"Computing PMI co-occurrence features from {file_path.name}")
                pmi_features = pmi_analyzer.extract_features(str(file_path))
                pmi_features.to_csv(output_path, index=False)
    
    def run_graph_construction(self):
        """Step 4: Build interaction graphs"""
        logger.info("=" * 80)
        logger.info("STEP 4: GRAPH CONSTRUCTION")
        logger.info("=" * 80)
        
        builder = GraphBuilder(self.config.get('graph', {}))
        processed_files = list(Path('data/processed').glob('*_cleaned.csv'))
        
        for file_path in processed_files:
            logger.info(f"Building graph from {file_path.name}")
            graph = builder.build_graph(str(file_path))
            output_path = f"data/graphs/{file_path.stem}_graph.gpickle"
            builder.save_graph(graph, output_path)

            # Graph statistics
            stats = builder.get_graph_statistics(graph)
            logger.info(f"Graph stats: {stats}")

        # --- Extract graph centrality features ---
        logger.info("Extracting graph centrality features ...")
        extract_all_graphs(graphs_dir='data/graphs', output_path='data/features/graph_features.csv')

    
    def run_community_detection(self):
        """Step 5: Detect coordinated communities"""
        logger.info("=" * 80)
        logger.info("STEP 5: COMMUNITY DETECTION")
        logger.info("=" * 80)
        
        detector = CommunityDetector(self.config.get('community', {}))
        graph_files = list(Path('data/graphs').glob('*.gpickle'))
        
        for file_path in graph_files:
            logger.info(f"Detecting communities in {file_path.name}")
            communities = detector.analyze_community_coordination(str(file_path))
            output_path = f"data/results/{file_path.stem}_communities.csv"
            communities.to_csv(output_path, index=False)
            logger.info(f"Detected {len(communities['community_id'].unique())} communities")

        # --- Combine ALL features into final_user_features.csv ---
        logger.info("Combining all features into final_user_features.csv ...")
        combine_feature_files()

    
    def run_model_training(self):
        """Step 6: Train ML models"""
        logger.info("=" * 80)
        logger.info("STEP 6: MODEL TRAINING")
        logger.info("=" * 80)
        
        # Ensure ground truth labels exist before training
        label_file = Path('data/labels/ground_truth_labels.csv')
        if not label_file.exists():
            logger.info("Ground truth labels not found at %s. Generating automatically...", label_file)
            create_robust_labels()

        # Supervised models
        if self.config.get('supervised', {}).get('enabled', True):
            supervised = SupervisedDetector(self.config.get('supervised', {}))
            supervised.train_models()
            supervised.save_models('data/models/')
        
        # Unsupervised models
        if self.config.get('unsupervised', {}).get('enabled', True):
            unsupervised = UnsupervisedDetector(self.config.get('unsupervised', {}))
            unsupervised.train_models()
            unsupervised.save_models('data/models/')
    
    def run_evaluation(self):
        """Step 7: Evaluate model performance"""
        logger.info("=" * 80)
        logger.info("STEP 7: MODEL EVALUATION")
        logger.info("=" * 80)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all_models('data/models/', 'data/results/')
        
        # Save evaluation metrics
        results.to_csv('data/results/evaluation_metrics.csv', index=False)
        logger.info("Evaluation complete. Metrics saved.")
    
    def run_visualization(self):
        """Step 8: Generate visualizations"""
        logger.info("=" * 80)
        logger.info("STEP 8: VISUALIZATION")
        logger.info("=" * 80)
        
        # Gephi export
        exporter = GephiExporter()
        graph_files = list(Path('data/graphs').glob('*.gpickle'))
        for file_path in graph_files:
            output_path = f"data/visualizations/{file_path.stem}.gexf"
            exporter.export_graph(str(file_path), output_path)
            logger.info(f"Exported {file_path.name} to GEXF format")

        # Interactive dashboard
        dashboard = DashboardGenerator(self.config.get('visualization', {}))
        dashboard.generate_dashboard('data/results/', 'data/graphs/')

    def run_prediction(self, input_file: Optional[str] = None):
        """Step 9: Run Bot / CIB Prediction on new or existing dataset"""
        logger.info("=" * 80)
        logger.info("INFERENCE & BOT PREDICTION")
        logger.info("=" * 80)

        predictor = BotPredictor()
        predictor.load_artefacts()

        if input_file is None:
            # Check for cleaned files first, then raw files
            cleaned_candidates = sorted(list(Path('data/processed').glob('*_cleaned.csv')))
            if cleaned_candidates:
                input_file = str(cleaned_candidates[0])
            else:
                raw_candidates = sorted(list(Path('data/raw').glob('*.csv')))
                input_file = str(raw_candidates[0]) if raw_candidates else None

        if input_file and Path(input_file).exists():
            logger.info(f"Running prediction pipeline on: {input_file}")
            predictions = predictor.predict_from_csv(input_file)
            logger.info(f"Predictions completed successfully for {len(predictions)} accounts.")
            return predictions
        else:
            logger.error("No valid input file provided or found for prediction.")
            return None
        
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        try:
            logger.info("Starting CIB Detection Pipeline")
            logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            steps = [
                ("Data Collection", self.run_data_collection),
                ("Preprocessing", self.run_preprocessing),
                # Step 3 includes: Stylometry, BERT, Temporal,
                # TF-IDF, NER, LDA Topics, Perplexity, PMI
                ("Feature Extraction (NLP Suite)", self.run_feature_extraction),
                ("Graph Construction", self.run_graph_construction),
                ("Community Detection", self.run_community_detection),
                ("Model Training", self.run_model_training),
                ("Evaluation", self.run_evaluation),
                ("Visualization", self.run_visualization)
            ]
            
            for step_name, step_func in steps:
                try:
                    step_func()
                    logger.info(f"✓ {step_name} completed successfully")
                except Exception as e:
                    logger.error(f"✗ {step_name} failed: {e}")
                    if not self.config.get('pipeline', {}).get('continue_on_error', False):
                        raise
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info("Results available in:")
            logger.info("  - data/results/: Detection results and metrics")
            logger.info("  - data/models/: Trained models")
            logger.info("  - data/visualizations/: Network visualizations")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='CIB Detection Pipeline - BITS WILP Dissertation Project'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=[
            'all', 'collect', 'preprocess', 'features',
            'graph', 'community', 'train', 'evaluate', 'visualize', 'predict'
        ],
        default='all',
        help='Run specific pipeline step'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Input file path for prediction or custom data'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CIBDetectionPipeline(args.config)
    
    # Execute requested step
    if args.step == 'all':
        pipeline.run_full_pipeline()
    elif args.step == 'collect':
        pipeline.run_data_collection()
    elif args.step == 'preprocess':
        pipeline.run_preprocessing()
    elif args.step == 'features':
        pipeline.run_feature_extraction()
    elif args.step == 'graph':
        pipeline.run_graph_construction()
    elif args.step == 'community':
        pipeline.run_community_detection()
    elif args.step == 'train':
        pipeline.run_model_training()
    elif args.step == 'evaluate':
        pipeline.run_evaluation()
    elif args.step == 'visualize':
        pipeline.run_visualization()
    elif args.step == 'predict':
        pipeline.run_prediction(args.input_file)


if __name__ == "__main__":
    main()