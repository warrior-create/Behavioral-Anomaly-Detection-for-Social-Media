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

# Import project modules
from preprocessing.text_cleaner import TextCleaner
from preprocessing.stylometry import StylometricAnalyzer
from features.bert_embeddings import BERTEmbedder
from features.temporal_features import TemporalFeatureExtractor
from graph.graph_builder import GraphBuilder
from graph.community_detection import CommunityDetector
from models.train_supervised import SupervisedDetector
from models.train_unsupervised import UnsupervisedDetector
from evaluation.metrics import ModelEvaluator
from visualization.plotly_dashboard import DashboardGenerator
from visualization.export_gephi import GephiExporter

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
        """Step 3: Extract stylometric and embedding features"""
        logger.info("=" * 80)
        logger.info("STEP 3: FEATURE EXTRACTION")
        logger.info("=" * 80)
        
        # Stylometric features
        stylo = StylometricAnalyzer()
        processed_files = list(Path('data/processed').glob('*_cleaned.csv'))
        
        for file_path in processed_files:
            logger.info(f"Extracting stylometric features from {file_path.name}")
            features = stylo.extract_features(str(file_path))
            output_path = f"data/features/{file_path.stem}_stylo.csv"
            features.to_csv(output_path, index=False)
        
        # BERT embeddings
        embedder = BERTEmbedder(self.config.get('bert', {}))
        for file_path in processed_files:
            logger.info(f"Generating BERT embeddings for {file_path.name}")
            embeddings = embedder.generate_embeddings(str(file_path))
            output_path = f"data/features/{file_path.stem}_embeddings.npy"
            embedder.save_embeddings(embeddings, output_path)
        
        # Temporal features
        temporal = TemporalFeatureExtractor()
        for file_path in processed_files:
            logger.info(f"Extracting temporal features from {file_path.name}")
            temp_features = temporal.extract_features(str(file_path))
            output_path = f"data/features/{file_path.stem}_temporal.csv"
            temp_features.to_csv(output_path, index=False)
    
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
    
    def run_model_training(self):
        """Step 6: Train ML models"""
        logger.info("=" * 80)
        logger.info("STEP 6: MODEL TRAINING")
        logger.info("=" * 80)
        
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
        
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        try:
            logger.info("Starting CIB Detection Pipeline")
            logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            steps = [
                ("Data Collection", self.run_data_collection),
                ("Preprocessing", self.run_preprocessing),
                ("Feature Extraction", self.run_feature_extraction),
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
            'graph', 'community', 'train', 'evaluate', 'visualize'
        ],
        default='all',
        help='Run specific pipeline step'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CIBDetectionPipeline(args.config)
    
    # Execute requested step
    if args.step == 'all':
        pipeline.run_full_pipeline()
    # elif args.step == 'collect':
    #     pipeline.run_data_collection()
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


if __name__ == "__main__":
    main()