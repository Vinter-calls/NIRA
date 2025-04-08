# neural_assistant.py
# Core system for Neural Interface Research Assistant

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
import pickle
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neural_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NeuralAssistant")

class NeuralDataProcessor:
    """Process neural data from various sources (EEG, fMRI, etc.)"""
    
    def __init__(self, data_type="eeg"):
        """Initialize processor for specific data type
        
        Args:
            data_type (str): Type of neural data ('eeg', 'fmri', etc.)
        """
        self.data_type = data_type
        logger.info(f"Initialized NeuralDataProcessor for {data_type} data")
        
    def load_eeg_data(self, file_path):
        """Load EEG data from various formats
        
        Args:
            file_path (str): Path to EEG data file
            
        Returns:
            mne.Raw: Loaded EEG data object
        """
        logger.info(f"Loading EEG data from {file_path}")
        try:
            # Detect file format by extension
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == '.edf':
                raw = mne.io.read_raw_edf(file_path, preload=True)
            elif ext.lower() == '.bdf':
                raw = mne.io.read_raw_bdf(file_path, preload=True)
            elif ext.lower() == '.set':
                raw = mne.io.read_raw_eeglab(file_path, preload=True)
            elif ext.lower() == '.fif':
                raw = mne.io.read_raw_fif(file_path, preload=True)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
            logger.info(f"Successfully loaded EEG data with {len(raw.ch_names)} channels")
            return raw
        except Exception as e:
            logger.error(f"Error loading EEG data: {str(e)}")
            raise
    
    def preprocess_eeg(self, raw, l_freq=1.0, h_freq=40.0, notch_freq=50.0, resample_freq=250):
        """Basic EEG preprocessing
        
        Args:
            raw (mne.Raw): Raw EEG data
            l_freq (float): Low frequency cutoff for bandpass filter
            h_freq (float): High frequency cutoff for bandpass filter
            notch_freq (float): Frequency for notch filter (line noise)
            resample_freq (int): Frequency to resample data to
            
        Returns:
            mne.Raw: Preprocessed EEG data
        """
        logger.info("Preprocessing EEG data")
        try:
            # Create a copy to avoid modifying the original
            raw_proc = raw.copy()
            
            # Apply bandpass filter
            raw_proc.filter(l_freq=l_freq, h_freq=h_freq)
            logger.info(f"Applied bandpass filter: {l_freq}-{h_freq} Hz")
            
            # Apply notch filter for line noise
            raw_proc.notch_filter(freqs=notch_freq)
            logger.info(f"Applied notch filter at {notch_freq} Hz")
            
            # Resample to reduce data size
            raw_proc.resample(resample_freq)
            logger.info(f"Resampled to {resample_freq} Hz")
            
            return raw_proc
        except Exception as e:
            logger.error(f"Error preprocessing EEG data: {str(e)}")
            raise
    
    def extract_features(self, raw, method="spectral", bands=None):
        """Extract features from EEG data
        
        Args:
            raw (mne.Raw): Preprocessed EEG data
            method (str): Feature extraction method ('spectral', 'connectivity', etc.)
            bands (dict): Frequency bands for spectral features
            
        Returns:
            np.ndarray: Extracted features
            list: Feature names
        """
        logger.info(f"Extracting features using {method} method")
        
        if bands is None:
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }
        
        try:
            if method == "spectral":
                # Calculate power spectral density
                psd, freqs = mne.time_frequency.psd_welch(
                    raw, 
                    fmin=0.5, 
                    fmax=45, 
                    n_fft=int(raw.info['sfreq'] * 2),
                    n_overlap=int(raw.info['sfreq']),
                    n_per_seg=int(raw.info['sfreq'] * 4)
                )
                
                # Extract band powers
                features = []
                feature_names = []
                
                for ch_idx, ch_name in enumerate(raw.ch_names):
                    for band_name, (fmin, fmax) in bands.items():
                        # Find frequencies within the band
                        freq_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
                        # Calculate average power in the band
                        band_power = np.mean(psd[ch_idx][freq_idx])
                        features.append(band_power)
                        feature_names.append(f"{ch_name}_{band_name}")
                
                logger.info(f"Extracted {len(features)} spectral features")
                return np.array(features), feature_names
            
            elif method == "connectivity":
                # Placeholder for connectivity features
                # Would use methods like PLV, coherence, etc.
                logger.warning("Connectivity feature extraction not yet implemented")
                return np.array([]), []
            
            else:
                raise ValueError(f"Unknown feature extraction method: {method}")
                
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

class NeuralPatternAnalyzer:
    """Analyze neural patterns using machine learning techniques"""
    
    def __init__(self):
        """Initialize neural pattern analyzer"""
        self.models = {}
        logger.info("Initialized NeuralPatternAnalyzer")
    
    def create_ml_pipeline(self, model_type="random_forest"):
        """Create ML pipeline with preprocessing and model
        
        Args:
            model_type (str): Type of model to use
            
        Returns:
            sklearn.pipeline.Pipeline: ML pipeline
        """
        logger.info(f"Creating ML pipeline with {model_type} model")
        
        try:
            # Preprocessing steps
            preprocessing = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95))  # Keep 95% of variance
            ])
            
            # Model selection
            if model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=None, 
                    min_samples_split=2,
                    random_state=42
                )
            elif model_type == "neural_network":
                # Using Keras/TensorFlow neural network
                # We'll return preprocessing only and handle the NN separately
                return preprocessing
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessing', preprocessing),
                ('model', model)
            ])
            
            return pipeline
        
        except Exception as e:
            logger.error(f"Error creating ML pipeline: {str(e)}")
            raise
    
    def build_deep_learning_model(self, input_shape, output_shape, model_type="fcnn"):
        """Build deep learning model for neural data
        
        Args:
            input_shape (tuple): Shape of input data
            output_shape (int): Number of output classes
            model_type (str): Type of neural network
            
        Returns:
            tf.keras.Model: Compiled neural network
        """
        logger.info(f"Building {model_type} deep learning model")
        
        try:
            if model_type == "fcnn":
                model = Sequential([
                    Dense(128, activation='relu', input_shape=(input_shape,)),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(output_shape, activation='softmax' if output_shape > 1 else 'sigmoid')
                ])
                
            elif model_type == "lstm":
                # Assuming input shape is (timesteps, features)
                model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=input_shape),
                    Dropout(0.3),
                    LSTM(32),
                    Dense(output_shape, activation='softmax' if output_shape > 1 else 'sigmoid')
                ])
                
            elif model_type == "cnn":
                # Assuming input shape is (timesteps, features)
                model = Sequential([
                    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
                    MaxPooling1D(pool_size=2),
                    Conv1D(filters=32, kernel_size=3, activation='relu'),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dense(output_shape, activation='softmax' if output_shape > 1 else 'sigmoid')
                ])
                
            else:
                raise ValueError(f"Unsupported deep learning model type: {model_type}")
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy' if output_shape > 1 else 'binary_crossentropy',
                metrics=['accuracy']
            )
            
            model.summary()
            return model
        
        except Exception as e:
            logger.error(f"Error building deep learning model: {str(e)}")
            raise
    
    def train_model(self, X, y, model_name="default_model", model_type="random_forest"):
        """Train ML model on neural data
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            model_name (str): Name to save model under
            model_type (str): Type of model to use
            
        Returns:
            float: Model accuracy on test set
        """
        logger.info(f"Training {model_type} model: {model_name}")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            if model_type in ["random_forest"]:
                # Create and train sklearn pipeline
                pipeline = self.create_ml_pipeline(model_type)
                pipeline.fit(X_train, y_train)
                
                # Evaluate
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Save model
                self.models[model_name] = pipeline
                
            elif model_type in ["fcnn", "lstm", "cnn"]:
                # For deep learning models
                
                # Preprocess data
                preprocessing = self.create_ml_pipeline("neural_network")
                X_train_proc = preprocessing.fit_transform(X_train)
                X_test_proc = preprocessing.transform(X_test)
                
                # One-hot encode targets if needed
                n_classes = len(np.unique(y))
                if n_classes > 1:
                    y_train_onehot = tf.keras.utils.to_categorical(y_train)
                    y_test_onehot = tf.keras.utils.to_categorical(y_test)
                else:
                    y_train_onehot = y_train
                    y_test_onehot = y_test
                
                # Build and train model
                if model_type == "fcnn":
                    input_shape = X_train_proc.shape[1]
                    model = self.build_deep_learning_model(input_shape, n_classes, model_type)
                    
                    history = model.fit(
                        X_train_proc, y_train_onehot,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss', patience=10, restore_best_weights=True
                            )
                        ],
                        verbose=1
                    )
                
                else:  # lstm or cnn
                    # Reshape data for sequential models: (samples, timesteps, features)
                    # This assumes we've organized our data properly
                    # Here we're creating a dummy temporal structure if needed
                    timesteps = 10  # This should be determined by your actual data
                    n_features = X_train_proc.shape[1] // timesteps
                    
                    if X_train_proc.shape[1] % timesteps != 0:
                        # If not divisible evenly, adjust n_features
                        n_features = X_train_proc.shape[1] // timesteps + 1
                        # Pad with zeros to make it fit
                        pad_width = n_features * timesteps - X_train_proc.shape[1]
                        X_train_proc = np.pad(X_train_proc, ((0, 0), (0, pad_width)))
                        X_test_proc = np.pad(X_test_proc, ((0, 0), (0, pad_width)))
                    
                    X_train_proc = X_train_proc.reshape(-1, timesteps, n_features)
                    X_test_proc = X_test_proc.reshape(-1, timesteps, n_features)
                    
                    model = self.build_deep_learning_model((timesteps, n_features), n_classes, model_type)
                    
                    history = model.fit(
                        X_train_proc, y_train_onehot,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss', patience=10, restore_best_weights=True
                            )
                        ],
                        verbose=1
                    )
                
                # Evaluate
                if model_type == "fcnn":
                    accuracy = model.evaluate(X_test_proc, y_test_onehot)[1]
                else:
                    accuracy = model.evaluate(X_test_proc, y_test_onehot)[1]
                
                # Save model and preprocessing pipeline
                self.models[model_name] = {
                    'preprocessing': preprocessing,
                    'model': model,
                    'model_type': model_type
                }
            
            logger.info(f"Model {model_name} trained with test accuracy: {accuracy:.4f}")
            return accuracy
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def save_model(self, model_name, file_path):
        """Save trained model to disk
        
        Args:
            model_name (str): Name of model to save
            file_path (str): Path to save model to
        """
        logger.info(f"Saving model {model_name} to {file_path}")
        
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Check if it's a deep learning model
            if isinstance(model, dict) and 'model' in model:
                # Save preprocessing pipeline
                preproc_path = f"{os.path.splitext(file_path)[0]}_preproc.pkl"
                with open(preproc_path, 'wb') as f:
                    pickle.dump(model['preprocessing'], f)
                
                # Save Keras model
                model_path = f"{os.path.splitext(file_path)[0]}_keras.h5"
                model['model'].save(model_path)
                
                # Save metadata
                meta_path = f"{os.path.splitext(file_path)[0]}_meta.json"
                with open(meta_path, 'w') as f:
                    json.dump({'model_type': model['model_type']}, f)
                
            else:
                # Save sklearn pipeline
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info(f"Model {model_name} saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, file_path, model_name=None):
        """Load trained model from disk
        
        Args:
            file_path (str): Path to load model from
            model_name (str): Name to give loaded model
            
        Returns:
            str: Name of loaded model
        """
        if model_name is None:
            model_name = os.path.basename(file_path).split('.')[0]
            
        logger.info(f"Loading model from {file_path} as {model_name}")
        
        try:
            # Check if it's a deep learning model by looking for metadata
            meta_path = f"{os.path.splitext(file_path)[0]}_meta.json"
            if os.path.exists(meta_path):
                # Load metadata
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                # Load preprocessing pipeline
                preproc_path = f"{os.path.splitext(file_path)[0]}_preproc.pkl"
                with open(preproc_path, 'rb') as f:
                    preprocessing = pickle.load(f)
                
                # Load Keras model
                model_path = f"{os.path.splitext(file_path)[0]}_keras.h5"
                model = tf.keras.models.load_model(model_path)
                
                self.models[model_name] = {
                    'preprocessing': preprocessing,
                    'model': model,
                    'model_type': metadata['model_type']
                }
                
            else:
                # Load sklearn pipeline
                with open(file_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
            
            logger.info(f"Model {model_name} loaded successfully")
            return model_name
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

class KnowledgeManager:
    """Manage knowledge base and research papers"""
    
    def __init__(self, db_path="knowledge_base"):
        """Initialize knowledge manager
        
        Args:
            db_path (str): Path to knowledge database
        """
        self.db_path = db_path
        self.papers = {}
        self.concepts = {}
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Load existing data if available
        self._load_database()
        
        logger.info(f"Initialized KnowledgeManager with database at {db_path}")
    
    def _load_database(self):
        """Load database from disk"""
        papers_file = os.path.join(self.db_path, "papers.json")
        concepts_file = os.path.join(self.db_path, "concepts.json")
        
        try:
            if os.path.exists(papers_file):
                with open(papers_file, 'r') as f:
                    self.papers = json.load(f)
                logger.info(f"Loaded {len(self.papers)} papers from database")
            
            if os.path.exists(concepts_file):
                with open(concepts_file, 'r') as f:
                    self.concepts = json.load(f)
                logger.info(f"Loaded {len(self.concepts)} concepts from database")
                
        except Exception as e:
            logger.error(f"Error loading database: {str(e)}")
            # Continue with empty database
            self.papers = {}
            self.concepts = {}
    
    def _save_database(self):
        """Save database to disk"""
        papers_file = os.path.join(self.db_path, "papers.json")
        concepts_file = os.path.join(self.db_path, "concepts.json")
        
        try:
            with open(papers_file, 'w') as f:
                json.dump(self.papers, f, indent=2)
            
            with open(concepts_file, 'w') as f:
                json.dump(self.concepts, f, indent=2)
                
            logger.info("Database saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving database: {str(e)}")
            raise
    
    def add_paper(self, paper_id, metadata):
        """Add research paper to knowledge base
        
        Args:
            paper_id (str): Unique identifier for paper
            metadata (dict): Paper metadata (title, authors, abstract, etc.)
        """
        logger.info(f"Adding paper {paper_id} to knowledge base")
        
        try:
            self.papers[paper_id] = metadata
            self._save_database()
            logger.info(f"Paper {paper_id} added successfully")
            
        except Exception as e:
            logger.error(f"Error adding paper: {str(e)}")
            raise
    
    def add_concept(self, concept_id, metadata):
        """Add neuroscience concept to knowledge base
        
        Args:
            concept_id (str): Unique identifier for concept
            metadata (dict): Concept metadata (name, description, related concepts, etc.)
        """
        logger.info(f"Adding concept {concept_id} to knowledge base")
        
        try:
            self.concepts[concept_id] = metadata
            self._save_database()
            logger.info(f"Concept {concept_id} added successfully")
            
        except Exception as e:
            logger.error(f"Error adding concept: {str(e)}")
            raise
    
    def search_papers(self, query, field="all"):
        """Search for papers in knowledge base
        
        Args:
            query (str): Search query
            field (str): Field to search in ("title", "abstract", "all")
            
        Returns:
            list: Matching paper IDs
        """
        logger.info(f"Searching papers for query: {query} in field: {field}")
        
        try:
            results = []
            query = query.lower()
            
            for paper_id, metadata in self.papers.items():
                if field == "title" and query in metadata.get("title", "").lower():
                    results.append(paper_id)
                elif field == "abstract" and query in metadata.get("abstract", "").lower():
                    results.append(paper_id)
                elif field == "all":
                    paper_text = " ".join([
                        metadata.get("title", ""),
                        metadata.get("abstract", ""),
                        " ".join(metadata.get("keywords", []))
                    ]).lower()
                    
                    if query in paper_text:
                        results.append(paper_id)
            
            logger.info(f"Found {len(results)} matching papers")
            return results
            
        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            raise
    
    def search_concepts(self, query):
        """Search for concepts in knowledge base
        
        Args:
            query (str): Search query
            
        Returns:
            list: Matching concept IDs
        """
        logger.info(f"Searching concepts for query: {query}")
        
        try:
            results = []
            query = query.lower()
            
            for concept_id, metadata in self.concepts.items():
                concept_text = " ".join([
                    metadata.get("name", ""),
                    metadata.get("description", ""),
                    " ".join(metadata.get("related_terms", []))
                ]).lower()
                
                if query in concept_text:
                    results.append(concept_id)
            
            logger.info(f"Found {len(results)} matching concepts")
            return results
            
        except Exception as e:
            logger.error(f"Error searching concepts: {str(e)}")
            raise

class NeuralAssistant:
    """Main class for Neural Interface Research Assistant"""
    
    def __init__(self):
        """Initialize Neural Assistant"""
        self.data_processor = NeuralDataProcessor()
        self.pattern_analyzer = NeuralPatternAnalyzer()
        self.knowledge_manager = KnowledgeManager()
        
        logger.info("Neural Interface Research Assistant initialized")
    
    def process_eeg_file(self, file_path, preprocess=True, extract_features=True):
        """Process EEG file end-to-end
        
        Args:
            file_path (str): Path to EEG file
            preprocess (bool): Whether to preprocess data
            extract_features (bool): Whether to extract features
            
        Returns:
            dict: Results including raw data, preprocessed data, and features
        """
        logger.info(f"Processing EEG file: {file_path}")
        
        try:
            results = {"file_path": file_path}
            
            # Load data
            raw = self.data_processor.load_eeg_data(file_path)
            results["raw_data"] = raw
            
            # Preprocess if requested
            if preprocess:
                raw_proc = self.data_processor.preprocess_eeg(raw)
                results["preprocessed_data"] = raw_proc
            else:
                raw_proc = raw
            
            # Extract features if requested
            if extract_features:
                features, feature_names = self.data_processor.extract_features(raw_proc)
                results["features"] = features
                results["feature_names"] = feature_names
            
            logger.info(f"Successfully processed EEG file: {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing EEG file: {str(e)}")
            raise
    
    def train_brain_state_classifier(self, data_files, labels, model_name="brain_state_model", model_type="random_forest"):
        """Train a classifier to identify brain states
        
        Args:
            data_files (list): List of EEG file paths
            labels (list): Corresponding labels for each file
            model_name (str): Name to save model under
            model_type (str): Type of model to use
            
        Returns:
            float: Model accuracy
        """
        logger.info(f"Training brain state classifier using {len(data_files)} files")
        
        try:
            # Process all files
            all_features = []
            
            for file_path in data_files:
                results = self.process_eeg_file(file_path)
                all_features.append(results["features"])
            
            # Convert to numpy array
            X = np.array(all_features)
            y = np.array(labels)
            
            # Train model
            accuracy = self.pattern_analyzer.train_model(X, y, model_name, model_type)
            
            logger.info(f"Brain state classifier trained with accuracy: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            logger.error(f"Error training brain state classifier: {str(e)}")
            raise
    
    def visualize_eeg(self, raw, n_channels=5, duration=10, start=0):
        """Visualize EEG data
        
        Args:
            raw (mne.Raw): EEG data
            n_channels (int): Number of channels to display
            duration (float): Duration in seconds to display
            start (float): Start time in seconds
            
        Returns:
            matplotlib.Figure: Figure with visualization
        """
        logger.info("Visualizing EEG data")
        
        try:
            # Get subset of channels if needed
            if n_channels < len(raw.ch_names):
                ch_names = raw.ch_names[:n_channels]
            else:
                ch_names = raw.ch_names
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(15, 8))
            raw.plot(duration=duration, start=start, n_channels=n_channels, 
                     scalings='auto', show=False, ax=ax)
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing EEG data: {str(e)}")
            raise
    
    def add_research_paper(self, title, authors, abstract, keywords=None, publication=None, year=None):
        """Add research paper to knowledge base
        
        Args:
            title (str): Paper title
            authors (list): List of authors
            abstract (str): Paper abstract
            keywords (list): List of keywords
            publication (str): Publication venue
            year (int): Publication year
            
        Returns:
            str: Assigned paper ID
        """
        logger.info(f"Adding research paper: {title}")
        
        try:
            # Create unique ID
            import hashlib
            paper_id = hashlib.md5(title.encode()).hexdigest()[:10]
            
            # Create metadata
            metadata = {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "keywords": keywords if keywords else [],
                "publication": publication,
                "year": year,
                "added_date": pd.Timestamp.now().strftime("%Y-%m-%d")
            }
            
            # Add to knowledge base
            self.knowledge_manager.add_paper(paper_id, metadata)
            
            logger.info(f"Research paper added with ID: {paper_id}")
            return paper_id
            
        except Exception as e:
            logger.error(f"Error adding research paper: {str(e)}")
            raise

# Example usage function
def run_example():
    """Run example workflow"""
    
    # Initialize assistant
    assistant = NeuralAssistant()
    
    # Add some papers to knowledge base
    assistant.add_research_paper(
        title="The