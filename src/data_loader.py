"""
Data Loader for Multi-Lingual Lip Reading
Handles data loading, preprocessing, and batch generation
Author: AI Assistant
Date: October 2025
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
import json


class CountableGenerator(keras.utils.Sequence):
    """Keras Sequence for data generation with automatic steps_per_epoch calculation"""
    
    def __init__(self, data_loader, video_paths, labels, preprocessor, augment, batch_size):
        self.data_loader = data_loader
        self.video_paths = video_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.augment = augment
        self.batch_size = batch_size
        self.num_samples = len(video_paths)
        self.indices = np.arange(self.num_samples)
        self.on_epoch_end()
    
    def __len__(self):
        """Return number of batches per epoch (required by Keras Sequence)"""
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        """Get batch at given index (required by Keras Sequence)"""
        # Generate indices for the batch
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Generate batch data
        batch_sequences = []
        batch_labels = []
        
        for idx in batch_indices:
            try:
                # Process video
                video_path = self.video_paths[idx]
                sequence = self.preprocessor.process_video(video_path)
                
                batch_sequences.append(sequence)
                batch_labels.append(self.labels[idx])
            
            except Exception as e:
                # Skip failed samples
                print(f"✗ Error processing {self.video_paths[idx]}: {e}")
                continue
        
        if len(batch_sequences) > 0:
            # Convert to numpy arrays
            X = np.array(batch_sequences)
            y = keras.utils.to_categorical(
                batch_labels, 
                num_classes=len(self.data_loader.label_to_idx)
            )
            return X, y
        else:
            # Return empty batch if all samples failed
            return np.array([]), np.array([])
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch (optional, called by Keras)"""
        np.random.shuffle(self.indices)


class DataLoader:
    """Data loader for multi-lingual lip reading dataset"""
    
    def __init__(self, 
                 data_dir='./data/videos',
                 preprocessed_dir='./data/preprocessed',
                 languages=['kannada', 'hindi', 'english'],
                 batch_size=8,
                 validation_split=0.2):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing video files
            preprocessed_dir: Directory for preprocessed data
            languages: List of languages to load
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
        """
        self.data_dir = Path(data_dir)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.languages = languages
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Create preprocessed directory
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.video_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.class_counts = {}
        
        print("✓ DataLoader initialized")
    
    def scan_dataset(self):
        """Scan the dataset and build label mappings"""
        print("\nScanning dataset...")
        
        label_set = set()
        video_data = []
        
        # Scan each language directory
        for language in self.languages:
            lang_dir = self.data_dir / language
            
            if not lang_dir.exists():
                print(f"✗ Warning: Directory not found: {lang_dir}")
                continue
            
            # Check if there are word subdirectories or videos directly
            subdirs = [d for d in lang_dir.iterdir() if d.is_dir()]
            
            if subdirs:
                # New structure: language/word/videos
                for word_dir in subdirs:
                    word = word_dir.name
                    
                    # Get all video files in this word directory
                    video_files = list(word_dir.glob('*.mp4')) + \
                                 list(word_dir.glob('*.avi')) + \
                                 list(word_dir.glob('*.mov'))
                    
                    print(f"  {language}/{word}: {len(video_files)} videos")
                    
                    for video_path in video_files:
                        # Create unique label (language, word)
                        label = f"{language}_{word}"
                        label_set.add(label)
                        
                        video_data.append({
                            'path': str(video_path),
                            'label': label,
                            'language': language,
                            'word': word
                        })
            else:
                # Old structure: language/videos (for backward compatibility)
                video_files = list(lang_dir.glob('*.mp4')) + \
                             list(lang_dir.glob('*.avi')) + \
                             list(lang_dir.glob('*.mov'))
                
                print(f"  {language}: {len(video_files)} videos")
                
                for video_path in video_files:
                    # Extract word from filename (format: word_speakerID.mp4)
                    filename = video_path.stem
                    word = filename.rsplit('_', 1)[0]  # Remove speaker ID
                    
                    # Create unique label (language, word)
                    label = f"{language}_{word}"
                    label_set.add(label)
                    
                    video_data.append({
                        'path': str(video_path),
                        'label': label,
                        'language': language,
                        'word': word
                    })
        
        # Create label mappings
        sorted_labels = sorted(label_set)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Store video paths and labels
        self.video_paths = [item['path'] for item in video_data]
        self.labels = [self.label_to_idx[item['label']] for item in video_data]
        
        # Count samples per class
        self.class_counts = {}
        for item in video_data:
            label = item['label']
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
        
        print(f"\n✓ Dataset scanned:")
        print(f"  Total videos: {len(self.video_paths)}")
        print(f"  Total classes: {len(self.label_to_idx)}")
        print(f"  Languages: {len(self.languages)}")
        
        return len(self.label_to_idx)
    
    def get_class_info(self):
        """Get information about classes"""
        return {
            'num_classes': len(self.label_to_idx),
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'class_counts': self.class_counts
        }
    
    def save_class_mapping(self, filepath='./models/class_mapping.json'):
        """Save class mapping to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        mapping = {
            'label_to_idx': self.label_to_idx,
            'idx_to_label': {int(k): v for k, v in self.idx_to_label.items()},
            'class_counts': self.class_counts,
            'languages': self.languages
        }
        
        with open(filepath, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"✓ Class mapping saved to {filepath}")
    
    def load_class_mapping(self, filepath='./models/class_mapping.json'):
        """Load class mapping from file"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Class mapping not found: {filepath}")
        
        with open(filepath, 'r') as f:
            mapping = json.load(f)
        
        self.label_to_idx = mapping['label_to_idx']
        self.idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}
        self.class_counts = mapping['class_counts']
        self.languages = mapping['languages']
        
        print(f"✓ Class mapping loaded from {filepath}")
        return len(self.label_to_idx)
    
    def split_dataset(self, random_state=42):
        """Split dataset into training and validation sets"""
        if len(self.video_paths) == 0:
            raise ValueError("No data loaded. Call scan_dataset() first.")
        
        # Stratified split to maintain class distribution
        X_train, X_val, y_train, y_val = train_test_split(
            self.video_paths,
            self.labels,
            test_size=self.validation_split,
            stratify=self.labels,
            random_state=random_state
        )
        
        print(f"\n✓ Dataset split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        return (X_train, y_train), (X_val, y_val)
    
    def create_data_generator(self, video_paths, labels, preprocessor, augment=False):
        """
        Create a data generator for training with length support
        
        Args:
            video_paths: List of video file paths
            labels: List of labels (indices)
            preprocessor: VideoPreprocessor instance
            augment: Whether to apply augmentation
        
        Returns:
            generator: Countable data generator (CountableGenerator instance)
        """
        # Return the countable generator instance
        return CountableGenerator(
            self, video_paths, labels, preprocessor, augment, self.batch_size
        )
    
    def preprocess_and_save(self, preprocessor, force=False):
        """
        Preprocess all videos and save to disk
        
        Args:
            preprocessor: VideoPreprocessor instance
            force: Force reprocessing even if cached files exist
        """
        print("\nPreprocessing videos...")
        
        for i, (video_path, label_idx) in enumerate(zip(self.video_paths, self.labels)):
            # Generate output path
            label = self.idx_to_label[label_idx]
            language, word = label.split('_', 1)
            
            output_dir = self.preprocessed_dir / language
            output_dir.mkdir(parents=True, exist_ok=True)
            
            video_name = Path(video_path).stem
            output_path = output_dir / f"{video_name}.npy"
            
            # Skip if already processed
            if output_path.exists() and not force:
                continue
            
            try:
                # Process video
                sequence = preprocessor.process_video(video_path)
                
                # Save to disk
                np.save(output_path, sequence)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(self.video_paths)} videos")
            
            except Exception as e:
                print(f"✗ Failed to process {video_path}: {e}")
        
        print(f"✓ Preprocessing complete")
    
    def load_preprocessed_data(self):
        """Load preprocessed data from disk"""
        print("\nLoading preprocessed data...")
        
        sequences = []
        labels = []
        
        for video_path, label_idx in zip(self.video_paths, self.labels):
            # Generate preprocessed file path
            label = self.idx_to_label[label_idx]
            language, word = label.split('_', 1)
            
            video_name = Path(video_path).stem
            preprocessed_path = self.preprocessed_dir / language / f"{video_name}.npy"
            
            if not preprocessed_path.exists():
                print(f"✗ Preprocessed file not found: {preprocessed_path}")
                continue
            
            try:
                sequence = np.load(preprocessed_path)
                sequences.append(sequence)
                labels.append(label_idx)
            except Exception as e:
                print(f"✗ Error loading {preprocessed_path}: {e}")
        
        print(f"✓ Loaded {len(sequences)} preprocessed sequences")
        
        return np.array(sequences), np.array(labels)
    
    def get_dataset_statistics(self):
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.video_paths),
            'num_classes': len(self.label_to_idx),
            'num_languages': len(self.languages),
            'samples_per_language': {},
            'samples_per_class': self.class_counts,
            'min_samples_per_class': min(self.class_counts.values()) if self.class_counts else 0,
            'max_samples_per_class': max(self.class_counts.values()) if self.class_counts else 0,
            'avg_samples_per_class': np.mean(list(self.class_counts.values())) if self.class_counts else 0
        }
        
        # Count samples per language
        for language in self.languages:
            count = sum(1 for label in self.class_counts.keys() if label.startswith(language))
            stats['samples_per_language'][language] = count
        
        return stats
    
    def add_new_language(self, language_name):
        """Add a new language to the dataset"""
        if language_name in self.languages:
            print(f"✗ Language '{language_name}' already exists")
            return False
        
        # Create directory for new language
        lang_dir = self.data_dir / language_name
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessed_dir = self.preprocessed_dir / language_name
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        self.languages.append(language_name)
        
        print(f"✓ Added new language: {language_name}")
        print(f"  Video directory: {lang_dir}")
        print(f"  Place training videos in this directory with format: word_speakerID.mp4")
        
        return True
    
    def add_new_word(self, word, language, video_path):
        """Add a new word to the dataset"""
        if language not in self.languages:
            print(f"✗ Language '{language}' not found")
            return False
        
        # Copy or move video to appropriate directory
        lang_dir = self.data_dir / language
        
        # Generate filename
        speaker_id = len(list(lang_dir.glob(f"{word}_*.mp4"))) + 1
        dest_path = lang_dir / f"{word}_{speaker_id:03d}.mp4"
        
        # Copy video file
        import shutil
        shutil.copy(video_path, dest_path)
        
        print(f"✓ Added new word: {word} ({language})")
        print(f"  Saved to: {dest_path}")
        
        # Rescan dataset
        self.scan_dataset()
        
        return True


def test_data_loader():
    """Test the data loader"""
    print("Testing DataLoader...")
    
    # Initialize loader
    loader = DataLoader(
        data_dir='./data/videos',
        languages=['kannada', 'hindi', 'english']
    )
    
    print("✓ DataLoader initialized")
    
    # Test adding new language
    loader.add_new_language('tamil')
    
    print("\n✓ DataLoader test complete")


if __name__ == "__main__":
    test_data_loader()
