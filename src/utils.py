"""
Utility Functions for Multi-Lingual Lip Reading System
Author: AI Assistant
Date: October 2025
"""

import os
import sys
import yaml
import urllib.request
import bz2
import tensorflow as tf
import numpy as np
from pathlib import Path


def create_directory_structure(base_path='./'):
    """Create the required directory structure for the project"""
    directories = [
        'models',
        'data',
        'data/videos',
        'data/videos/kannada',
        'data/videos/hindi',
        'data/videos/english',
        'data/preprocessed',
        'data/preprocessed/kannada',
        'data/preprocessed/hindi',
        'data/preprocessed/english',
        'logs',
        'logs/training',
        'logs/tensorboard',
        'configs',
        'outputs',
        'outputs/predictions',
        'outputs/recordings'
    ]
    
    base = Path(base_path)
    for directory in directories:
        dir_path = base / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    return True


def download_shape_predictor(models_dir='./models'):
    """Download dlib shape predictor model if not exists"""
    model_path = Path(models_dir) / 'shape_predictor_68_face_landmarks.dat'
    
    if model_path.exists():
        print(f"✓ Shape predictor already exists at {model_path}")
        return str(model_path)
    
    print("Downloading dlib shape predictor model...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_path = Path(models_dir) / 'shape_predictor_68_face_landmarks.dat.bz2'
    
    try:
        # Create models directory
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        
        # Download compressed file
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, compressed_path)
        
        # Decompress
        print("Decompressing...")
        with bz2.open(compressed_path, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Remove compressed file
        compressed_path.unlink()
        
        print(f"✓ Downloaded and extracted shape predictor to {model_path}")
        return str(model_path)
    
    except Exception as e:
        print(f"✗ Error downloading shape predictor: {e}")
        print("Please download manually from:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return None


def setup_gpu(device='GPU', memory_growth=True):
    """
    Configure TensorFlow GPU settings
    
    Args:
        device: 'GPU' or 'CPU'
        memory_growth: Enable memory growth to avoid OOM errors
    """
    if device.upper() == 'GPU':
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Enable memory growth
                if memory_growth:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"✓ GPU(s) available: {len(gpus)}")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")
                
                return True
            except RuntimeError as e:
                print(f"✗ GPU setup error: {e}")
                return False
        else:
            print("✗ No GPU found. Falling back to CPU.")
            return False
    else:
        # Force CPU
        tf.config.set_visible_devices([], 'GPU')
        print("✓ Using CPU for computation")
        return True


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"✗ Config file not found: {config_path}")
        return get_default_config()
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config from {config_path}")
        return config
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return get_default_config()


def save_config(config, config_path='configs/config.yaml'):
    """Save configuration to YAML file"""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✓ Saved config to {config_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving config: {e}")
        return False


def get_default_config():
    """Return default configuration"""
    config = {
        'model': {
            'sequence_length': 75,
            'frame_height': 100,
            'frame_width': 100,
            'channels': 3,
            'lstm_units': [256, 128],
            'dense_units': [512, 256],
            'dropout_rate': 0.5
        },
        'training': {
            'batch_size': 8,
            'epochs': 100,
            'learning_rate': 0.001,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 5,
            'reduce_lr_factor': 0.5,
            'validation_split': 0.2
        },
        'preprocessing': {
            'target_fps': 25,
            'mouth_roi_padding': 0.2,
            'normalize': True,
            'augmentation': {
                'enabled': True,
                'brightness_range': [0.8, 1.2],
                'contrast_range': [0.8, 1.2],
                'rotation_range': 5,
                'noise_std': 0.01
            }
        },
        'languages': ['kannada', 'hindi', 'english'],
        'paths': {
            'data_dir': './data',
            'models_dir': './models',
            'logs_dir': './logs',
            'shape_predictor': './models/shape_predictor_68_face_landmarks.dat'
        }
    }
    return config


def ensure_shape_predictor():
    """Ensure shape predictor model exists (optional - only needed for dlib)"""
    try:
        config = load_config()
        predictor_path = config['paths']['shape_predictor']
        
        if not Path(predictor_path).exists():
            print("Shape predictor not found (optional - only needed if using dlib)")
            print("Skipping download. MediaPipe will be used for face detection.")
            return None
        
        return predictor_path
    except Exception as e:
        print(f"Warning: Could not load shape predictor: {e}")
        print("Using MediaPipe for face detection instead.")
        return None


def get_available_languages(data_dir='./data/videos'):
    """Get list of available languages from data directory"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return []
    
    languages = [d.name for d in data_path.iterdir() if d.is_dir()]
    return sorted(languages)


def count_videos_per_language(data_dir='./data/videos'):
    """Count number of videos for each language"""
    data_path = Path(data_dir)
    counts = {}
    
    for lang_dir in data_path.iterdir():
        if lang_dir.is_dir():
            video_files = list(lang_dir.glob('*.mp4')) + \
                         list(lang_dir.glob('*.avi')) + \
                         list(lang_dir.glob('*.mov'))
            counts[lang_dir.name] = len(video_files)
    
    return counts


def format_time(seconds):
    """Format seconds to human-readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def calculate_model_size(model):
    """Calculate model size in MB"""
    total_params = model.count_params()
    # Assuming float32 (4 bytes per parameter)
    size_mb = (total_params * 4) / (1024 ** 2)
    return size_mb


def print_system_info():
    """Print system and environment information"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    print(f"Python Version: {sys.version}")
    print(f"TensorFlow Version: {tf.__version__}")
    
    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs Available: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("GPUs Available: 0 (CPU only)")
    
    # Memory info
    if gpus:
        for gpu in gpus:
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"GPU Details: {gpu_details}")
            except:
                pass
    
    print("="*60 + "\n")


def validate_video_file(video_path):
    """Validate if video file is readable"""
    import cv2
    
    if not Path(video_path).exists():
        return False, "File does not exist"
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        ret, frame = cap.read()
        if not ret:
            return False, "Cannot read video frames"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        cap.release()
        
        if frame_count < 10:
            return False, "Video too short (less than 10 frames)"
        
        return True, f"Valid video: {frame_count} frames @ {fps} FPS"
    
    except Exception as e:
        return False, f"Error validating video: {str(e)}"


def create_class_mapping(languages, words_per_language):
    """
    Create class mapping for multi-language model
    
    Args:
        languages: List of language names
        words_per_language: Dict of {language: [word1, word2, ...]}
    
    Returns:
        class_to_idx: Dict mapping (language, word) -> class_index
        idx_to_class: Dict mapping class_index -> (language, word)
    """
    class_to_idx = {}
    idx_to_class = {}
    
    idx = 0
    for language in languages:
        if language in words_per_language:
            for word in words_per_language[language]:
                class_to_idx[(language, word)] = idx
                idx_to_class[idx] = (language, word)
                idx += 1
    
    return class_to_idx, idx_to_class


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...\n")
    
    print_system_info()
    
    print("Creating directory structure...")
    create_directory_structure()
    
    print("\nSetting up GPU...")
    setup_gpu('GPU')
    
    print("\nLoading default config...")
    config = get_default_config()
    print(f"Config loaded: {len(config)} sections")
    
    print("\nEnsuring shape predictor...")
    ensure_shape_predictor()
    
    print("\n✓ All utility functions tested successfully!")
