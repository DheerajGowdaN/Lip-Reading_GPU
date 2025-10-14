"""
Initialization script for Multi-Lingual Lip Reading System
Run this once after installation to set up the environment
Author: AI Assistant
Date: October 2025
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import *


def main():
    """Main initialization function"""
    
    print("\n" + "="*70)
    print("MULTI-LINGUAL LIP READING SYSTEM - INITIALIZATION")
    print("="*70 + "\n")
    
    # Step 1: Check Python version
    print("Step 1: Checking Python version...")
    import sys
    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor != 10:
        print("  ⚠ Warning: This project is designed for Python 3.10.x")
        print("  ⚠ You're using a different version. Some packages may not work correctly.")
    else:
        print("  ✓ Python version is correct")
    
    print()
    
    # Step 2: Check TensorFlow installation
    print("Step 2: Checking TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__} installed")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ {len(gpus)} GPU(s) detected:")
            for i, gpu in enumerate(gpus):
                print(f"    - GPU {i}: {gpu.name}")
        else:
            print("  ⚠ No GPU detected. Training will use CPU (slower).")
            print("    For GPU support, install CUDA Toolkit 11.8 and cuDNN 8.6")
    except ImportError:
        print("  ✗ TensorFlow not installed!")
        print("    Please run: pip install -r requirements.txt")
        return False
    
    print()
    
    # Step 3: Check other dependencies
    print("Step 3: Checking other dependencies...")
    dependencies = [
        ('cv2', 'opencv-python'),
        ('dlib', 'dlib'),
        ('mediapipe', 'mediapipe'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'matplotlib'),
    ]
    
    all_installed = True
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name} installed")
        except ImportError:
            print(f"  ✗ {package_name} not installed")
            all_installed = False
    
    if not all_installed:
        print("\n  Some dependencies are missing. Please run:")
        print("    pip install -r requirements.txt")
        return False
    
    print()
    
    # Step 4: Create directory structure
    print("Step 4: Creating directory structure...")
    create_directory_structure()
    
    print()
    
    # Step 5: Download shape predictor
    print("Step 5: Downloading facial landmark predictor...")
    predictor_path = download_shape_predictor()
    
    if predictor_path:
        print(f"  ✓ Shape predictor ready at: {predictor_path}")
    else:
        print("  ⚠ Shape predictor download failed.")
        print("    Please download manually from:")
        print("    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("    Extract and place in: ./models/")
    
    print()
    
    # Step 6: Create default config
    print("Step 6: Creating default configuration...")
    config = get_default_config()
    save_config(config)
    print("  ✓ Configuration saved to: ./configs/config.yaml")
    
    print()
    
    # Step 7: Test imports
    print("Step 7: Testing project imports...")
    try:
        from src.model import LipReadingModel
        print("  ✓ Model module imported")
        
        from src.preprocessor import VideoPreprocessor
        print("  ✓ Preprocessor module imported")
        
        from src.data_loader import DataLoader
        print("  ✓ Data loader module imported")
        
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False
    
    print()
    
    # Step 8: Print system information
    print("Step 8: System information")
    print_system_info()
    
    # Step 9: Final instructions
    print("\n" + "="*70)
    print("INITIALIZATION COMPLETE!")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("  1. Prepare your training data in: ./data/videos/")
    print("     Format: ./data/videos/{language}/{word}_{speaker_id}.mp4")
    print()
    print("  2. Train the model:")
    print("     python train_gui.py")
    print()
    print("  3. Run real-time prediction:")
    print("     python predict_gui.py")
    print()
    print("For detailed instructions, see:")
    print("  - README.md: Quick start guide")
    print("  - SETUP_GUIDE.md: Step-by-step setup")
    print("  - DOCUMENTATION.md: Complete documentation")
    print()
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n⚠ Initialization encountered errors. Please fix them before proceeding.\n")
        sys.exit(1)
    else:
        print("✓ System ready! You can now start training or predicting.\n")
        sys.exit(0)
