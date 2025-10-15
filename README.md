# 🗣️ Multi-Lingual Lip Reading System

**Visual Speech Recognition for Indian Languages**

A complete deep learning-based lip reading system that recognizes spoken words from **visual information only** (no audio required). Supports **Kannada**, **Hindi**, and **English** with real-time prediction capabilities.

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 What Is This?

A **lip reading system** that recognizes spoken words by analyzing lip movements **without any audio input**. Perfect for:
- 🔇 Silent communication
- 🌐 Multi-lingual applications  
- 🎓 Research in visual speech recognition
- ♿ Accessibility solutions

---

## ✨ Key Features

- 👄 **Visual-Only Processing**: No microphone needed, uses only lip movements
- 🌐 **Multi-Language Support**: Kannada (ಕನ್ನಡ), Hindi (हिन्दी), English
- 🚀 **GPU Accelerated**: Fast training and inference with NVIDIA GPUs
- 📹 **Real-Time Prediction**: Live webcam-based lip reading with 20-30 FPS
- 🎨 **User-Friendly GUIs**: 
  - Training GUI with recording and metrics
  - Prediction GUI with stabilized output
- 📊 **High Accuracy**: 80-95% with sufficient training data
- 🎯 **Stable Predictions**: Advanced stabilization prevents flickering
- 📈 **Complete Monitoring**: TensorBoard integration

---

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
# Create conda environment
conda create -n lipread_gpu python=3.9.18 -y
conda activate lipread_gpu

# Install TensorFlow GPU
pip install tensorflow-gpu==2.6.0

# Install other dependencies
pip install opencv-python mediapipe numpy scikit-learn pillow matplotlib scipy pyyaml albumentations
```

### 2. Run Real-Time Prediction
```powershell
python predict_gui.py
```

### 3. Train Your Own Model
```powershell
python train_gui_with_recording.py
```

---

## 🏗️ System Architecture

```
Input Video → Face Detection (MediaPipe) → Lip Landmark Tracking (31 points) → 
Geometric Feature Extraction (330 features) → Temporal Smoothing → 
Bidirectional LSTM + Attention → Classification → Stabilized Prediction
```

**Model**: Bidirectional LSTM with Attention Mechanism  
**Input**: 75 frames (3 seconds) × 330 features per frame  
**Parameters**: ~1.8 million  
**Size**: 7.1 MB

---

## 🎓 How It Works

### 1. Video Preprocessing
- Extract frames at 25 FPS
- Detect face using MediaPipe FaceMesh
- Track 31 lip landmarks (20 outer + 11 inner)
- Apply temporal smoothing (EMA)

### 2. Feature Extraction
- Compute 110 geometric features:
  - Normalized landmark coordinates
  - Mouth dimensions (width, height, aspect ratio)
  - Distances, angles, and curvature
  - Lip contour area
- Add temporal features (velocity + acceleration)
- **Total: 330 features per frame**

### 3. Deep Learning Model
- **Architecture**: Bidirectional LSTM with Attention
- **Input**: Sequence of 75 frames (3 seconds)
- **Layers**:
  - Dense feature processing (256, 128)
  - Bidirectional LSTM (256, 128) for temporal modeling
  - Attention mechanism for focus
  - Classification output (softmax)
- **Training**: Adam optimizer, early stopping, learning rate reduction

### 4. Prediction Stabilization
- Frequency-based voting (analyzes last 5 predictions)
- Confidence threshold filtering (65% default)
- Stability counter (requires 2 consecutive confirmations)
- **Result**: Smooth, reliable real-time predictions without flickering

---

## 📁 Project Structure

```
multi-lingual-lip-reading/
├── configs/
│   └── config.yaml              # System configuration
├── data/
│   ├── videos/                  # Training videos (by language/word)
│   │   ├── kannada/
│   │   │   ├── ನಮಸ್ಕಾರ/         # Word folder with 30-50 videos
│   │   │   └── ರಾಮ/
│   │   ├── hindi/
│   │   └── english/
│   └── preprocessed/            # Processed features (.npy files)
├── models/
│   ├── best_model.h5           # Trained model (7.1 MB)
│   └── class_mapping.json      # Label mappings
├── logs/
│   ├── tensorboard/            # TensorBoard logs
│   └── training/               # Training logs
├── outputs/
│   ├── predictions/            # Prediction results
│   └── recordings/             # Recorded videos
├── src/
│   ├── data_loader.py          # Data loading utilities
│   ├── model.py                # LSTM + Attention model
│   ├── preprocessor.py         # Feature extraction
│   └── utils.py                # Helper functions
├── predict_gui.py              # Real-time prediction GUI
├── train_gui_with_recording.py # Training + recording GUI
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## 📊 Performance

### Training Time (50 epochs, ~100 videos)
| GPU | Time |
|-----|------|
| GTX 1660 Ti | ~10 min |
| RTX 3060 | ~5 min |
| CPU | ~2.5 hours ⚠️ |

### Prediction Performance
| System | FPS | Latency |
|--------|-----|---------|
| RTX 3060 | 25-30 | ~120ms |
| GTX 1660 Ti | 20-25 | ~150ms |
| CPU | 5-8 | ~500ms |

### Accuracy
| Training Videos/Word | Validation Accuracy | Notes |
|---------------------|---------------------|-------|
| 10-20 | 60-70% | Poor - Need more data |
| 30-50 | 80-90% | Good - Recommended |
| 50+ | 85-95% | Excellent - Best results |

---

## 🎯 Training Your Own Model

### Quick Guide

1. **Prepare Training Data**
   - Record 30-50 videos per word (3-5 seconds each)
   - Use consistent lighting and camera position
   - Keep face centered with clear view of lips

2. **Preprocess Data**
   ```powershell
   python train_gui_with_recording.py
   # Click "⚙️ Preprocess Data"
   ```

3. **Train Model**
   ```powershell
   # In the same GUI, click "🚀 Start Training"
   # Configure epochs (50-100 recommended)
   # Monitor progress in GUI and TensorBoard
   ```

4. **Test Predictions**
   ```powershell
   python predict_gui.py
   # Load your trained model
   # Start camera and make predictions!
   ```

---

## 📸 Visual Features

### Real-Time Prediction GUI
- ✅ Enhanced lip landmark tracking (31 points visualized)
- ✅ Stable predictions with `[STABLE]` indicator
- ✅ Green/Yellow color coding for confidence
- ✅ Real-time feedback: Buffer, Opening, Movement
- ✅ Smooth contours without jitter

### Training GUI
- ✅ Video recording interface
- ✅ Real-time preprocessing progress
- ✅ Training metrics visualization
- ✅ TensorBoard integration
- ✅ Automatic best model saving

---

## 🛠️ System Requirements

### Hardware
- **CPU**: Intel i5 / AMD Ryzen 5 or better
- **RAM**: 8 GB minimum, 16 GB recommended
- **GPU**: NVIDIA GPU with CUDA (GTX 1650+) - *Optional but highly recommended*
- **Webcam**: Any webcam or phone camera (via DroidCam/OBS)
- **Storage**: 5 GB free space

### Software
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS
- **Python**: 3.9.18 (recommended)
- **CUDA**: 11.3 (for GPU acceleration)
- **cuDNN**: 8.2.1 (for GPU acceleration)

---

## 🔧 Troubleshooting

### Common Issues

**❌ Camera not detected**
- Try different camera indices (0, 1, 2) in GUI
- Use DroidCam + OBS Virtual Camera
- Check camera permissions in Windows Settings
- Close other apps using the camera

**❌ Low training accuracy (<70%)**
- Record more training videos (50+ per word recommended)
- Improve lighting and video quality
- Train for more epochs (100+)
- Verify preprocessing completed successfully

**❌ GPU not detected**
```powershell
# Reinstall CUDA/cuDNN via conda
conda install cudatoolkit=11.3 cudnn=8.2.1 -c conda-forge -y

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**❌ Predictions flickering between words**
- ✅ Already fixed! Latest version includes stabilization system
- Look for `[STABLE]` indicator (green) vs `[UPDATING...]` (yellow)
- Only trust predictions with `[STABLE]` badge

**❌ Import errors in VS Code**
- These are just linting warnings
- Make sure to activate conda environment before running:
  ```powershell
  conda activate lipread_gpu
  python predict_gui.py  # Works fine!
  ```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[MediaPipe](https://mediapipe.dev/)** - Robust face and landmark detection
- **[TensorFlow](https://www.tensorflow.org/)** - Deep learning framework
- **[OpenCV](https://opencv.org/)** - Computer vision utilities
- **Community** - Feedback, contributions, and support

---

## 📧 Contact

**Author**: Chinthan  
**Repository**: [https://github.com/ChinthanEdu/Lip](https://github.com/ChinthanEdu/Lip)  
**Issues**: [GitHub Issues](https://github.com/ChinthanEdu/Lip/issues)

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

Stars help others discover this project and motivate continued development.

---

## 🎯 Key Highlights

- ✅ **Production Ready**: Stable predictions, professional appearance
- ✅ **Well Documented**: Comprehensive guides for all features
- ✅ **User Friendly**: Simple GUIs, no command-line expertise needed
- ✅ **Extensible**: Easy to add new languages and words
- ✅ **GPU Accelerated**: Fast training and inference
- ✅ **Open Source**: Free to use, modify, and distribute

---

**Made with ❤️ for accessible communication**

*Last Updated: January 2025 | Version: 2.0*
