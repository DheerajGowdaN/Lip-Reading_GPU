# 🗣️ Multi-Lingual Lip Reading System# 🗣️ Multi-Lingual Lip Reading System



**Visual Speech Recognition for Indian Languages****Visual Speech Recognition for Indian Languages**



A complete deep learning-based lip reading system that recognizes spoken words from **visual information only** (no audio required). Supports **Kannada**, **Hindi**, and **English** with real-time prediction capabilities.A complete deep learning-based lip reading system that recognizes spoken words from **visual information only** (no audio required). Supports **Kannada**, **Hindi**, and **English** with real-time prediction capabilities.



[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6.0-orange.svg)](https://www.tensorflow.org/)[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6.0-orange.svg)](https://www.tensorflow.org/)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



------



## 🎯 What Is This?## 🎯 What Is This?



A **lip reading system** that recognizes spoken words by analyzing lip movements **without any audio input**. Perfect for:A **lip reading system** that recognizes spoken words by analyzing lip movements **without any audio input**. Perfect for:

- 🔇 Silent communication- 🔇 Silent communication

- 🌐 Multi-lingual applications  - 🌐 Multi-lingual applications

- 🎓 Research in visual speech recognition- 🎓 Research in visual speech recognition

- ♿ Accessibility solutions- ♿ Accessibility solutions



---## ✨ Key Features



## ✨ Key Features- 👄 **Visual-Only Processing**: No microphone needed, uses only lip movements

- 🌐 **Multi-Language Support**: Kannada (ಕನ್ನಡ), Hindi (हिन्दी), English

- 👄 **Visual-Only Processing**: No microphone needed, uses only lip movements- 🚀 **GPU Accelerated**: Fast training and inference with NVIDIA GPUs

- 🌐 **Multi-Language Support**: Kannada (ಕನ್ನಡ), Hindi (हिन्दी), English- 📹 **Real-Time Prediction**: Live webcam-based lip reading

- 🚀 **GPU Accelerated**: Fast training and inference with NVIDIA GPUs- 🎨 **User-Friendly GUIs**: 

- 📹 **Real-Time Prediction**: Live webcam-based lip reading with 20-30 FPS  - Training GUI with recording and metrics

- 🎨 **User-Friendly GUIs**:   - Prediction GUI with stabilization

  - Training GUI with recording and metrics- � **High Accuracy**: 80-95% with sufficient training data

  - Prediction GUI with stabilized output- 🎯 **Stable Predictions**: Advanced stabilization prevents flickering

- 📊 **High Accuracy**: 80-95% with sufficient training data- 📈 **Complete Monitoring**: TensorBoard integration for training visualization

- 🎯 **Stable Predictions**: Advanced stabilization prevents flickering

- 📈 **Complete Monitoring**: TensorBoard integration## 🏗️ System Architecture



---```

Input Video → Face Detection (MediaPipe) → Lip Landmark Tracking (31 points) → 

## 🚀 Quick StartGeometric Feature Extraction (330 features) → Temporal Smoothing → 

Bidirectional LSTM + Attention → Classification → Stabilized Prediction

### 1. Install Dependencies```

```powershell

# Create conda environment**Model**: Bidirectional LSTM with Attention Mechanism  

conda create -n lipread_gpu python=3.9.18 -y**Input**: 75 frames (3 seconds) × 330 features per frame  

conda activate lipread_gpu**Parameters**: ~1.8 million  

**Size**: 7.1 MB

# Install TensorFlow GPU

pip install tensorflow-gpu==2.6.0## 🏗️ How It Works



# Install other dependencies### Model Architecture

pip install opencv-python mediapipe numpy scikit-learn pillow matplotlib scipy pyyaml albumentations

``````- **3D Convolutional Layers**: Spatial-temporal feature extraction



### 2. Run Real-Time PredictionVideo → Face Detection → Lip Landmarks → Feature Extraction → - **Bidirectional LSTM**: Sequential modeling

```powershell

python predict_gui.py3D CNN + LSTM + Attention → Word Prediction- **Attention Mechanism**: Focus on important frames

```

```- **Dense Layers**: Final classification

### 3. Train Your Own Model

```powershell

python train_gui_with_recording.py

```**Model Architecture:**## 📋 Prerequisites



**📖 For detailed setup instructions, see [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)**- **3D CNN**: Extracts spatial-temporal features from lip movements



---- **Bidirectional LSTM**: Models sequential patterns- Python 3.10.0



## 🏗️ System Architecture- **Attention Mechanism**: Focuses on important frames- Windows OS (PowerShell)



```- **Dense Layers**: Final word classification- 8GB+ RAM recommended

Input Video → Face Detection (MediaPipe) → Lip Landmark Tracking (31 points) → 

Geometric Feature Extraction (330 features) → Temporal Smoothing → - NVIDIA GPU (optional, for faster training)

Bidirectional LSTM + Attention → Classification → Stabilized Prediction

```---- Webcam (for real-time prediction)



**Model**: Bidirectional LSTM with Attention Mechanism  

**Input**: 75 frames (3 seconds) × 330 features per frame  

**Parameters**: ~1.8 million  ## 📋 Quick Start## 🚀 Quick Start

**Size**: 7.1 MB



---

### System Requirements### 1. Clone/Download the Project

## 🎓 How It Works

- **OS:** Windows 10/11

### 1. Video Preprocessing

- Extract frames at 25 FPS- **RAM:** 8GB minimum (16GB recommended)Place the project in `d:\P\multi-lingual-lip-reading\`

- Detect face using MediaPipe FaceMesh

- Track 31 lip landmarks (20 outer + 11 inner)- **GPU:** NVIDIA GPU with 4GB+ VRAM (optional but recommended)

- Apply temporal smoothing (EMA)

- **Camera:** Webcam for predictions### 2. Create Virtual Environment

### 2. Feature Extraction

- Compute 110 geometric features:

  - Normalized landmark coordinates

  - Mouth dimensions (width, height, aspect ratio)### Installation (5 Minutes)```powershell

  - Distances, angles, and curvature

  - Lip contour areacd d:\P\multi-lingual-lip-reading

- Add temporal features (velocity + acceleration)

- **Total: 330 features per frame**1. **Install Miniconda** (if not installed):python -m venv venv



### 3. Deep Learning Model   - Download from: https://docs.conda.io/en/latest/miniconda.html.\venv\Scripts\Activate.ps1

- **Architecture**: Bidirectional LSTM with Attention

- **Input**: Sequence of 75 frames (3 seconds)   - Run installer (check "Add to PATH")```

- **Layers**:

  - Dense feature processing (256, 128)

  - Bidirectional LSTM (256, 128) for temporal modeling

  - Attention mechanism for focus2. **Create Environment:**### 3. Install Dependencies

  - Classification output (softmax)

- **Training**: Adam optimizer, early stopping, learning rate reduction   ```powershell



### 4. Prediction Stabilization   cd d:\P\multi-lingual-lip-reading```powershell

- Frequency-based voting (analyzes last 5 predictions)

- Confidence threshold filtering (65% default)   conda create -n lipread_gpu python=3.9 tensorflow-gpu=2.6.0 cudatoolkit=11.3.1 cudnn=8.2.1 -c conda-forge -ypython -m pip install --upgrade pip

- Stability counter (requires 2 consecutive confirmations)

- **Result**: Smooth, reliable real-time predictions without flickering   conda activate lipread_gpupip install -r requirements.txt



---   ``````



## 📁 Project Structure



```3. **Install Dependencies:****Note**: Installation may take 10-15 minutes depending on your internet connection.

multi-lingual-lip-reading/

├── configs/   ```powershell

│   └── config.yaml              # System configuration

├── data/   conda install -c conda-forge opencv=4.8.1 -y### 4. Verify Installation

│   ├── videos/                  # Training videos (by language/word)

│   │   ├── kannada/   pip install mediapipe==0.10.7 albumentations==1.3.1 matplotlib scikit-learn pandas keras==2.6.0

│   │   │   ├── ನಮಸ್ಕಾರ/         # Word folder with 30-50 videos

│   │   │   └── ರಾಮ/   ``````powershell

│   │   ├── hindi/

│   │   └── english/python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"

│   └── preprocessed/            # Processed features (.npy files)

├── models/4. **Verify GPU (if using GPU):**```

│   ├── best_model.h5           # Trained model (7.1 MB)

│   └── class_mapping.json      # Label mappings   ```powershell

├── logs/

│   ├── tensorboard/            # TensorBoard logs   python verify_gpu.py### 5. Prepare Your Data

│   └── training/               # Training logs

├── outputs/   ```

│   ├── predictions/            # Prediction results

│   └── recordings/             # Recorded videosOrganize your training videos in the following structure:

├── src/

│   ├── data_loader.py          # Data loading utilities5. **Initialize Project:**

│   ├── model.py                # LSTM + Attention model

│   ├── preprocessor.py         # Feature extraction   ```powershell```

│   └── utils.py                # Helper functions

├── predict_gui.py              # Real-time prediction GUI   python initialize.pydata/videos/

├── train_gui_with_recording.py # Training + recording GUI

├── COMPLETE_SETUP_GUIDE.md    # Detailed documentation (START HERE!)   ```├── kannada/

├── PREDICTION_ENHANCEMENTS.md  # Lip tracking improvements

├── STABILIZATION_UPDATE.md     # Prediction stabilization details│   ├── namaste_001.mp4

└── README.md                   # This file

```✅ **Setup Complete!**│   ├── namaste_002.mp4



---│   └── dhanyavaada_001.mp4



## 📊 Performance---├── hindi/



### Training Time (50 epochs, ~100 videos)│   ├── namaste_001.mp4

| GPU | Time |

|-----|------|## 🎓 Training│   └── shukriya_001.mp4

| GTX 1660 Ti | ~10 min |

| RTX 3060 | ~5 min |└── english/

| CPU | ~2.5 hours ⚠️ |

### 1. Prepare Training Data    ├── hello_001.mp4

### Prediction Performance

| System | FPS | Latency |    └── thanks_001.mp4

|--------|-----|---------|

| RTX 3060 | 25-30 | ~120ms |Record or collect videos (20-50 per word):```

| GTX 1660 Ti | 20-25 | ~150ms |

| CPU | 5-8 | ~500ms |```



### Accuracydata/videos/**Naming Convention**: `{word}_{speaker_id}.mp4`

| Training Videos/Word | Validation Accuracy | Notes |

|---------------------|---------------------|-------|├── english/

| 10-20 | 60-70% | Poor - Need more data |

| 30-50 | 80-90% | Good - Recommended |│   ├── hello/### 6. Train the Model

| 50+ | 85-95% | Excellent - Best results |

│   │   ├── hello_001.mp4

---

│   │   ├── hello_002.mp4```powershell

## 🎯 Training Your Own Model

│   │   └── ... (20-50 videos)python train_gui.py

### Quick Guide

│   └── goodbye/```

1. **Prepare Training Data**

   - Record 30-50 videos per word (3-5 seconds each)├── hindi/

   - Use consistent lighting and camera position

   - Keep face centered with clear view of lips│   └── नमस्ते/**Training GUI Features**:



2. **Preprocess Data**└── kannada/- Select language or use all languages

   ```powershell

   python train_gui_with_recording.py    └── ನಮಸ್ಕಾರ/- Configure epochs, batch size, learning rate

   # Click "⚙️ Preprocess Data"

   ``````- Choose CPU or GPU for training



3. **Train Model**- Real-time metrics display (loss, accuracy, bias, variance)

   ```powershell

   # In the same GUI, click "🚀 Start Training"**Video Requirements:**- Visual graphs for training progress

   # Configure epochs (50-100 recommended)

   # Monitor progress in GUI and TensorBoard- Duration: 2-5 seconds- Add new languages and words on-the-fly

   ```

- Quality: 640x480 or higher

4. **Test Predictions**

   ```powershell- Content: Clear face/lip view### 7. Real-Time Prediction

   python predict_gui.py

   # Load your trained model- Quantity: **20-50 videos per word** (minimum)

   # Start camera and make predictions!

   ``````powershell



**📖 For detailed training guide, see [COMPLETE_SETUP_GUIDE.md - Model Training](COMPLETE_SETUP_GUIDE.md#model-training)**### 2. Train Modelpython predict_gui.py



---```



## 📸 Visual Features```powershell



### Real-Time Prediction GUIconda activate lipread_gpu**Prediction GUI Features**:

- ✅ Enhanced lip landmark tracking (31 points visualized)

- ✅ Stable predictions with `[STABLE]` indicatorpython train_gui_with_recording.py- Load trained model

- ✅ Green/Yellow color coding for confidence

- ✅ Real-time feedback: Buffer, Opening, Movement```- Select camera device

- ✅ Smooth contours without jitter

- Real-time lip reading

### Training GUI

- ✅ Video recording interface**In GUI:**- Display top-3 predictions with confidence

- ✅ Real-time preprocessing progress

- ✅ Training metrics visualization1. Set batch size (8 default)- Video recording capability

- ✅ TensorBoard integration

- ✅ Automatic best model saving2. Set epochs (50-100)- Confidence threshold adjustment



---3. Select GPU device



## 🛠️ System Requirements4. Enable mixed precision## 📁 Project Structure



### Hardware5. Click "Start Training"

- **CPU**: Intel i5 / AMD Ryzen 5 or better

- **RAM**: 8 GB minimum, 16 GB recommended```

- **GPU**: NVIDIA GPU with CUDA (GTX 1650+) - *Optional but highly recommended*

- **Webcam**: Any webcam or phone camera (via DroidCam/OBS)**Training Time:**multi-lingual-lip-reading/

- **Storage**: 5 GB free space

- GPU: 30-60 seconds per epoch├── src/                        # Source code

### Software

- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS- CPU: 3-5 minutes per epoch│   ├── model.py               # Deep learning model

- **Python**: 3.9.18 (recommended)

- **CUDA**: 11.3 (for GPU acceleration)│   ├── preprocessor.py        # Video preprocessing

- **cuDNN**: 8.2.1 (for GPU acceleration)

---│   ├── data_loader.py         # Data loading utilities

---

│   └── utils.py               # Utility functions

## 📚 Documentation

## 🔮 Prediction├── models/                     # Saved models

| Document | Description |

|----------|-------------|│   ├── best_model.h5          # Trained model

| **[COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)** | 📖 Full installation, training, and usage guide **(START HERE!)** |

| **[PREDICTION_ENHANCEMENTS.md](PREDICTION_ENHANCEMENTS.md)** | 🎯 Lip tracking and ROI improvements |### Real-Time Lip Reading│   ├── class_mapping.json     # Class mappings

| **[STABILIZATION_UPDATE.md](STABILIZATION_UPDATE.md)** | 🎨 Prediction stabilization system details |

| **[GITHUB_UPLOAD_GUIDE.md](GITHUB_UPLOAD_GUIDE.md)** | 📤 Repository management and upload |│   └── shape_predictor_68_face_landmarks.dat



---```powershell├── data/                       # Training data



## 🔧 Troubleshootingconda activate lipread_gpu│   ├── videos/                # Raw videos



### Common Issuespython predict_gui.py│   │   ├── kannada/



**❌ Camera not detected**```│   │   ├── hindi/

- Try different camera indices (0, 1, 2) in GUI

- Use DroidCam + OBS Virtual Camera│   │   └── english/

- Check camera permissions in Windows Settings

- Close other apps using the camera**In GUI:**│   └── preprocessed/          # Preprocessed sequences



**❌ Low training accuracy (<70%)**1. Test camera connection├── logs/                       # Training logs

- Record more training videos (50+ per word recommended)

- Improve lighting and video quality2. Click "Start Recording"│   ├── training/

- Train for more epochs (100+)

- Verify preprocessing completed successfully3. Speak a trained word│   └── tensorboard/



**❌ GPU not detected**4. View prediction + confidence├── outputs/                    # Predictions and recordings

```powershell

# Reinstall CUDA/cuDNN via conda│   ├── predictions/

conda install cudatoolkit=11.3 cudnn=8.2.1 -c conda-forge -y

**Tips:**│   └── recordings/

# Verify GPU

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"- Face camera directly├── configs/                    # Configuration files

```

- Good lighting│   └── config.yaml

**❌ Predictions flickering between words**

- ✅ Already fixed! Latest version includes stabilization system- Clear lip movements├── train_gui.py               # Training GUI

- Look for `[STABLE]` indicator (green) vs `[UPDATING...]` (yellow)

- Only trust predictions with `[STABLE]` badge- 1-2 feet from camera├── predict_gui.py             # Prediction GUI



**📖 For more troubleshooting, see [COMPLETE_SETUP_GUIDE.md - Troubleshooting](COMPLETE_SETUP_GUIDE.md#troubleshooting)**├── requirements.txt           # Python dependencies



------├── SETUP_GUIDE.md            # Setup instructions



## 🤝 Contributing├── DOCUMENTATION.md          # Complete documentation



Contributions are welcome! To contribute:## 📊 Results└── README.md                 # This file



1. Fork the repository```

2. Create a feature branch

   ```bashWith **30-50 videos per word**:

   git checkout -b feature/AmazingFeature

   ```- Training Accuracy: 85-95%## 🎓 Usage Guide

3. Commit your changes

   ```bash- Validation Accuracy: 75-90%

   git commit -m 'Add some AmazingFeature'

   ```- Real-Time Predictions: 2-3 FPS### Training a Model

4. Push to the branch

   ```bash

   git push origin feature/AmazingFeature

   ```**Important:** Model accuracy depends on training data quantity and quality!1. **Launch Training GUI**: `python train_gui.py`

5. Open a Pull Request



---

---2. **Configure Data**:

## 📄 License

   - Click "Browse" to select your video directory

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔧 Troubleshooting   - Click "Scan Dataset" to analyze your data

---

   - The system will display total videos and classes

## 🙏 Acknowledgments

| Issue | Solution |

- **[MediaPipe](https://mediapipe.dev/)** - Robust face and landmark detection

- **[TensorFlow](https://www.tensorflow.org/)** - Deep learning framework|-------|----------|3. **Configure Training**:

- **[OpenCV](https://opencv.org/)** - Computer vision utilities

- **Community** - Feedback, contributions, and support| GPU not detected | Run `verify_gpu.py`, check NVIDIA drivers |   - Select Device: GPU (recommended) or CPU



---| Out of memory | Reduce batch size (8→4→2) |   - Set Epochs: 50-100 for good results



## 📧 Contact| Poor accuracy | Need 20-50 videos per word minimum |   - Set Batch Size: 8 (reduce if GPU memory issues)



**Author**: Chinthan  | Camera not working | Try camera index 1 or 2 (for external/phone) |   - Set Learning Rate: 0.001 (default)

**Repository**: [https://github.com/ChinthanEdu/Lip](https://github.com/ChinthanEdu/Lip)  

**Issues**: [GitHub Issues](https://github.com/ChinthanEdu/Lip/issues)| Import errors | Verify environment: `conda activate lipread_gpu` |



---4. **Start Training**:



## ⭐ Star History---   - Click "Start Training"



If you find this project helpful, please consider giving it a star! ⭐   - Monitor real-time metrics:



Stars help others discover this project and motivate continued development.## 📚 Documentation     - **Loss**: Should decrease over time



---     - **Accuracy**: Should increase over time



## 🎯 Key Highlights- **📖 [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** ← **READ THIS FIRST!**     - **Bias**: Train_acc - Val_acc (should be low)



- ✅ **Production Ready**: Stable predictions, professional appearance  - Complete setup instructions     - **Variance**: Stability of predictions

- ✅ **Well Documented**: Comprehensive guides for all features

- ✅ **User Friendly**: Simple GUIs, no command-line expertise needed  - Step-by-step operation guide   - Training stops automatically when validation loss stops improving

- ✅ **Extensible**: Easy to add new languages and words

- ✅ **GPU Accelerated**: Fast training and inference  - Troubleshooting solutions

- ✅ **Open Source**: Free to use, modify, and distribute

  - Advanced configuration5. **Save Model**:

---

   - Model is auto-saved as `models/best_model.h5`

**Made with ❤️ for accessible communication**

- **📝 LICENSE** - MIT License   - Manually save with "Save Model" button

*Last Updated: January 2025 | Version: 2.0*



---### Real-Time Prediction



## 🎯 Project Structure1. **Launch Prediction GUI**: `python predict_gui.py`



```2. **Load Model**:

multi-lingual-lip-reading/   - Browse and select trained model (`.h5` file)

├── src/                      # Source code   - Click "Load Model"

│   ├── model.py             # Neural network   - Verify model info is displayed

│   ├── preprocessor.py      # Video processing

│   ├── data_loader.py       # Data loading3. **Start Camera**:

│   └── utils.py             # Utilities   - Set Camera ID (usually 0 for default webcam)

├── configs/   - Adjust Confidence Threshold (0.5 recommended)

│   └── config.yaml          # Configuration   - Click "Start Camera"

├── data/

│   ├── videos/              # Training videos4. **Perform Lip Reading**:

│   └── preprocessed/        # Processed data   - Position your face in front of camera

├── models/                   # Saved models   - Ensure good lighting

├── logs/                     # Training logs   - Speak words clearly (facing camera)

├── train_gui_with_recording.py  # Training GUI   - System displays:

├── predict_gui.py               # Prediction GUI     - Current prediction

├── verify_gpu.py                # GPU verification     - Confidence score

└── COMPLETE_GUIDE.md            # Full documentation     - Top-3 predictions

```

5. **Record Video** (optional):

---   - Click "Record Video" to save session

   - Video saved in `outputs/recordings/`

## 🚀 Quick Commands

## 🛠️ Advanced Configuration

```powershell

# Activate environmentEdit `configs/config.yaml` to customize:

conda activate lipread_gpu

- Model architecture (LSTM units, dense layers)

# Verify GPU- Training parameters (batch size, learning rate)

python verify_gpu.py- Preprocessing settings (FPS, augmentation)

- Data paths

# Train model

python train_gui_with_recording.py## 📊 Training Tips



# Make predictions### For Best Results:

python predict_gui.py

1. **Data Quality**:

# View logs   - Good lighting in videos

tensorboard --logdir=logs/tensorboard   - Clear face visibility

```   - Multiple speakers per word (3-5+)

   - Consistent video quality

---

2. **Data Quantity**:

## 🎓 How to Use   - Minimum 10 videos per word recommended

   - More diverse speakers = better generalization

### For First-Time Users   - Balance classes (similar samples per word)

1. ✅ Install Miniconda

2. ✅ Create conda environment3. **Training Strategy**:

3. ✅ Verify GPU (optional)   - Start with fewer epochs (20-30) to test

4. ✅ Record 20-50 videos per word   - Use GPU for faster training (10-20x speedup)

5. ✅ Train model using GUI   - Monitor bias-variance:

6. ✅ Test predictions     - High bias → Increase model complexity

     - High variance → Add regularization/more data

### For Adding New Words

1. Create folder: `data/videos/<language>/<word>/`4. **Augmentation**:

2. Add 20-50 videos of that word   - Enabled by default

3. Retrain model   - Helps with generalization

4. System automatically learns new word!   - Simulates different lighting/angles



### For Adding New Languages## 🐛 Troubleshooting

1. Create folder: `data/videos/<new_language>/`

2. Add word folders inside### Installation Issues

3. Record videos (20-50 per word)

4. Retrain model**Problem**: `dlib` installation fails

5. Done! System now supports new language- **Solution**: Install Visual Studio Build Tools

  - Download from: https://visualstudio.microsoft.com/downloads/

---  - Select "Desktop development with C++"



## 💡 Tips for Best Results**Problem**: GPU not detected

- **Solution**: Install CUDA Toolkit 11.8 and cuDNN 8.6

### Recording Videos  - CUDA: https://developer.nvidia.com/cuda-11-8-0-download-archive

- ✅ Good, consistent lighting  - cuDNN: https://developer.nvidia.com/cudnn

- ✅ Clear face visibility

- ✅ Speak naturally with slight exaggeration### Training Issues

- ✅ Record variations (angles, expressions)

- ✅ 20-50 videos minimum per word**Problem**: Low training accuracy

- **Solution**: 

### Training  - Increase epochs

- ✅ Use GPU for faster training (10x speed)  - Check data quality

- ✅ Enable mixed precision (2-3x speed boost)  - Increase model complexity

- ✅ Monitor validation accuracy (most important!)

- ✅ Early stopping prevents overfitting**Problem**: Overfitting (high bias)

- **Solution**:

### Prediction  - Enable data augmentation

- ✅ Good lighting (same as training videos)  - Add more training data

- ✅ Face camera directly  - Increase dropout rate

- ✅ Clear lip movements

- ✅ Consistent distance from camera**Problem**: Out of memory during training

- **Solution**:

---  - Reduce batch size (try 4 or 2)

  - Use CPU training

## 🛠️ Technologies Used  - Close other applications



- **TensorFlow 2.6.0**: Deep learning framework### Prediction Issues

- **CUDA 11.3.1 + cuDNN 8.2.1**: GPU acceleration

- **OpenCV**: Video processing**Problem**: Poor real-time accuracy

- **MediaPipe**: Face and landmark detection- **Solution**:

- **Keras**: High-level neural network API  - Ensure good lighting

- **Albumentations**: Data augmentation  - Face camera directly

- **Python 3.9**: Programming language  - Speak clearly and slowly

  - Lower confidence threshold

---

**Problem**: Camera not opening

## 📊 Model Details- **Solution**:

  - Check camera ID (try 0, 1, 2)

### Architecture  - Ensure no other app is using camera

- **Input:** 75 frames × 200 features per frame  - Check camera permissions

- **3D Convolutional Layers:** (32, 64, 128 filters)

- **Bidirectional LSTM:** 128 units × 2 layers## 📈 Performance Expectations

- **Attention Mechanism:** Temporal attention

- **Dense Layers:** 256 → 128 → num_classes### Training Time (for 1000 samples, 50 epochs):

- **Activation:** Softmax for multi-class classification- **GPU (RTX 3060)**: ~2-3 hours

- **CPU (i7)**: ~20-30 hours

### Training

- **Optimizer:** Adam### Accuracy Expectations:

- **Loss:** Categorical Crossentropy- **50 samples per class**: 60-70% accuracy

- **Metrics:** Accuracy- **100 samples per class**: 75-85% accuracy

- **Regularization:** Dropout (0.3), L2 regularization- **200+ samples per class**: 85-95% accuracy

- **Early Stopping:** Patience = 10 epochs

### Real-Time Performance:

### Preprocessing Pipeline- **GPU**: 25-30 FPS

1. Frame extraction (25 FPS)- **CPU**: 5-10 FPS

2. Face detection (MediaPipe + dlib fallback)

3. Lip landmark detection (68 points)## 🔬 Technical Details

4. Grayscale conversion

5. ROI alignment and normalization### Technologies Used:

6. Data augmentation (noise, scaling, offsets)- **TensorFlow 2.13**: Deep learning framework

7. Feature extraction (200+ features: landmarks, geometry, temporal)- **OpenCV**: Video processing

8. Lip highlighting and tracking- **MediaPipe**: Face detection

- **Dlib**: Facial landmark detection

---- **Tkinter**: GUI framework



## 📈 Performance### Model Specifications:

- **Input**: 75 frames × 100×100 RGB

### Benchmark (with good training data)- **Output**: Probability distribution over classes

- **Preprocessing:** 1-5 minutes per video- **Parameters**: ~5-10 million (depends on classes)

- **Training:** 30-60 sec/epoch (GPU) or 3-5 min/epoch (CPU)- **Model Size**: ~50-100 MB

- **Prediction:** 0.3-0.5 seconds per video (real-time capable)

- **Accuracy:** 75-90% with 30-50 videos per word## 🌟 Future Enhancements



### Hardware Comparison- [ ] Add more Indian languages (Tamil, Telugu, Malayalam)

| GPU Model | VRAM | Training Speed | Batch Size |- [ ] Sentence-level lip reading (currently word-level)

|-----------|------|----------------|------------|- [ ] Mobile app deployment

| GTX 1660 Ti | 6GB | 30-45 sec/epoch | 8-16 |- [ ] Cloud-based training/inference API

| RTX 2060 | 6GB | 20-30 sec/epoch | 16-32 |- [ ] Pre-trained models for common words

| RTX 3060 | 12GB | 15-25 sec/epoch | 32-64 |- [ ] Voice synthesis from predictions

| CPU (i5) | - | 3-5 min/epoch | 4-8 |

## 📝 License

---

MIT License - See LICENSE file for details

## 🤝 Contributing

## 👨‍💻 Contributing

Want to improve the system? Ideas:

- Add more languagesContributions welcome! Areas for improvement:

- Improve preprocessing- More language support

- Optimize model architecture- Better model architectures

- Better data augmentation- Improved preprocessing

- Multi-word phrase recognition- Mobile deployment

- Documentation improvements

---

## 📚 References

## 📝 License

### Research Papers:

MIT License - See [LICENSE](LICENSE) file for details.- "Lip Reading Sentences in the Wild" - Chung et al., 2017

- "Deep Audio-Visual Speech Recognition" - Afouras et al., 2018

---

### Libraries:

## 🎯 Summary- TensorFlow: https://www.tensorflow.org

- MediaPipe: https://google.github.io/mediapipe

### What You Need- OpenCV: https://opencv.org

1. Windows PC with webcam- Dlib: http://dlib.net

2. NVIDIA GPU (optional but recommended)

3. Miniconda installed## 📞 Support

4. 20-50 training videos per word

For issues, questions, or contributions:

### What You Get1. Check DOCUMENTATION.md for detailed information

1. Trained lip reading model2. Review SETUP_GUIDE.md for installation help

2. Real-time prediction system3. Check Troubleshooting section above

3. Multi-language support

4. Easy-to-use GUI interface## 🎉 Getting Started Checklist



### Next Steps- [ ] Python 3.10.0 installed

1. **Read [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** for detailed setup- [ ] Virtual environment created

2. Install and verify environment- [ ] Dependencies installed

3. Record training videos- [ ] GPU setup verified (optional)

4. Train your model- [ ] Shape predictor downloaded

5. Start predicting! 🎉- [ ] Training videos organized

- [ ] Training GUI tested

---- [ ] Model trained

- [ ] Prediction GUI tested

**Need Help?** Check [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md) for:- [ ] Webcam working

- Detailed installation steps

- Troubleshooting solutions---

- Advanced configuration

- Best practices**Created**: October 2025  

**Version**: 1.0.0  

**Ready to start?** → Read the **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** now! 📖**Status**: Production Ready



---**Ready to revolutionize visual speech recognition! 🚀**


*Last Updated: October 14, 2025*
