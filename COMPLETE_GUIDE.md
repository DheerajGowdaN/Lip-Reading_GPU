# Multi-Lingual Lip Reading System - Complete Setup & Operation Guide

**Version:** 2.0  
**Date:** October 14, 2025  
**Platform:** Windows 10/11  
**GPU Support:** NVIDIA GPUs with CUDA

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Software Installation](#software-installation)
3. [Environment Setup](#environment-setup)
4. [Project Installation](#project-installation)
5. [GPU Verification](#gpu-verification)
6. [Preparing Training Data](#preparing-training-data)
7. [Training the Model](#training-the-model)
8. [Making Predictions](#making-predictions)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Configuration](#advanced-configuration)

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS:** Windows 10/11 (64-bit)
- **CPU:** Intel i5 or AMD Ryzen 5 (4+ cores)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Storage:** 10 GB free space
- **Camera:** Webcam (720p or higher for predictions)

### Recommended for GPU Training
- **GPU:** NVIDIA GPU with CUDA support
  - GTX 1060 (6GB) or higher
  - RTX 2060 or higher (recommended)
  - Minimum 4GB VRAM
- **CUDA Compute Capability:** 3.5 or higher
- **Driver:** Latest NVIDIA GPU drivers

### Check Your GPU
```powershell
# Open PowerShell and run:
nvidia-smi
```

If you see GPU information, you're ready for GPU training!

---

## 📦 Software Installation

### Step 1: Install Miniconda (Python Package Manager)

#### Why Miniconda?
- Manages Python versions easily
- Handles complex dependencies (TensorFlow GPU, CUDA, cuDNN)
- Isolates environments to avoid conflicts

#### Download & Install
1. **Download Miniconda:**
   - Visit: https://docs.conda.io/en/latest/miniconda.html
   - Download: **Miniconda3 Windows 64-bit** (Python 3.9+)

2. **Run Installer:**
   - Double-click the downloaded `.exe` file
   - Select "Just Me" (recommended)
   - Installation path: `C:\Users\<YourName>\Miniconda3` (default is fine)
   - ✅ Check: "Add Anaconda to my PATH environment variable" (Important!)
   - ✅ Check: "Register Anaconda as my default Python"
   - Click "Install"

3. **Verify Installation:**
   ```powershell
   # Open NEW PowerShell window
   conda --version
   # Should show: conda 23.x.x or higher
   ```

#### Troubleshooting Conda Installation
If `conda` command not found:
1. Restart PowerShell completely
2. Or manually add to PATH:
   - Search "Environment Variables" in Windows
   - Edit "Path" variable
   - Add: `C:\Users\<YourName>\Miniconda3\Scripts`
   - Add: `C:\Users\<YourName>\Miniconda3`

---

## 🔧 Environment Setup

### Step 2: Create Conda Environment with TensorFlow GPU

#### Create Environment (One Command!)
```powershell
# Navigate to project directory
cd d:\P\multi-lingual-lip-reading

# Create environment with Python 3.9 and TensorFlow GPU
conda create -n lipread_gpu python=3.9 tensorflow-gpu=2.6.0 cudatoolkit=11.3.1 cudnn=8.2.1 -c conda-forge -y
```

**What this does:**
- Creates environment named `lipread_gpu`
- Installs Python 3.9.18
- Installs TensorFlow 2.6.0 with GPU support
- Installs CUDA Toolkit 11.3.1
- Installs cuDNN 8.2.1
- All dependencies compatible and working!

#### Activate Environment
```powershell
conda activate lipread_gpu
```

**You should see:** `(lipread_gpu)` at the start of your command prompt.

---

### Step 3: Install Additional Dependencies

```powershell
# Make sure lipread_gpu environment is active
conda activate lipread_gpu

# Install computer vision packages
conda install -c conda-forge opencv=4.8.1 -y

# Install other required packages via pip
pip install mediapipe==0.10.7
pip install albumentations==1.3.1
pip install matplotlib
pip install scikit-learn
pip install pandas
pip install keras==2.6.0

# Verify NumPy version (should be 1.23.5)
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

#### Complete Package List
After installation, you should have:
```
✅ Python: 3.9.18
✅ TensorFlow GPU: 2.6.0
✅ CUDA Toolkit: 11.3.1
✅ cuDNN: 8.2.1
✅ NumPy: 1.23.5
✅ OpenCV: 4.8.1.78
✅ MediaPipe: 0.10.7
✅ Keras: 2.6.0
✅ Albumentations: 1.3.1
✅ Matplotlib: latest
✅ Scikit-learn: latest
✅ Pandas: latest
```

#### Verify All Packages
```powershell
conda list
```

---

## 📂 Project Installation

### Step 4: Set Up Project Structure

```powershell
cd d:\P\multi-lingual-lip-reading

# Initialize project (creates necessary directories)
python initialize.py
```

**This creates:**
```
multi-lingual-lip-reading/
├── src/                      # Source code
│   ├── model.py             # Neural network model
│   ├── preprocessor.py      # Video preprocessing
│   ├── data_loader.py       # Data loading
│   └── utils.py             # Utility functions
├── configs/
│   └── config.yaml          # Configuration file
├── data/
│   ├── videos/              # Training videos
│   │   ├── english/
│   │   ├── hindi/
│   │   └── kannada/
│   └── preprocessed/        # Processed data
├── models/                   # Saved models
├── logs/
│   ├── training/            # Training logs
│   └── tensorboard/         # TensorBoard logs
└── outputs/
    ├── predictions/         # Prediction results
    └── recordings/          # Recorded videos
```

---

## ✅ GPU Verification

### Step 5: Verify GPU Setup

```powershell
# Run GPU verification script
python verify_gpu.py
```

**Expected Output (GPU Working):**
```
=========================================
GPU VERIFICATION REPORT
=========================================

✓ TensorFlow installed: 2.6.0
  Built with CUDA: True
  
✓ Found 1 GPU(s):
  [0] /physical_device:GPU:0
      Type: GPU
      Memory: 3.96 GB
      
✓ CUDA GPU Available: True
✓ GPU computation successful!
✓ Model compilation test successful!

=========================================
RECOMMENDATIONS:
=========================================

✅ GPU configuration looks good!
   Your system has GPU(s) and TensorFlow can use them.
   Make sure to select 'GPU' in the training GUI dropdown.
```

**If GPU Not Detected:**
1. Check NVIDIA drivers: `nvidia-smi`
2. Reinstall environment:
   ```powershell
   conda deactivate
   conda env remove -n lipread_gpu
   # Then repeat Step 2
   ```
3. See [Troubleshooting](#troubleshooting) section

---

## 🎥 Preparing Training Data

### Step 6: Record or Collect Videos

#### Video Requirements
- **Format:** MP4, AVI, or MOV
- **Duration:** 2-5 seconds per video
- **Resolution:** 640x480 or higher
- **Frame Rate:** 25-30 FPS
- **Content:** Clear view of face/lips
- **Quantity:** **20-50 videos per word minimum**
- **Lighting:** Good, consistent lighting
- **Background:** Any (face detection handles this)

#### Directory Structure
Place videos in appropriate folders:

```
data/videos/
├── english/
│   ├── hello/
│   │   ├── hello_001.mp4
│   │   ├── hello_002.mp4
│   │   ├── ... (20-50 videos)
│   ├── goodbye/
│   │   ├── goodbye_001.mp4
│   │   └── ...
│   └── thankyou/
├── hindi/
│   ├── नमस्ते/
│   │   ├── namaste_001.mp4
│   │   └── ... (20-50 videos)
│   └── धन्यवाद/
└── kannada/
    ├── ನಮಸ್ಕಾರ/
    │   ├── namaskara_001.mp4
    │   └── ... (20-50 videos)
    └── ಧನ್ಯವಾದ/
```

#### Recording Tips
1. **Camera Position:**
   - Face camera directly
   - Distance: 1-2 feet from camera
   - Keep face centered

2. **Lighting:**
   - Face should be well-lit
   - Avoid harsh shadows
   - Natural light or soft indoor lighting

3. **Recording:**
   - Speak clearly and naturally
   - Slightly exaggerate lip movements
   - Maintain consistent speed
   - Record same word multiple times with slight variations

4. **Variations (Important!):**
   - Different facial expressions
   - Slightly different speaking speeds
   - Different angles (±10-15 degrees)
   - This helps model generalize better

#### Using Phone as Webcam
If using phone camera via WiFi:
1. Install "DroidCam" or "IP Webcam" app
2. Connect phone to same WiFi as computer
3. Phone will appear as webcam in system
4. Record videos using phone camera

---

## 🎓 Training the Model

### Step 7: Start Training

#### Launch Training GUI
```powershell
# Make sure environment is active
conda activate lipread_gpu

# Start training GUI
python train_gui_with_recording.py
```

#### Training GUI Interface

**Main Controls:**
1. **Dataset Split:**
   - Train: 70% (recommended)
   - Validation: 15%
   - Test: 15%

2. **Training Parameters:**
   - Batch Size: 8 (default)
     - Lower if GPU memory error (try 4)
     - Higher for faster training (16 or 32)
   - Epochs: 50-100
   - Learning Rate: 0.0001 (default, works well)

3. **Device Selection:**
   - GPU: Select this if GPU detected
   - CPU: Fallback option (slower)

4. **Mixed Precision:**
   - ✅ Enable for faster training (GPU only)
   - Reduces memory usage
   - Speeds up training by 2-3x

**Training Process:**

1. **Click "Start Training"**
   - System will preprocess videos
   - Creates features from lip movements
   - This takes 1-5 minutes per video

2. **Monitor Progress:**
   ```
   Epoch 1/50
   Steps: 3/3 [======] - ETA: 0s
   Loss: 1.234 | Accuracy: 35.67%
   Val Loss: 1.456 | Val Accuracy: 28.34%
   ```

3. **Watch Metrics:**
   - **Training Loss:** Should decrease steadily
   - **Training Accuracy:** Should increase
   - **Validation Accuracy:** Important! Should increase
   - **Bias/Variance:** Monitor overfitting

4. **Early Stopping:**
   - Training stops if validation accuracy doesn't improve
   - Patience: 10 epochs (default)
   - Best model is automatically saved

#### Training Time Estimates

| Dataset Size | Device | Time per Epoch | Total (50 epochs) |
|-------------|--------|----------------|------------------|
| 100 videos  | GPU    | 30-60 sec      | 25-50 min        |
| 100 videos  | CPU    | 3-5 min        | 2.5-4 hours      |
| 500 videos  | GPU    | 2-3 min        | 1.5-2.5 hours    |
| 500 videos  | CPU    | 10-15 min      | 8-12 hours       |

#### What Happens During Training

1. **Preprocessing Pipeline (Automatic):**
   - Frame extraction from videos
   - Face detection (MediaPipe)
   - Lip landmark detection (68 points)
   - Grayscale conversion
   - ROI alignment and normalization
   - Data augmentation (noise, scaling, offsets)
   - Feature extraction (200+ features per frame)
   - Lip highlighting and tracking

2. **Model Training:**
   - 3D CNN extracts spatial-temporal features
   - Bidirectional LSTM models sequences
   - Attention mechanism focuses on important frames
   - Dense layers perform final classification

3. **Checkpointing:**
   - Best model saved automatically
   - Training logs saved to `logs/training/`
   - TensorBoard logs for visualization

#### Viewing Training Progress with TensorBoard

```powershell
# In another PowerShell window
conda activate lipread_gpu
tensorboard --logdir=logs/tensorboard

# Open browser to: http://localhost:6006
```

**TensorBoard shows:**
- Loss curves (training vs validation)
- Accuracy curves
- Learning rate schedule
- Model architecture
- Real-time updates during training

---

## 🔮 Making Predictions

### Step 8: Real-Time Lip Reading

#### Launch Prediction GUI
```powershell
# Make sure environment is active
conda activate lipread_gpu

# Start prediction GUI
python predict_gui.py
```

#### Prediction GUI Interface

**Main Components:**

1. **Model Information:**
   - Model Status: Loaded ✓
   - Classes: Number of words model knows
   - Accuracy: Model's validation accuracy
   - Device: GPU/CPU being used

2. **Camera Settings:**
   - Camera Index: 0 (default), 1 (external/phone)
   - Test camera before recording
   - Click "Test Camera" to verify

3. **Recording Controls:**
   - Duration: 3-5 seconds recommended
   - Click "Start Recording" to begin
   - Countdown shows before recording
   - Recording indicator displays during capture

4. **Prediction Display:**
   - Predicted Word: Shows detected word
   - Confidence: Prediction certainty (%)
   - Top 3 Predictions: Alternative possibilities
   - Processing Time: Speed of prediction

**Using the System:**

1. **Position Yourself:**
   - Face camera directly
   - Distance: 1-2 feet
   - Ensure good lighting
   - Face should be fully visible

2. **Start Recording:**
   - Click "Start Recording"
   - Wait for countdown (3, 2, 1...)
   - Speak the word clearly
   - Recording stops automatically

3. **View Results:**
   - Prediction appears instantly
   - Check confidence level
   - View alternative predictions
   - Review processing time

4. **Tips for Better Predictions:**
   - Speak clearly with natural pace
   - Slightly exaggerate lip movements
   - Keep head stable (minor movement OK)
   - Maintain consistent lighting
   - Look at camera, not screen

#### Understanding Predictions

**Confidence Levels:**
- **90-100%:** Very confident (excellent)
- **70-89%:** Good confidence (reliable)
- **50-69%:** Moderate confidence (uncertain)
- **Below 50%:** Low confidence (unreliable)

**If Predictions Are Wrong:**
1. Check if word was in training data
2. Verify training videos are good quality
3. Need 20-50 videos per word minimum
4. Record more training videos
5. Retrain model with expanded dataset

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. GPU Not Detected

**Symptoms:**
- verify_gpu.py shows "No GPU devices found"
- Training uses CPU (very slow)

**Solutions:**

**Check NVIDIA Driver:**
```powershell
nvidia-smi
```
If error, install latest driver from: https://www.nvidia.com/Download/index.aspx

**Recreate Environment:**
```powershell
conda deactivate
conda env remove -n lipread_gpu
conda create -n lipread_gpu python=3.9 tensorflow-gpu=2.6.0 cudatoolkit=11.3.1 cudnn=8.2.1 -c conda-forge -y
conda activate lipread_gpu
```

**Verify TensorFlow Sees GPU:**
```powershell
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

---

#### 2. Out of Memory Error (GPU)

**Symptoms:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**

**Reduce Batch Size:**
- Change from 8 to 4 or 2
- In training GUI, modify batch size

**Enable Memory Growth:**
- Already enabled in code
- Automatically manages GPU memory

**Reduce Sequence Length:**
- Edit `configs/config.yaml`
- Change `sequence_length: 75` to `50`

**Close Other GPU Programs:**
- Chrome hardware acceleration
- Other deep learning applications
- Games

---

#### 3. Camera Not Working

**Symptoms:**
- Black screen in prediction GUI
- "Cannot open camera" error

**Solutions:**

**Try Different Camera Index:**
- 0: Built-in webcam
- 1: External/USB webcam
- 2+: Phone camera (DroidCam/IP Webcam)

**Check Camera Permissions:**
- Windows Settings → Privacy → Camera
- Allow apps to access camera

**Test Camera Externally:**
- Open Windows Camera app
- Verify camera works

**Phone as Webcam:**
- Install "DroidCam" (free)
- Connect phone to same WiFi
- Launch DroidCam on phone
- Install DroidCam Client on PC
- Phone appears as Camera 1 or 2

---

#### 4. Poor Prediction Accuracy

**Symptoms:**
- Wrong predictions
- Low confidence scores
- Model guesses randomly

**Root Causes & Solutions:**

**Insufficient Training Data:**
- ❌ Problem: Only 3-10 videos per word
- ✅ Solution: Record 20-50 videos per word
- ✅ More data = better accuracy

**Poor Video Quality:**
- ❌ Problem: Dark, blurry, or obstructed face
- ✅ Solution: Good lighting, clear videos
- ✅ Record in consistent conditions

**Overfitting:**
- ❌ Problem: Train accuracy 95%, Val accuracy 30%
- ✅ Solution: More training data
- ✅ Enable data augmentation (already enabled)

**Need Retraining:**
- After adding more videos
- Retrain model completely
- Don't use old model with new data structure

---

#### 5. Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'mediapipe'
ImportError: cannot import name 'xyz'
```

**Solutions:**

**Verify Environment Active:**
```powershell
conda activate lipread_gpu
# Should see (lipread_gpu) at prompt
```

**Reinstall Package:**
```powershell
pip install --force-reinstall mediapipe==0.10.7
```

**Check All Packages:**
```powershell
conda list
# Verify all required packages present
```

---

#### 6. Training Stuck or Frozen

**Symptoms:**
- Training doesn't progress
- GPU usage 0%
- Stuck on "Preprocessing..."

**Solutions:**

**Check Data:**
```powershell
# Verify videos exist
Get-ChildItem -Path "data\videos" -Recurse -Filter "*.mp4"
```

**Restart Training:**
- Close GUI
- Reactivate environment
- Clear cache: Delete `data/preprocessed/`
- Restart training

**Check Logs:**
```powershell
# View error details
Get-Content "logs\training\latest.log"
```

---

#### 7. Preprocessing Errors

**Symptoms:**
```
Error: Could not detect face in video
Error: MediaPipe initialization failed
```

**Solutions:**

**Video Quality:**
- Ensure face is visible
- Good lighting
- Face centered in frame
- Not too far from camera

**MediaPipe Issues:**
```powershell
pip install --force-reinstall mediapipe==0.10.7
```

**Skip Bad Videos:**
- System automatically skips videos without faces
- Check preprocessing logs for details
- Remove or re-record problem videos

---

## ⚙️ Advanced Configuration

### Configuration File

Edit `configs/config.yaml` to customize:

```yaml
# Model Architecture
model:
  sequence_length: 75        # Number of frames per video
  num_features: 200          # Features per frame
  
# Training Parameters
training:
  batch_size: 8              # Samples per batch
  learning_rate: 0.0001      # Learning rate
  epochs: 100                # Max epochs
  early_stopping_patience: 10
  
# Data Augmentation
augmentation:
  noise_probability: 0.5     # 50% chance
  scale_probability: 0.3     # 30% chance
  offset_probability: 0.3    # 30% chance
  
# Preprocessing
preprocessing:
  target_fps: 25             # Target frame rate
  grayscale: true            # Convert to grayscale
  
# Paths
paths:
  videos: "data/videos"
  preprocessed: "data/preprocessed"
  models: "models"
  logs: "logs"
```

### Environment Variables

Set custom paths:

```powershell
# In PowerShell
$env:CUDA_VISIBLE_DEVICES = "0"  # Use first GPU
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"
```

### Performance Tuning

**Faster Training:**
1. Increase batch size (if GPU memory allows)
2. Enable mixed precision (FP16)
3. Reduce sequence length
4. Use GPU (2-10x faster than CPU)

**Better Accuracy:**
1. More training data (most important!)
2. Longer training (more epochs)
3. Lower learning rate for fine-tuning
4. Data augmentation (already enabled)

**Less Memory Usage:**
1. Reduce batch size
2. Reduce sequence length
3. Reduce num_features
4. Use mixed precision

---

## 📊 Project Structure

```
multi-lingual-lip-reading/
├── src/                              # Source Code
│   ├── model.py                      # Neural network model
│   ├── preprocessor.py               # Video preprocessing
│   ├── data_loader.py                # Data loading & batching
│   └── utils.py                      # Helper functions
│
├── configs/
│   └── config.yaml                   # Configuration file
│
├── data/
│   ├── videos/                       # Training videos
│   │   ├── english/
│   │   ├── hindi/
│   │   └── kannada/
│   └── preprocessed/                 # Processed features
│
├── models/
│   ├── best_model.h5                 # Trained model
│   └── class_mapping.json            # Label mappings
│
├── logs/
│   ├── training/                     # Training logs
│   └── tensorboard/                  # TensorBoard logs
│
├── outputs/
│   ├── predictions/                  # Prediction results
│   └── recordings/                   # Recorded videos
│
├── train_gui_with_recording.py       # Training interface
├── predict_gui.py                    # Prediction interface
├── verify_gpu.py                     # GPU verification
├── initialize.py                     # Project setup
├── LICENSE                           # License file
└── COMPLETE_GUIDE.md                 # This file!
```

---

## 🎯 Quick Reference Commands

### Environment Management
```powershell
# Activate environment
conda activate lipread_gpu

# Deactivate environment
conda deactivate

# List all environments
conda env list

# List installed packages
conda list

# Remove environment
conda env remove -n lipread_gpu
```

### Training & Prediction
```powershell
# Verify GPU
python verify_gpu.py

# Initialize project
python initialize.py

# Train model
python train_gui_with_recording.py

# Make predictions
python predict_gui.py

# View TensorBoard
tensorboard --logdir=logs/tensorboard
```

### Data Management
```powershell
# Count training videos
Get-ChildItem -Path "data\videos" -Recurse -Filter "*.mp4" | Measure-Object

# Clear preprocessed data
Remove-Item -Path "data\preprocessed\*" -Recurse -Force

# Check model exists
Test-Path "models\best_model.h5"
```

---

## 📚 Additional Resources

### Documentation
- **TensorFlow GPU:** https://www.tensorflow.org/install/gpu
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-toolkit
- **MediaPipe:** https://google.github.io/mediapipe/
- **OpenCV:** https://opencv.org/

### Lip Reading Research
- LipNet: https://arxiv.org/abs/1611.01599
- Deep Learning for Lip Reading: Various papers on arXiv

### Getting Help
1. Check this guide first
2. Review error messages in logs
3. Run `verify_gpu.py` for diagnostics
4. Check GitHub issues (if using Git)

---

## 🚀 Next Steps

### For New Users
1. ✅ Complete Steps 1-5 (Environment Setup)
2. ✅ Verify GPU working
3. ✅ Record 20-50 videos per word
4. ✅ Train model
5. ✅ Test predictions

### For Improving Accuracy
1. 📹 Record more training videos (50-100 per word)
2. 🎬 Ensure video quality is good
3. 🔄 Retrain with expanded dataset
4. 🎯 Test and iterate

### For Adding New Languages
1. Create folder: `data/videos/<language>/`
2. Add word folders inside
3. Record videos (20-50 per word)
4. Retrain model
5. System automatically supports new language!

---

## ✅ Summary Checklist

Before starting, ensure you have:

- [ ] Windows 10/11 (64-bit)
- [ ] NVIDIA GPU (optional, but recommended)
- [ ] Miniconda installed
- [ ] `lipread_gpu` conda environment created
- [ ] All dependencies installed
- [ ] GPU verified (if using GPU)
- [ ] Project initialized
- [ ] 20-50 training videos per word
- [ ] Videos organized in correct folders
- [ ] Good webcam for predictions

You're ready to train and use the lip reading system! 🎉

---

## 📞 Support

**Quick Diagnostics:**
```powershell
# Run this to check everything
python verify_gpu.py
conda list
python -c "import tensorflow as tf, cv2, mediapipe; print('All imports OK!')"
```

**Common Issues:**
- GPU not detected → Check NVIDIA drivers, recreate environment
- Out of memory → Reduce batch size
- Poor accuracy → Need more training data (20-50 videos per word)
- Camera not working → Try different camera index (0, 1, 2)

---

**Last Updated:** October 14, 2025  
**Version:** 2.0  
**Author:** Multi-Lingual Lip Reading Team

**Good luck with your lip reading system! 🎯👄🤖**
