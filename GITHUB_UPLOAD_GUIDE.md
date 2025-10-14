# GitHub Upload Guide

**Date:** October 14, 2025

---

## ✅ .gitignore Created Successfully!

Your `.gitignore` file is configured to exclude:
- ❌ Large training videos (can be several GB)
- ❌ Trained model files (.h5 files can be 100+ MB)
- ❌ Preprocessed data (can be regenerated)
- ❌ Log files and TensorBoard data
- ❌ Python cache and temporary files

**Repository will be ~500 KB instead of 2-5 GB!**

---

## 🚀 Quick Upload to GitHub (First Time)

### Step 1: Initialize Git Repository

```powershell
# Navigate to project directory (if not already there)
cd d:\P\multi-lingual-lip-reading

# Initialize git repository
git init

# Check git status
git status
```

### Step 2: Add Files to Git

```powershell
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status
```

### Step 3: Create First Commit

```powershell
# Create first commit
git commit -m "Initial commit: Multi-lingual lip reading system"
```

### Step 4: Create GitHub Repository

1. Go to: https://github.com/new
2. Repository name: `multi-lingual-lip-reading` (or your choice)
3. Description: "Deep learning lip reading system for Kannada, Hindi, and English"
4. Choose: **Public** or **Private**
5. **DO NOT** initialize with README (you already have one)
6. Click "Create repository"

### Step 5: Connect to GitHub

```powershell
# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/multi-lingual-lip-reading.git

# Verify remote
git remote -v
```

### Step 6: Push to GitHub

```powershell
# Push to GitHub (first time)
git branch -M main
git push -u origin main
```

**Enter your GitHub credentials when prompted.**

---

## 🔄 Updating GitHub (After Changes)

### After Making Changes to Code

```powershell
# Check what changed
git status

# Add all changes
git add .

# Create commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

### Common Commit Messages

```powershell
# After adding new feature
git commit -m "Add new preprocessing feature"

# After fixing bug
git commit -m "Fix camera connection issue"

# After updating documentation
git commit -m "Update README with setup instructions"

# After improving model
git commit -m "Improve model accuracy with attention mechanism"
```

---

## 📋 What Gets Uploaded

### ✅ Files That WILL Be Uploaded:

```
✅ All Python source code:
   - src/*.py
   - train_gui_with_recording.py
   - predict_gui.py
   - verify_gpu.py
   - initialize.py

✅ Documentation:
   - README.md
   - COMPLETE_GUIDE.md
   - CLEANUP_SUMMARY.md
   - LICENSE

✅ Configuration:
   - configs/config.yaml
   - models/class_mapping.json

✅ Directory structure:
   - data/videos/.gitkeep files
   - logs/.gitkeep files
   - outputs/.gitkeep files

✅ Git files:
   - .gitignore
```

### ❌ Files That WILL NOT Be Uploaded:

```
❌ Training videos:
   - data/videos/**/*.mp4, *.avi, etc.
   - Can be several GB!

❌ Trained models:
   - models/*.h5
   - Can be 100+ MB each

❌ Preprocessed data:
   - data/preprocessed/**
   - Can be regenerated

❌ Logs:
   - logs/training/**
   - logs/tensorboard/**

❌ Cache files:
   - src/__pycache__/
   - *.pyc

❌ IDE files:
   - .vscode/
   - .idea/

❌ Output files:
   - outputs/predictions/**
   - outputs/recordings/**
```

---

## 🔍 Verify Before Pushing

### Check Repository Size

```powershell
# Check what will be uploaded
git status

# Check repository size (should be < 1 MB)
git count-objects -vH
```

### Test .gitignore

```powershell
# List all tracked files
git ls-files

# Should NOT see:
# - .mp4, .avi files
# - .h5 model files
# - __pycache__ folders
```

---

## ⚠️ Important Notes

### 1. **Large Files Warning**

If you see this error:
```
remote: error: File data/videos/test.mp4 is 100 MB; this exceeds GitHub's file size limit of 100 MB
```

**Solution:**
```powershell
# Remove large file from git history
git rm --cached data/videos/test.mp4
git commit -m "Remove large video file"
git push
```

### 2. **Model Files**

**DO NOT upload .h5 model files to GitHub!**
- GitHub has 100 MB file limit
- Models can be 100+ MB
- Use alternative storage:
  - Google Drive
  - Dropbox
  - Git LFS (Large File Storage)
  - HuggingFace Model Hub

### 3. **Training Videos**

**DO NOT upload training videos!**
- Videos can be several GB
- Will make repository huge
- Users should record their own videos
- Provide sample instructions instead

---

## 📦 Sharing Trained Models (Optional)

### Option 1: Google Drive

1. Upload `models/best_model.h5` to Google Drive
2. Get shareable link
3. Add link to README.md:
   ```markdown
   ## Pre-trained Model
   Download trained model: [Google Drive Link](https://drive.google.com/...)
   ```

### Option 2: Git LFS (Large File Storage)

```powershell
# Install Git LFS
git lfs install

# Track .h5 files
git lfs track "*.h5"

# Add .gitattributes
git add .gitattributes

# Commit and push
git add models/best_model.h5
git commit -m "Add pre-trained model"
git push
```

**Note:** Git LFS has storage limits (1 GB free)

### Option 3: HuggingFace Hub

Upload to: https://huggingface.co/
- Free model hosting
- Versioning
- Easy sharing

---

## 🔐 GitHub Authentication

### Using HTTPS (Recommended)

```powershell
# Clone repository
git clone https://github.com/YOUR_USERNAME/multi-lingual-lip-reading.git

# GitHub will prompt for credentials
# Use Personal Access Token (not password)
```

### Create Personal Access Token:

1. GitHub → Settings → Developer settings
2. Personal access tokens → Tokens (classic)
3. Generate new token
4. Select scopes: `repo` (all)
5. Copy token (save it somewhere!)
6. Use token as password when pushing

---

## 📝 Repository Description

**Suggested GitHub repository description:**

```
Multi-Lingual Lip Reading System

A deep learning system for visual speech recognition (lip reading) 
supporting Kannada, Hindi, and English. Uses geometric feature extraction, 
Bidirectional LSTM, and attention mechanisms for real-time predictions.

Technologies: TensorFlow GPU, OpenCV, MediaPipe, Python 3.9
```

**Topics/Tags:**
```
lip-reading
deep-learning
tensorflow
computer-vision
speech-recognition
indian-languages
kannada
hindi
lstm
attention-mechanism
opencv
mediapipe
```

---

## 🎯 After Uploading

### Update README.md with GitHub Link

Add badges:
```markdown
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
```

### Add Installation from GitHub

```markdown
## Installation

### Clone Repository
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/multi-lingual-lip-reading.git
cd multi-lingual-lip-reading
\`\`\`

### Setup Environment
Follow instructions in [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)
```

---

## 🚀 Quick Commands Summary

```powershell
# First time setup
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main

# Regular updates
git add .
git commit -m "Your message"
git push

# Check status
git status
git log --oneline

# View remote
git remote -v
```

---

## ✅ Verification Checklist

Before pushing to GitHub:

- [ ] `.gitignore` file created
- [ ] No large video files in `git status`
- [ ] No large model files (.h5) in `git status`
- [ ] Repository size < 1 MB (check with `git count-objects -vH`)
- [ ] All `.py` files included
- [ ] All `.md` documentation included
- [ ] `LICENSE` file included
- [ ] `configs/config.yaml` included
- [ ] `.gitkeep` files in empty directories
- [ ] Tested: `git ls-files` shows only code/docs

---

## 🎉 You're Ready!

Your repository is configured and ready to upload to GitHub!

**Next Steps:**
1. Create GitHub repository
2. Run the commands above
3. Share your project with the world! 🌍

---

*Guide created: October 14, 2025*
