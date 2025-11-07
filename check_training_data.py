"""
Comprehensive check of training data quality
"""
import numpy as np
from pathlib import Path
import json

print("="*70)
print("TRAINING DATA QUALITY CHECK")
print("="*70)

# Check preprocessed data
preprocessed_dir = Path('./data/preprocessed/kannada')
if not preprocessed_dir.exists():
    print("❌ No preprocessed data found!")
    exit(1)

npy_files = list(preprocessed_dir.glob('*.npy'))
print(f"\n✓ Found {len(npy_files)} preprocessed files")

# Categorize by word
word_files = {}
for file in npy_files:
    word = file.name.split('_')[0]
    if word not in word_files:
        word_files[word] = []
    word_files[word].append(file)

print(f"\n📊 DATA DISTRIBUTION:")
for word, files in sorted(word_files.items()):
    print(f"   {word}: {len(files)} samples")

# Check data quality
print(f"\n🔍 DATA QUALITY CHECK:")
issues = []
all_shapes = []
all_means = []
all_stds = []
word_stats = {}

for word, files in word_files.items():
    word_shapes = []
    word_means = []
    word_stds = []
    
    print(f"\n   Checking {word}...")
    
    for i, file in enumerate(files[:10]):  # Check first 10 of each
        try:
            data = np.load(file)
            word_shapes.append(data.shape)
            all_shapes.append(data.shape)
            
            # Check for NaN or Inf
            if np.any(np.isnan(data)):
                issues.append(f"   ❌ {file.name}: Contains NaN values")
            if np.any(np.isinf(data)):
                issues.append(f"   ❌ {file.name}: Contains Inf values")
            
            # Statistics
            mean = np.mean(data)
            std = np.std(data)
            word_means.append(mean)
            word_stds.append(std)
            all_means.append(mean)
            all_stds.append(std)
            
            # Check if all zeros
            if np.all(data == 0):
                issues.append(f"   ❌ {file.name}: All zeros!")
            
            # Check variance
            if std < 0.01:
                issues.append(f"   ⚠️  {file.name}: Very low variance (std={std:.4f})")
                
        except Exception as e:
            issues.append(f"   ❌ {file.name}: Error loading - {e}")
    
    word_stats[word] = {
        'mean': np.mean(word_means),
        'std': np.mean(word_stds),
        'shape': word_shapes[0] if word_shapes else None
    }
    
    print(f"      Shape: {word_shapes[0] if word_shapes else 'ERROR'}")
    print(f"      Mean: {np.mean(word_means):.4f} (±{np.std(word_means):.4f})")
    print(f"      Std Dev: {np.mean(word_stds):.4f}")

# Check shape consistency
unique_shapes = set([str(s) for s in all_shapes])
if len(unique_shapes) > 1:
    print(f"\n   ⚠️  WARNING: Inconsistent shapes found!")
    for shape in unique_shapes:
        print(f"      {shape}")
else:
    print(f"\n   ✓ All samples have consistent shape: {all_shapes[0]}")

# Check if classes are distinguishable
print(f"\n📈 CLASS SEPARABILITY:")
words = list(word_stats.keys())
if len(words) == 2:
    diff_mean = abs(word_stats[words[0]]['mean'] - word_stats[words[1]]['mean'])
    diff_std = abs(word_stats[words[0]]['std'] - word_stats[words[1]]['std'])
    
    print(f"   Mean difference: {diff_mean:.4f}")
    print(f"   Std difference: {diff_std:.4f}")
    
    if diff_mean < 0.01 and diff_std < 0.01:
        print(f"   ⚠️  WARNING: Classes have very similar statistics!")
        print(f"   This might make them hard to distinguish.")
    else:
        print(f"   ✓ Classes show statistical differences")

# Report issues
if issues:
    print(f"\n⚠️  ISSUES FOUND ({len(issues)}):")
    for issue in issues[:20]:  # Show first 20
        print(issue)
    if len(issues) > 20:
        print(f"   ... and {len(issues)-20} more issues")
else:
    print(f"\n✓ No major issues found in sample data")

# Check original videos
print(f"\n📹 ORIGINAL VIDEO CHECK:")
video_dir = Path('./data/videos/kannada')
for word, files in word_files.items():
    video_folder = video_dir / word
    if video_folder.exists():
        videos = list(video_folder.glob('*.mp4'))
        print(f"   {word}: {len(videos)} videos")
        if len(videos) != len(files):
            print(f"      ⚠️  Mismatch: {len(videos)} videos but {len(files)} preprocessed files")
    else:
        print(f"   {word}: Video folder not found")

# Summary
print(f"\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70)

if not issues:
    print("✓ Training data looks good!")
    print("\nRecommendations for training:")
    print("  1. Use 80-100 epochs")
    print("  2. Batch size: 8 (current setting)")
    print("  3. Monitor validation accuracy - should reach 80%+")
    print("  4. Watch for both classes in predictions during training")
else:
    print("⚠️  Some issues detected in training data")
    print("\nBefore training:")
    print("  1. Review the issues above")
    print("  2. Consider re-preprocessing if major issues found")
    print("  3. Check if recordings were clear and well-lit")

print("\n💡 Training Tips:")
print("  - If accuracy stays at 50%, stop and retrain from scratch")
print("  - If loss doesn't decrease, try lowering learning rate")
print("  - If one class dominates, check class weights")
print("  - Validation accuracy should be 80-95% for good results")

print("\n" + "="*70)
