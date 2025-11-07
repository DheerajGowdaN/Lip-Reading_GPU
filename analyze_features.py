"""
Deep dive into feature differences between classes
"""
import numpy as np
from pathlib import Path

print("="*70)
print("DETAILED FEATURE ANALYSIS")
print("="*70)

preprocessed_dir = Path('./data/preprocessed/kannada')

# Load all samples
kutumba_files = sorted(preprocessed_dir.glob('ಕುಟುಂಬ*.npy'))
basava_files = sorted(preprocessed_dir.glob('ಬಸವ*.npy'))

print(f"\nLoading {len(kutumba_files)} ಕುಟುಂಬ samples...")
kutumba_data = [np.load(f) for f in kutumba_files[:10]]  # Load first 10

print(f"Loading {len(basava_files)} ಬಸವ samples...")
basava_data = [np.load(f) for f in basava_files[:10]]  # Load first 10

# Check actual feature values
print(f"\n📊 SAMPLE FEATURE VALUES:")
print(f"\nಕುಟುಂಬ Sample 1:")
print(f"   Shape: {kutumba_data[0].shape}")
print(f"   First 5 features of first frame: {kutumba_data[0][0, :5]}")
print(f"   Min: {kutumba_data[0].min():.4f}, Max: {kutumba_data[0].max():.4f}")
print(f"   Mean: {kutumba_data[0].mean():.4f}, Std: {kutumba_data[0].std():.4f}")

print(f"\nಬಸವ Sample 1:")
print(f"   Shape: {basava_data[0].shape}")
print(f"   First 5 features of first frame: {basava_data[0][0, :5]}")
print(f"   Min: {basava_data[0].min():.4f}, Max: {basava_data[0].max():.4f}")
print(f"   Mean: {basava_data[0].mean():.4f}, Std: {basava_data[0].std():.4f}")

# Compare sequences
print(f"\n🔄 TEMPORAL PATTERN ANALYSIS:")
kutumba_avg = np.mean([d for d in kutumba_data], axis=0)  # Average across all samples
basava_avg = np.mean([d for d in basava_data], axis=0)

print(f"   ಕುಟುಂಬ average sequence shape: {kutumba_avg.shape}")
print(f"   ಬಸವ average sequence shape: {basava_avg.shape}")

# Calculate difference
diff = np.abs(kutumba_avg - basava_avg)
print(f"\n   Absolute difference between classes:")
print(f"      Mean: {diff.mean():.6f}")
print(f"      Max: {diff.max():.6f}")
print(f"      Min: {diff.min():.6f}")

# Find most discriminative features
feature_variance = np.var(diff, axis=0)
top_features = np.argsort(feature_variance)[-10:][::-1]
print(f"\n   Top 10 most discriminative features (by variance):")
for i, feat_idx in enumerate(top_features, 1):
    print(f"      {i}. Feature {feat_idx}: variance={feature_variance[feat_idx]:.6f}")

# Check if data is normalized
print(f"\n🔍 NORMALIZATION CHECK:")
all_data = kutumba_data + basava_data
all_means = [d.mean() for d in all_data]
all_stds = [d.std() for d in all_data]

print(f"   Mean range: [{min(all_means):.4f}, {max(all_means):.4f}]")
print(f"   Std range: [{min(all_stds):.4f}, {max(all_stds):.4f}]")

if abs(np.mean(all_means)) < 0.01 and abs(np.mean(all_stds) - 1.0) < 0.01:
    print(f"   ✓ Data appears to be normalized (mean≈0, std≈1)")
else:
    print(f"   ⚠️  Data normalization may be off")

# Check for actual movement
print(f"\n📹 MOVEMENT DETECTION:")
for i, (k_sample, b_sample) in enumerate(zip(kutumba_data[:3], basava_data[:3]), 1):
    # Calculate frame-to-frame differences
    k_movement = np.mean([np.linalg.norm(k_sample[j] - k_sample[j-1]) 
                          for j in range(1, len(k_sample))])
    b_movement = np.mean([np.linalg.norm(b_sample[j] - b_sample[j-1]) 
                          for j in range(1, len(b_sample))])
    
    print(f"\n   Sample {i}:")
    print(f"      ಕುಟುಂಬ movement: {k_movement:.4f}")
    print(f"      ಬಸವ movement: {b_movement:.4f}")

# Final assessment
print(f"\n" + "="*70)
print("ASSESSMENT")
print("="*70)

if diff.mean() < 0.1:
    print("⚠️  WARNING: Classes are very similar in feature space!")
    print("\nPossible reasons:")
    print("  1. Lip movements for these words are naturally similar")
    print("  2. Videos may have been recorded in similar conditions")
    print("  3. Feature extraction might not be capturing differences")
    print("\nSolutions:")
    print("  1. Record more varied samples (different angles, speeds)")
    print("  2. Add more exaggerated pronunciation")
    print("  3. Use longer training (100+ epochs)")
    print("  4. Consider adding more distinct words first")
else:
    print("✓ Classes show measurable differences in feature space")
    print("   The model should be able to learn these patterns")

print("="*70)
