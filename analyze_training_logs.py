"""
Analyze the most recent training logs to see what went wrong
"""
from pathlib import Path
import struct

print("="*70)
print("TRAINING LOG ANALYSIS")
print("="*70)

# Find most recent tensorboard log
log_dir = Path('./logs/tensorboard/train')
event_files = list(log_dir.glob('events.out.tfevents.*'))

if not event_files:
    print("❌ No training logs found!")
    exit(1)

# Sort by modification time
event_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
most_recent = event_files[0]

print(f"\n📁 Most recent log: {most_recent.name}")
print(f"   Size: {most_recent.stat().st_size / 1024:.1f} KB")
print(f"   Modified: {Path(most_recent).stat().st_mtime}")

# Try to read with tensorboard
try:
    from tensorboard.backend.event_processing import event_accumulator
    
    print("\n📊 Loading training history...")
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    # Get available tags
    print(f"\n   Available metrics: {ea.Tags()['scalars']}")
    
    # Get training metrics
    if 'epoch_loss' in ea.Tags()['scalars']:
        losses = ea.Scalars('epoch_loss')
        print(f"\n📉 LOSS HISTORY:")
        for i, event in enumerate(losses[-10:]):  # Last 10 epochs
            print(f"   Epoch {i}: {event.value:.4f}")
    
    if 'epoch_accuracy' in ea.Tags()['scalars']:
        accuracies = ea.Scalars('epoch_accuracy')
        print(f"\n📈 ACCURACY HISTORY:")
        for i, event in enumerate(accuracies[-10:]):  # Last 10 epochs
            print(f"   Epoch {i}: {event.value:.4f} ({event.value*100:.1f}%)")
        
        # Check if stuck
        recent_acc = [e.value for e in accuracies[-10:]]
        if len(recent_acc) > 5:
            if max(recent_acc) - min(recent_acc) < 0.01:
                print(f"\n   ⚠️  WARNING: Accuracy was stuck at {recent_acc[-1]*100:.1f}%")
            if abs(recent_acc[-1] - 0.5) < 0.01:
                print(f"   ⚠️  WARNING: Accuracy stuck at ~50% (random guessing)")
            if recent_acc[-1] < 0.6:
                print(f"   ❌ PROBLEM: Final accuracy too low ({recent_acc[-1]*100:.1f}%)")
    
    if 'epoch_val_accuracy' in ea.Tags()['scalars']:
        val_accuracies = ea.Scalars('epoch_val_accuracy')
        print(f"\n📊 VALIDATION ACCURACY HISTORY:")
        for i, event in enumerate(val_accuracies[-10:]):  # Last 10 epochs
            print(f"   Epoch {i}: {event.value:.4f} ({event.value*100:.1f}%)")

except ImportError:
    print("\n⚠️  tensorboard not available, trying manual analysis...")
    print("   Install with: pip install tensorboard")
except Exception as e:
    print(f"\n⚠️  Error reading tensorboard logs: {e}")

# Check for checkpoint files
checkpoint_dir = Path('./models')
h5_files = list(checkpoint_dir.glob('*.h5'))
if h5_files:
    print(f"\n💾 MODEL CHECKPOINTS:")
    for f in h5_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"   {f.name}: {size_mb:.1f} MB")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

print("""
If the model completed all 50 epochs but still predicts only one class:

❌ PROBLEM: Model converged to a degenerate solution
   - Loss decreased but model learned to always predict one class
   - This is called "mode collapse" or "class imbalance collapse"

🔍 POSSIBLE CAUSES:
   1. Class imbalance in batches (even with equal data)
   2. Poor weight initialization (unlucky start)
   3. Learning rate too high (jumped over optimal solution)
   4. Loss function not properly penalizing wrong predictions

✅ SOLUTIONS:
   1. Add class weights to balance learning:
      class_weight = {0: 1.0, 1: 1.0}
      
   2. Use different loss function:
      - Try focal loss instead of categorical crossentropy
      
   3. Lower learning rate:
      - Try 0.0005 instead of 0.001
      
   4. Add more regularization:
      - Increase dropout from 0.5 to 0.6
      
   5. Different optimizer:
      - Try Adam with different beta values

🚀 IMMEDIATE ACTION:
   Retrain with modified settings - I'll update the config for you!
""")

print("="*70)
