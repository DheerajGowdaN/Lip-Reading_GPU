"""
Test if the model is biased towards one class
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load model using the LipReadingModel class
print("Loading model...")
# Load mapping first to get num_classes
with open('models/class_mapping_kannada.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)

num_classes = len(mapping['idx_to_label'])

# Initialize and load model
from model import LipReadingModel
lip_model = LipReadingModel(
    num_classes=num_classes,
    sequence_length=75,
    frame_height=100,
    frame_width=100
)
lip_model.load_model('models/best_model_kannada.h5')
model = lip_model.model

# Load class mapping
with open('models/class_mapping_kannada.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)

print('\n=== MODEL INFO ===')
print(f'Input Shape: {model.input_shape}')
print(f'Output Shape: {model.output_shape}')
print('\nClass Mapping:')
for idx, label in mapping['idx_to_label'].items():
    print(f'  Class {idx}: {label}')

# Test with random inputs
print('\n=== TESTING WITH RANDOM INPUTS ===')
class_counts = {0: 0, 1: 0}

# Get correct feature size from model input shape
feature_size = model.input_shape[-1]
print(f'Using feature size: {feature_size}\n')

for i in range(50):
    random_input = np.random.randn(1, 75, feature_size)  # (batch, sequence, features)
    predictions = model.predict(random_input, verbose=0)
    
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx]
    
    class_counts[pred_idx] += 1
    
    if i < 10:  # Show first 10
        print(f'Test {i+1}: Class {pred_idx} - {confidence:.1%} | Probs: [{predictions[0][0]:.3f}, {predictions[0][1]:.3f}]')

print('\n=== RESULTS SUMMARY ===')
total = sum(class_counts.values())
print(f'Class 0 (ಕುಟುಂಬ): {class_counts[0]}/{total} = {class_counts[0]/total*100:.1f}%')
print(f'Class 1 (ಬಸವ): {class_counts[1]}/{total} = {class_counts[1]/total*100:.1f}%')

if class_counts[0] > 45 or class_counts[1] > 45:
    print('\n⚠️  WARNING: Model shows strong bias towards one class!')
    print('This indicates the model may not have learned properly.')
    print('\nRecommendations:')
    print('1. Retrain the model with more epochs')
    print('2. Check if training data is balanced')
    print('3. Verify training accuracy was good')
else:
    print('\n✓ Model distribution looks reasonable for random inputs')
