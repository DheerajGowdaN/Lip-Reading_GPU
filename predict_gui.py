"""
Real-Time Prediction GUI for Multi-Lingual Lip Reading System
Author: AI Assistant
Date: October 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import queue
import sys
import os
from collections import deque
import tensorflow as tf
from tensorflow import keras
from scipy.signal import savgol_filter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import LipReadingModel
from src.preprocessor import VideoPreprocessor
from src.utils import *


def put_unicode_text(img, text, position, font_size=24, color=(0, 255, 255)):
    """
    Put Unicode text on OpenCV image using PIL
    
    Args:
        img: OpenCV image (BGR)
        text: Text to display (supports Unicode)
        position: (x, y) position
        font_size: Font size
        color: Text color in BGR format
    
    Returns:
        img: Image with text
    """
    try:
        # Convert OpenCV BGR to PIL RGB
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Try to use a font that supports Unicode (Kannada)
        try:
            # Windows fonts that support Kannada
            font_paths = [
                "C:/Windows/Fonts/segoeui.ttf",  # Segoe UI
                "C:/Windows/Fonts/arial.ttf",     # Arial
                "C:/Windows/Fonts/NotoSans-Regular.ttf",  # Noto Sans
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Convert BGR to RGB for PIL
        rgb_color = (color[2], color[1], color[0])
        
        # Draw text
        draw.text(position, text, font=font, fill=rgb_color)
        
        # Convert back to OpenCV BGR
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        return img
    except Exception as e:
        print(f"Error rendering Unicode text: {e}")
        # Fallback to cv2.putText (will show as ????)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img


class PredictionGUI:
    """GUI for real-time lip reading prediction"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Lingual Lip Reading - Real-Time Prediction")
        self.root.geometry("1000x700")
        
        # Initialize variables
        self.config = load_config()
        self.model = None
        self.preprocessor = None
        self.class_mapping = None
        self.idx_to_label = {}
        
        # Video capture
        self.cap = None
        self.is_capturing = False
        self.capture_thread = None
        
        # Frame buffer for sequences
        self.frame_buffer = deque(maxlen=75)  # 75 frames for prediction
        self.lip_buffer = deque(maxlen=75)
        
        # Prediction
        self.current_prediction = "No prediction"
        self.current_prediction_display = "No prediction"  # ASCII-safe version for overlay
        self.prediction_confidence = 0.0
        self.top_predictions = []
        
        # Prediction stabilization
        self.prediction_history = deque(maxlen=5)  # Last 5 predictions
        self.stable_prediction = "No prediction"
        self.stable_prediction_display = "No prediction"
        self.stable_confidence = 0.0
        self.prediction_change_threshold = 0.65  # Minimum confidence to change prediction
        self.stability_count = 0  # Consecutive same predictions
        self.stability_required = 2  # Need 2 consecutive predictions to change
        
        # Lip tracking improvements
        self.previous_lip_landmarks = None
        self.landmark_smoothing_alpha = 0.6  # Stronger smoothing for landmarks
        
        # Frame queue for thread-safe GUI updates
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Setup GUI
        self.setup_ui()
        
        # Initialize system
        self.initialize_system()
        
        # Start GUI update loop
        self.update_gui()
    
    def setup_ui(self):
        """Setup the GUI layout"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Multi-Lingual Lip Reading - Real-Time Prediction",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        # Left panel - Video feed
        self.setup_video_panel(main_frame)
        
        # Right panel - Controls and results
        self.setup_control_panel(main_frame)
    
    def setup_video_panel(self, parent):
        """Setup video display panel"""
        frame = ttk.LabelFrame(parent, text="Video Feed", padding="10")
        frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        
        # Video display
        self.video_label = ttk.Label(frame, text="Camera not started", 
                                     background="black", foreground="white",
                                     font=("Arial", 14))
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # FPS and status
        status_frame = ttk.Frame(frame)
        status_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0", font=("Arial", 10))
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready", 
                                      font=("Arial", 10), foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=10)
    
    def setup_control_panel(self, parent):
        """Setup control and results panel"""
        frame = ttk.Frame(parent)
        frame.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame.rowconfigure(2, weight=1)
        
        # Model selection
        model_frame = ttk.LabelFrame(frame, text="Model Configuration", padding="10")
        model_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_path_var = tk.StringVar(value="./models/best_model.h5")
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=25)
        model_entry.grid(row=0, column=1, padx=5, pady=5)
        
        browse_btn = ttk.Button(model_frame, text="Browse", command=self.browse_model)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        load_btn = ttk.Button(model_frame, text="Load Model", command=self.load_model)
        load_btn.grid(row=1, column=0, columnspan=3, pady=10)
        
        self.model_info_var = tk.StringVar(value="No model loaded")
        info_label = ttk.Label(model_frame, textvariable=self.model_info_var, 
                              foreground="blue", wraplength=250)
        info_label.grid(row=2, column=0, columnspan=3, pady=5)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(frame, text="Camera Controls", padding="10")
        camera_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.start_camera_btn = ttk.Button(camera_frame, text="Start Camera", 
                                          command=self.start_camera, width=20)
        self.start_camera_btn.pack(pady=5)
        
        self.stop_camera_btn = ttk.Button(camera_frame, text="Stop Camera", 
                                         command=self.stop_camera, width=20,
                                         state=tk.DISABLED)
        self.stop_camera_btn.pack(pady=5)
        
        # Settings
        ttk.Label(camera_frame, text="Camera ID:").pack(pady=5)
        self.camera_id_var = tk.IntVar(value=0)
        ttk.Spinbox(camera_frame, from_=0, to=5, textvariable=self.camera_id_var, 
                   width=10).pack(pady=5)
        
        ttk.Label(camera_frame, text="Confidence Threshold:").pack(pady=5)
        self.confidence_threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = ttk.Scale(camera_frame, from_=0.0, to=1.0, 
                                   variable=self.confidence_threshold_var,
                                   orient=tk.HORIZONTAL, length=200)
        threshold_scale.pack(pady=5)
        
        self.threshold_label = ttk.Label(camera_frame, text="0.50")
        self.threshold_label.pack(pady=5)
        
        # Update threshold label
        def update_threshold_label(val):
            self.threshold_label.config(text=f"{float(val):.2f}")
        
        threshold_scale.config(command=update_threshold_label)
        
        # Prediction results
        results_frame = ttk.LabelFrame(frame, text="Prediction Results", padding="10")
        results_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Current prediction (large)
        self.prediction_label = ttk.Label(results_frame, 
                                         text="Waiting...",
                                         font=("Arial", 20, "bold"),
                                         foreground="green",
                                         wraplength=250)
        self.prediction_label.pack(pady=10)
        
        # Confidence
        self.confidence_label = ttk.Label(results_frame, 
                                         text="Confidence: 0%",
                                         font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        
        # Top 3 predictions
        ttk.Label(results_frame, text="Top 3 Predictions:", 
                 font=("Arial", 10, "bold")).pack(pady=10)
        
        self.top_predictions_text = tk.Text(results_frame, height=6, width=30,
                                           font=("Arial", 9))
        self.top_predictions_text.pack(pady=5)
        self.top_predictions_text.config(state=tk.DISABLED)
        
        # Recording controls
        recording_frame = ttk.LabelFrame(frame, text="Recording", padding="10")
        recording_frame.grid(row=3, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.record_btn = ttk.Button(recording_frame, text="Record Video", 
                                    command=self.toggle_recording, width=20)
        self.record_btn.pack(pady=5)
        
        self.is_recording = False
        self.video_writer = None
    
    def initialize_system(self):
        """Initialize the system"""
        print("Initializing prediction system...")
        
        # Setup GPU
        setup_gpu('GPU')
        
        # Ensure shape predictor
        ensure_shape_predictor()
        
        # Initialize preprocessor
        self.preprocessor = VideoPreprocessor(
            shape_predictor_path=self.config['paths']['shape_predictor'],
            sequence_length=self.config['model']['sequence_length'],
            target_size=(self.config['model']['frame_height'], 
                        self.config['model']['frame_width']),
            augment=False
        )
        
        print("✓ Prediction system initialized\n")
    
    def browse_model(self):
        """Browse for model file"""
        filepath = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
            initialdir="./models"
        )
        if filepath:
            self.model_path_var.set(filepath)
    
    def load_model(self):
        """Load the trained model"""
        model_path = Path(self.model_path_var.get())
        
        if not model_path.exists():
            messagebox.showerror("Error", f"Model file not found:\n{model_path}")
            return
        
        try:
            print(f"Loading model from {model_path}...")
            
            # Load class mapping
            mapping_path = model_path.parent / 'class_mapping.json'
            if not mapping_path.exists():
                messagebox.showerror("Error", "class_mapping.json not found!\n"
                                            "Please ensure it's in the models directory.")
                return
            
            import json
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            
            self.idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}
            num_classes = len(self.idx_to_label)
            
            # Initialize and load model
            self.model = LipReadingModel(
                num_classes=num_classes,
                sequence_length=self.config['model']['sequence_length'],
                frame_height=self.config['model']['frame_height'],
                frame_width=self.config['model']['frame_width']
            )
            
            self.model.load_model(str(model_path))
            
            # Update info
            info = self.model.get_model_info()
            info_text = f"✓ Model loaded\nClasses: {num_classes}\n" \
                       f"Parameters: {info['total_parameters']:,}\n" \
                       f"Size: {info['model_size_mb']:.2f} MB"
            self.model_info_var.set(info_text)
            
            print(f"✓ Model loaded successfully")
            print(f"  Classes: {num_classes}")
            print(f"  Parameters: {info['total_parameters']:,}\n")
            
            messagebox.showinfo("Success", "Model loaded successfully!")
        
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def start_camera(self):
        """Start camera capture"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        if self.is_capturing:
            messagebox.showwarning("Warning", "Camera already running!")
            return
        
        try:
            camera_id = self.camera_id_var.get()
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {camera_id}")
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_capturing = True
            
            # Update buttons
            self.start_camera_btn.config(state=tk.DISABLED)
            self.stop_camera_btn.config(state=tk.NORMAL)
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self.capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            self.status_label.config(text="Status: Camera running", foreground="green")
            print("✓ Camera started\n")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")
            print(f"✗ Camera error: {e}\n")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_capturing = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Update buttons
        self.start_camera_btn.config(state=tk.NORMAL)
        self.stop_camera_btn.config(state=tk.DISABLED)
        
        # Clear video display
        self.video_label.config(image='', text="Camera stopped")
        
        self.status_label.config(text="Status: Camera stopped", foreground="red")
        print("Camera stopped\n")
    
    def capture_loop(self):
        """Main capture loop (runs in separate thread)"""
        import time
        
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.is_capturing:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to read frame")
                break
            
            # Mirror frame for better user experience
            frame = cv2.flip(frame, 1)
            
            # Add frame to buffer
            self.frame_buffer.append(frame.copy())
            
            # Process frame for enhanced lip feature extraction
            try:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe FaceMesh for enhanced landmark detection
                results = self.preprocessor.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    h, w = frame.shape[:2]
                    
                    # MediaPipe lip landmark indices
                    # Outer lip: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88
                    # Inner lip: 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
                    outer_lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]
                    inner_lip_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
                    
                    # Extract lip landmark coordinates
                    outer_lip_points = []
                    inner_lip_points = []
                    
                    for idx in outer_lip_indices:
                        landmark = face_landmarks.landmark[idx]
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        outer_lip_points.append([x, y])
                    
                    for idx in inner_lip_indices:
                        landmark = face_landmarks.landmark[idx]
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        inner_lip_points.append([x, y])
                    
                    outer_lip_points = np.array(outer_lip_points, dtype=np.float32)
                    inner_lip_points = np.array(inner_lip_points, dtype=np.float32)
                    
                    # Apply temporal smoothing to landmarks for stability
                    if self.previous_lip_landmarks is not None:
                        prev_outer, prev_inner = self.previous_lip_landmarks
                        # Exponential moving average for smooth tracking
                        alpha = self.landmark_smoothing_alpha
                        outer_lip_points = alpha * outer_lip_points + (1 - alpha) * prev_outer
                        inner_lip_points = alpha * inner_lip_points + (1 - alpha) * prev_inner
                    
                    # Store for next frame
                    self.previous_lip_landmarks = (outer_lip_points.copy(), inner_lip_points.copy())
                    
                    # Convert back to int for drawing
                    outer_lip_points_int = outer_lip_points.astype(np.int32)
                    inner_lip_points_int = inner_lip_points.astype(np.int32)
                    
                    # Draw smoothed lip landmarks
                    for point in outer_lip_points_int:
                        cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
                    for point in inner_lip_points_int:
                        cv2.circle(frame, tuple(point), 2, (255, 0, 0), -1)
                    
                    # Draw lip contours for visualization
                    cv2.polylines(frame, [outer_lip_points_int], True, (0, 255, 0), 2)
                    cv2.polylines(frame, [inner_lip_points_int], True, (255, 0, 0), 1)
                    
                    # Calculate lip bounding box with padding for ROI
                    lip_all_points = np.vstack([outer_lip_points_int, inner_lip_points_int])
                    x_min, y_min = np.min(lip_all_points, axis=0)
                    x_max, y_max = np.max(lip_all_points, axis=0)
                    
                    # Add padding (30% on each side)
                    padding_x = int((x_max - x_min) * 0.3)
                    padding_y = int((y_max - y_min) * 0.3)
                    
                    x_min = max(0, x_min - padding_x)
                    y_min = max(0, y_min - padding_y)
                    x_max = min(w, x_max + padding_x)
                    y_max = min(h, y_max + padding_y)
                    
                    # Draw enhanced ROI box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                    
                    # Calculate mouth opening metrics for visual feedback
                    mouth_width = x_max - x_min
                    mouth_height = y_max - y_min
                    aspect_ratio = mouth_height / (mouth_width + 1e-6)
                    
                    # Normalize lip coordinates relative to ROI center
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    outer_normalized = outer_lip_points - [center_x, center_y]
                    inner_normalized = inner_lip_points - [center_x, center_y]
                    
                    # Compute enhanced geometric features
                    lip_features = self.preprocessor._compute_geometric_features(
                        outer_normalized, inner_normalized
                    )
                    
                    # Apply temporal smoothing if buffer has data
                    if len(self.lip_buffer) > 0:
                        # Smooth with previous frame (exponential moving average)
                        alpha = 0.7  # Current frame weight
                        previous_features = self.lip_buffer[-1]
                        lip_features = alpha * lip_features + (1 - alpha) * previous_features
                    
                    # Add features to buffer
                    self.lip_buffer.append(lip_features)
                    
                    # Enhanced visual feedback
                    cv2.putText(frame, "✓ Lips tracked", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Buffer: {len(self.lip_buffer)}/75", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Opening: {aspect_ratio:.2f}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                    
                    # Show lip movement indicator
                    if len(self.lip_buffer) >= 2:
                        feature_diff = np.linalg.norm(self.lip_buffer[-1] - self.lip_buffer[-2])
                        movement_level = min(int(feature_diff * 10), 100)
                        color = (0, 255, 0) if movement_level > 5 else (0, 150, 255)
                        cv2.putText(frame, f"Movement: {movement_level}%", (10, 120),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(frame, "✗ No face detected", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            except Exception as e:
                print(f"Frame processing error: {e}")
                import traceback
                traceback.print_exc()
            
            # Make prediction if buffer is full
            if len(self.lip_buffer) == 75:
                try:
                    self.predict_sequence()
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = time.time()
                fps = fps_counter / (fps_end_time - fps_start_time)
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display stable prediction on frame (using romanized ASCII-safe text)
            # Color based on stability: Green if stable, Yellow if changing
            pred_color = (0, 255, 0) if self.stability_count == 0 else (0, 255, 255)
            
            cv2.putText(frame, f"Prediction: {self.current_prediction_display}", 
                       (10, frame.shape[0] - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2)
            
            # Show stable confidence (not fluctuating raw confidence)
            cv2.putText(frame, f"Confidence: {self.stable_confidence*100:.1f}%",
                       (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
            
            # Stability indicator
            if self.stable_prediction != "No prediction":
                stability_text = "STABLE" if self.stability_count == 0 else "UPDATING..."
                stability_color = (0, 255, 0) if self.stability_count == 0 else (0, 165, 255)
                cv2.putText(frame, f"[{stability_text}]",
                           (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)
            
            # Recording
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)
            
            # Add frame to queue for GUI update
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Skip frame if queue is full
        
        print("Capture loop ended")
    
    def predict_sequence(self):
        """Make prediction on current sequence with enhanced preprocessing"""
        if self.model is None or len(self.lip_buffer) < 75:
            return
        
        try:
            # Prepare sequence (geometric features)
            sequence = np.array(list(self.lip_buffer))
            print(f"[DEBUG] Raw sequence shape: {sequence.shape}")
            
            # Apply Savitzky-Golay filter for noise reduction while preserving edges
            # This smooths the feature trajectory over time
            try:
                # Apply smoothing to each feature dimension
                window_length = min(11, len(sequence) if len(sequence) % 2 == 1 else len(sequence) - 1)
                if window_length >= 5:
                    sequence = savgol_filter(sequence, window_length=window_length, 
                                            polyorder=3, axis=0)
                    print(f"[DEBUG] Applied Savitzky-Golay smoothing (window={window_length})")
            except Exception as e:
                print(f"[DEBUG] Smoothing skipped: {e}")
            
            # Remove outlier frames (z-score method)
            # Calculate z-scores for feature magnitudes
            feature_magnitudes = np.linalg.norm(sequence, axis=1)
            z_scores = np.abs((feature_magnitudes - np.mean(feature_magnitudes)) / 
                            (np.std(feature_magnitudes) + 1e-8))
            
            # Replace outliers with interpolated values
            outlier_threshold = 3.0
            outlier_mask = z_scores > outlier_threshold
            if np.any(outlier_mask):
                print(f"[DEBUG] Detected {np.sum(outlier_mask)} outlier frames, interpolating...")
                for i in np.where(outlier_mask)[0]:
                    # Interpolate from neighbors
                    if i > 0 and i < len(sequence) - 1:
                        sequence[i] = (sequence[i-1] + sequence[i+1]) / 2
                    elif i == 0 and len(sequence) > 1:
                        sequence[i] = sequence[i+1]
                    elif i == len(sequence) - 1 and len(sequence) > 1:
                        sequence[i] = sequence[i-1]
            
            # Add temporal features (velocity and acceleration)
            # This increases features from ~110 to ~330
            sequence = self.preprocessor._add_temporal_features(sequence)
            print(f"[DEBUG] After temporal features: {sequence.shape}")
            
            # Normalize features with robust scaling
            sequence = self.preprocessor._normalize_features(sequence)
            print(f"[DEBUG] After normalization: {sequence.shape}")
            
            # Add batch dimension: (1, 75, num_features)
            sequence = np.expand_dims(sequence, axis=0)
            print(f"[DEBUG] With batch dimension: {sequence.shape}")
            
            # Predict using the model wrapper (already handles batch extraction)
            predictions = self.model.predict(sequence)
            print(f"[DEBUG] Predictions: {predictions}, type: {type(predictions)}")
            
            # Ensure it's a numpy array
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            
            print(f"[DEBUG] Final predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'scalar'}")
            
            # Get top predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            
            # Update current prediction (raw)
            top_idx = top_indices[0]
            self.prediction_confidence = float(predictions[top_idx])
            
            # Store top 3 predictions
            self.top_predictions = [
                (self.idx_to_label[idx], float(predictions[idx]))
                for idx in top_indices
            ]
            
            # === PREDICTION STABILIZATION ===
            # Add to prediction history
            if self.prediction_confidence >= self.confidence_threshold_var.get():
                label = self.idx_to_label[top_idx]
                language, word = label.split('_', 1)
                current_pred = f"{word} ({language})"
                self.prediction_history.append({
                    'prediction': current_pred,
                    'word': word,
                    'confidence': self.prediction_confidence
                })
            else:
                self.prediction_history.append({
                    'prediction': 'Low confidence',
                    'word': None,
                    'confidence': self.prediction_confidence
                })
            
            # Analyze prediction history for stability
            if len(self.prediction_history) >= 3:
                # Get most recent predictions
                recent_predictions = list(self.prediction_history)[-3:]
                
                # Count occurrences of each prediction
                pred_counts = {}
                total_confidence = {}
                for pred_data in recent_predictions:
                    pred = pred_data['prediction']
                    conf = pred_data['confidence']
                    pred_counts[pred] = pred_counts.get(pred, 0) + 1
                    total_confidence[pred] = total_confidence.get(pred, 0) + conf
                
                # Find most frequent prediction
                most_frequent = max(pred_counts.items(), key=lambda x: x[1])
                frequent_pred = most_frequent[0]
                frequency = most_frequent[1]
                avg_confidence = total_confidence[frequent_pred] / frequency
                
                # Update stable prediction only if:
                # 1. It appears at least 2 times in last 3 predictions
                # 2. Average confidence is above threshold
                # 3. OR if it has very high confidence (>0.8) even once
                should_update = False
                
                if frequency >= 2 and avg_confidence >= self.prediction_change_threshold:
                    should_update = True
                elif self.prediction_confidence >= 0.80:
                    should_update = True
                    frequent_pred = current_pred
                
                if should_update and frequent_pred != "Low confidence":
                    # Check if it's actually different from current stable prediction
                    if frequent_pred != self.stable_prediction:
                        self.stability_count += 1
                        if self.stability_count >= self.stability_required:
                            # Update stable prediction
                            self.stable_prediction = frequent_pred
                            self.stable_confidence = avg_confidence
                            self.stability_count = 0
                            
                            # Extract word for romanization
                            if '(' in frequent_pred:
                                word_part = frequent_pred.split('(')[0].strip()
                                lang_part = frequent_pred.split('(')[1].strip(')')
                                
                                word_romanized = {
                                    'ನಮಸ್ಕಾರ': 'Namaskara',
                                    'ರಾಮ': 'Rama',
                                    'hello': 'Hello',
                                }.get(word_part, word_part)
                                
                                self.stable_prediction_display = f"{word_romanized} ({lang_part})"
                            else:
                                self.stable_prediction_display = frequent_pred
                            
                            print(f"[STABLE] Prediction updated to: {self.stable_prediction} (confidence: {self.stable_confidence:.2%})")
                    else:
                        # Same prediction, reset stability counter
                        self.stability_count = 0
            
            # Update display with stable prediction
            self.current_prediction = self.stable_prediction
            self.current_prediction_display = self.stable_prediction_display
            
            print(f"[DEBUG] Raw: {label if self.prediction_confidence >= self.confidence_threshold_var.get() else 'Low'} ({self.prediction_confidence:.2%}) | Stable: {self.stable_prediction} ({self.stable_confidence:.2%})")
        
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            self.current_prediction = "Error"
    
    def update_gui(self):
        """Update GUI with latest frame (runs in main thread)"""
        try:
            # Get frame from queue
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                
                # Convert to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize if needed
                max_width = 640
                max_height = 480
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image=img)
                
                # Update label
                self.video_label.config(image=photo, text='')
                self.video_label.image = photo  # Keep reference
            
            # Update prediction display
            self.prediction_label.config(text=self.current_prediction)
            self.confidence_label.config(
                text=f"Confidence: {self.prediction_confidence*100:.1f}%"
            )
            
            # Update top predictions
            if self.top_predictions:
                self.top_predictions_text.config(state=tk.NORMAL)
                self.top_predictions_text.delete(1.0, tk.END)
                
                for i, (label, conf) in enumerate(self.top_predictions, 1):
                    language, word = label.split('_', 1)
                    text = f"{i}. {word} ({language})\n   {conf*100:.1f}%\n"
                    self.top_predictions_text.insert(tk.END, text)
                
                self.top_predictions_text.config(state=tk.DISABLED)
        
        except queue.Empty:
            pass
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # Schedule next update
        self.root.after(30, self.update_gui)  # ~33 FPS GUI update
    
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.is_capturing:
            messagebox.showwarning("Warning", "Start camera first!")
            return
        
        if not self.is_recording:
            # Start recording
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./outputs/recordings/recording_{timestamp}.mp4"
            
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 25.0, (640, 480))
            
            self.is_recording = True
            self.record_btn.config(text="Stop Recording")
            self.status_label.config(text=f"Status: Recording to {filename}", 
                                   foreground="red")
            print(f"Recording started: {filename}")
        else:
            # Stop recording
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            self.record_btn.config(text="Record Video")
            self.status_label.config(text="Status: Recording stopped", 
                                   foreground="blue")
            print("Recording stopped\n")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_capturing:
            self.stop_camera()
        
        if self.is_recording:
            self.toggle_recording()
        
        self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()
    app = PredictionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
