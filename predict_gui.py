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
from PIL import Image, ImageTk
import threading
import queue
import sys
import os
from collections import deque
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import LipReadingModel
from src.preprocessor import VideoPreprocessor
from src.utils import *


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
        self.prediction_confidence = 0.0
        self.top_predictions = []
        
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
            
            # Process frame for lip extraction
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_bbox = self.preprocessor._detect_face(rgb_frame)
                
                if face_bbox is not None:
                    # Draw face bounding box
                    cv2.rectangle(frame,
                                (face_bbox.left(), face_bbox.top()),
                                (face_bbox.right(), face_bbox.bottom()),
                                (0, 255, 0), 2)
                    
                    # Detect landmarks
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    landmarks = self.preprocessor._detect_landmarks(gray, face_bbox)
                    
                    if landmarks is not None:
                        # Draw lip landmarks
                        mouth_points = landmarks[self.preprocessor.MOUTH_LANDMARKS]
                        for point in mouth_points:
                            cv2.circle(frame, tuple(point), 2, (0, 0, 255), -1)
                        
                        # Extract lip region
                        lip_roi = self.preprocessor._extract_lip_region(frame, landmarks)
                        if lip_roi is not None:
                            self.lip_buffer.append(lip_roi)
                    
                    # Add status text
                    cv2.putText(frame, "Face detected", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            except Exception as e:
                print(f"Frame processing error: {e}")
            
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
            
            # Display current prediction on frame
            cv2.putText(frame, f"Prediction: {self.current_prediction}", 
                       (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {self.prediction_confidence*100:.1f}%",
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
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
        """Make prediction on current sequence"""
        if self.model is None or len(self.lip_buffer) < 75:
            return
        
        try:
            # Prepare sequence
            sequence = np.array(list(self.lip_buffer))
            sequence = sequence / 255.0  # Normalize
            
            # Predict
            predictions = self.model.predict(sequence)
            
            # Get top predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            
            # Update current prediction
            top_idx = top_indices[0]
            self.prediction_confidence = predictions[top_idx]
            
            # Check confidence threshold
            if self.prediction_confidence >= self.confidence_threshold_var.get():
                label = self.idx_to_label[top_idx]
                language, word = label.split('_', 1)
                self.current_prediction = f"{word} ({language})"
            else:
                self.current_prediction = "Low confidence"
            
            # Store top 3 predictions
            self.top_predictions = [
                (self.idx_to_label[idx], predictions[idx])
                for idx in top_indices
            ]
        
        except Exception as e:
            print(f"Prediction error: {e}")
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
