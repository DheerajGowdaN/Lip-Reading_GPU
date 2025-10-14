"""
Enhanced Training GUI with Video Recording for Multi-Lingual Lip Reading System
Includes webcam recording to capture training videos
Author: AI Assistant
Date: October 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
from pathlib import Path
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
import os
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image, ImageTk
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import LipReadingModel
from src.preprocessor import VideoPreprocessor
from src.data_loader import DataLoader
from src.utils import *


class VideoRecorder:
    """Video recorder for capturing training data"""
    
    def __init__(self, output_dir, language, word):
        # Create directory structure: language/word/
        self.output_dir = Path(output_dir) / language / word
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.word = word
        self.language = language
        self.is_recording = False
        self.cap = None
        self.out = None
        self.frames = []
        
    def start_recording(self):
        """Start recording from webcam"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        
        self.is_recording = True
        self.frames = []
        
    def capture_frame(self):
        """Capture a single frame"""
        if not self.is_recording or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.frames.append(frame.copy())
            return frame
        return None
    
    def stop_recording(self):
        """Stop recording and save video"""
        self.is_recording = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if len(self.frames) == 0:
            return None
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Find next available ID
        existing_files = list(self.output_dir.glob(f"{self.word}_*.mp4"))
        next_id = len(existing_files) + 1
        filename = f"{self.word}_{next_id:03d}_{timestamp}.mp4"
        output_path = self.output_dir / filename
        
        # Save video
        height, width = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 25.0, (width, height))
        
        for frame in self.frames:
            out.write(frame)
        
        out.release()
        self.frames = []
        
        return output_path


class TrainingGUIWithRecording:
    """Enhanced GUI with video recording capability"""
    
    @staticmethod
    def check_gpu_availability():
        """
        Check if GPU is available for TensorFlow
        Returns dict with GPU status information
        """
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': [],
            'cuda_built': False,
            'warning': None
        }
        
        try:
            # Check if TensorFlow is built with CUDA
            gpu_info['cuda_built'] = tf.test.is_built_with_cuda()
            
            if not gpu_info['cuda_built']:
                gpu_info['warning'] = (
                    "TensorFlow is NOT built with CUDA support!\n"
                    "GPU training will NOT work.\n\n"
                    "Solution:\n"
                    "1. pip uninstall tensorflow\n"
                    "2. pip install tensorflow[and-cuda]\n\n"
                    "See GPU_SETUP_GUIDE.md for details."
                )
                return gpu_info
            
            # Check for GPU devices
            gpus = tf.config.list_physical_devices('GPU')
            gpu_info['count'] = len(gpus)
            gpu_info['devices'] = [gpu.name for gpu in gpus]
            gpu_info['available'] = len(gpus) > 0
            
            if not gpu_info['available']:
                gpu_info['warning'] = (
                    "No GPU devices found!\n\n"
                    "Possible causes:\n"
                    "1. No NVIDIA GPU in system\n"
                    "2. GPU drivers not installed\n"
                    "3. CUDA toolkit not installed\n\n"
                    "Run: python verify_gpu.py\n"
                    "See GPU_SETUP_GUIDE.md for setup."
                )
            
        except Exception as e:
            gpu_info['warning'] = f"Error checking GPU: {str(e)}"
        
        return gpu_info
    
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Lingual Lip Reading - Training with Recording")
        self.root.geometry("1400x900")
        
        # Check GPU availability at startup
        self.gpu_available = self.check_gpu_availability()
        
        # Initialize variables
        self.config = load_config()
        self.data_loader = None
        self.preprocessor = None
        self.model = None
        self.training_thread = None
        self.is_training = False
        self.is_paused = False
        
        # Recording variables
        self.recorder = None
        self.recording_thread = None
        self.is_showing_camera = False
        self.camera_label = None
        self.record_countdown = 0
        self.show_lip_tracking = tk.BooleanVar(value=True)
        
        # Initialize MediaPipe Face Mesh for lip tracking
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Training metrics
        self.epochs_completed = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.bias_values = []
        self.variance_values = []
        
        # Time tracking
        self.training_start_time = None
        self.epoch_start_time = None
        self.epoch_times = []
        self.estimated_time_remaining = 0
        
        # Setup GUI
        self.setup_ui()
        
        # Show GPU status warning if needed
        self.show_gpu_status_warning()
        
        # Initialize system
        self.initialize_system()
    
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
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Multi-Lingual Lip Reading - Training Interface with Recording",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        # Left side - Training components
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, rowspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(3, weight=1)
        
        # Top Panel - Data Configuration
        self.setup_data_panel(left_frame)
        
        # Middle Panel - Training Configuration
        self.setup_training_panel(left_frame)
        
        # Bottom Panel - Metrics and Progress
        self.setup_metrics_panel(left_frame)
        
        # Console Output
        self.setup_console_panel(left_frame)
        
        # Right side - Recording panel
        self.setup_recording_panel(main_frame)
    
    def setup_recording_panel(self, parent):
        """Setup video recording panel"""
        frame = ttk.LabelFrame(parent, text="Video Recording", padding="10")
        frame.grid(row=1, column=1, rowspan=4, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        
        # Recording configuration
        config_frame = ttk.Frame(frame)
        config_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        config_frame.columnconfigure(1, weight=1)
        
        # Language for recording
        ttk.Label(config_frame, text="Language:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.rec_language_var = tk.StringVar(value="english")
        rec_language_combo = ttk.Combobox(
            config_frame,
            textvariable=self.rec_language_var,
            values=["english", "hindi", "kannada"],
            state="readonly",
            width=15
        )
        rec_language_combo.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Word to record
        ttk.Label(config_frame, text="Word:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.rec_word_var = tk.StringVar()
        rec_word_entry = ttk.Entry(config_frame, textvariable=self.rec_word_var)
        rec_word_entry.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Recording duration
        ttk.Label(config_frame, text="Duration (sec):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.rec_duration_var = tk.IntVar(value=3)
        duration_spin = ttk.Spinbox(
            config_frame,
            from_=2,
            to=10,
            textvariable=self.rec_duration_var,
            width=5
        )
        duration_spin.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Lip tracking toggle
        ttk.Label(config_frame, text="Show Lip Tracking:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        lip_tracking_check = ttk.Checkbutton(
            config_frame,
            variable=self.show_lip_tracking,
            text="Enable"
        )
        lip_tracking_check.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Recording buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=0, pady=10, sticky=(tk.W, tk.E))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        self.start_camera_btn = ttk.Button(
            btn_frame,
            text="📹 Start Camera",
            command=self.start_camera
        )
        self.start_camera_btn.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.record_btn = ttk.Button(
            btn_frame,
            text="⏺ Record Video",
            command=self.start_recording,
            state="disabled"
        )
        self.record_btn.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.stop_camera_btn = ttk.Button(
            btn_frame,
            text="⏹ Stop Camera",
            command=self.stop_camera,
            state="disabled"
        )
        self.stop_camera_btn.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Camera feed
        camera_frame = ttk.LabelFrame(frame, text="Camera Feed", padding="5")
        camera_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        camera_frame.columnconfigure(0, weight=1)
        camera_frame.rowconfigure(0, weight=1)
        
        self.camera_label = ttk.Label(camera_frame, text="Camera Off", anchor="center")
        self.camera_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Recording status
        self.rec_status_var = tk.StringVar(value="Ready to record")
        status_label = ttk.Label(
            frame,
            textvariable=self.rec_status_var,
            font=("Arial", 10),
            foreground="blue"
        )
        status_label.grid(row=3, column=0, pady=5)
        
        # Videos recorded list
        list_frame = ttk.LabelFrame(frame, text="Recorded Videos", padding="5")
        list_frame.grid(row=4, column=0, pady=5, sticky=(tk.W, tk.E))
        list_frame.columnconfigure(0, weight=1)
        
        self.video_listbox = tk.Listbox(list_frame, height=6)
        self.video_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.video_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.video_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Refresh button
        ttk.Button(
            frame,
            text="🔄 Refresh Video List",
            command=self.refresh_video_list
        ).grid(row=5, column=0, pady=5, sticky=(tk.W, tk.E))
    
    def start_camera(self):
        """Start camera feed"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open webcam")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_showing_camera = True
            self.start_camera_btn.config(state="disabled")
            self.stop_camera_btn.config(state="normal")
            self.record_btn.config(state="normal")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.update_camera_feed, daemon=True)
            self.camera_thread.start()
            
            self.rec_status_var.set("Camera started - Ready to record")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def update_camera_feed(self):
        """Update camera feed in GUI with lip tracking visualization"""
        while self.is_showing_camera:
            ret, frame = self.cap.read()
            if ret:
                # Apply lip tracking visualization if enabled
                if self.show_lip_tracking.get():
                    frame = self._draw_lip_tracking(frame)
                
                # Draw recording indicator if recording
                if hasattr(self, 'recorder') and self.recorder and self.recorder.is_recording:
                    cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 0, 255), 2)
                    
                    # Show countdown if active
                    if self.record_countdown > 0:
                        cv2.putText(frame, str(self.record_countdown), (frame.shape[1]//2 - 50, frame.shape[0]//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
                
                # Convert to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((400, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)
                
                # Update label
                self.camera_label.configure(image=photo, text="")
                self.camera_label.image = photo
            
            time.sleep(0.03)  # ~30 FPS
    
    def _draw_lip_tracking(self, frame):
        """Draw lip landmarks and bounding box on frame"""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe Face Mesh
            results = self.face_mesh_detector.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                
                # Lip landmark indices (outer and inner)
                outer_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 
                                    146, 91, 181, 84, 17, 314, 405, 321, 375]
                inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
                
                # Draw outer lip landmarks and contour
                outer_points = []
                for idx in outer_lip_indices:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    outer_points.append((x, y))
                    # Draw landmark point
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Draw outer lip contour
                outer_points_np = np.array(outer_points, dtype=np.int32)
                cv2.polylines(frame, [outer_points_np], True, (0, 255, 0), 2)
                
                # Draw inner lip landmarks and contour
                inner_points = []
                for idx in inner_lip_indices:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    inner_points.append((x, y))
                    # Draw landmark point
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                
                # Draw inner lip contour
                inner_points_np = np.array(inner_points, dtype=np.int32)
                cv2.polylines(frame, [inner_points_np], True, (255, 0, 0), 2)
                
                # Draw bounding box around mouth
                all_points = outer_points + inner_points
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                
                # Add label
                cv2.putText(frame, "Lip Region", (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Calculate and display mouth metrics
                mouth_width = x_max - x_min
                mouth_height = y_max - y_min
                aspect_ratio = mouth_height / (mouth_width + 1e-6)
                
                # Display metrics on frame
                cv2.putText(frame, f"Width: {mouth_width}px", (10, h - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Height: {mouth_height}px", (10, h - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Ratio: {aspect_ratio:.3f}", (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        except Exception as e:
            # If error occurs, just return original frame
            pass
        
        return frame
    
    def start_recording(self):
        """Start recording a video"""
        word = self.rec_word_var.get().strip().lower()
        if not word:
            messagebox.showwarning("Warning", "Please enter a word to record")
            return
        
        language = self.rec_language_var.get()
        duration = self.rec_duration_var.get()
        
        # Disable buttons during recording
        self.record_btn.config(state="disabled")
        self.stop_camera_btn.config(state="disabled")
        
        # Start recording in thread
        recording_thread = threading.Thread(
            target=self._record_video_thread,
            args=(language, word, duration),
            daemon=True
        )
        recording_thread.start()
    
    def _record_video_thread(self, language, word, duration):
        """Recording thread"""
        try:
            # Countdown
            for i in range(3, 0, -1):
                self.record_countdown = i
                self.rec_status_var.set(f"Starting in {i}...")
                time.sleep(1)
            
            self.record_countdown = 0
            self.rec_status_var.set(f"Recording '{word}' - Say it now!")
            
            # Create recorder
            data_dir = self.config['paths']['video_dir']
            self.recorder = VideoRecorder(data_dir, language, word)
            self.recorder.start_recording()
            
            # Record for duration
            start_time = time.time()
            while time.time() - start_time < duration:
                frame = self.recorder.capture_frame()
                time.sleep(0.04)  # 25 FPS
            
            # Stop recording
            output_path = self.recorder.stop_recording()
            
            if output_path:
                self.rec_status_var.set(f"Video saved: {output_path.name}")
                messagebox.showinfo("Success", f"Video recorded successfully!\n\nSaved to:\n{output_path}")
                self.refresh_video_list()
            else:
                self.rec_status_var.set("Recording failed - no frames captured")
            
            self.recorder = None
            
        except Exception as e:
            self.rec_status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Recording failed: {str(e)}")
        
        finally:
            # Re-enable buttons
            self.record_btn.config(state="normal")
            self.stop_camera_btn.config(state="normal")
    
    def stop_camera(self):
        """Stop camera feed"""
        self.is_showing_camera = False
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_label.configure(image='', text="Camera Off")
        self.camera_label.image = None
        
        self.start_camera_btn.config(state="normal")
        self.stop_camera_btn.config(state="disabled")
        self.record_btn.config(state="disabled")
        
        self.rec_status_var.set("Camera stopped")
    
    def __del__(self):
        """Cleanup when GUI is closed"""
        if hasattr(self, 'face_mesh_detector'):
            self.face_mesh_detector.close()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
    
    def refresh_video_list(self):
        """Refresh the list of recorded videos"""
        self.video_listbox.delete(0, tk.END)
        
        data_dir = Path(self.config['paths']['video_dir'])
        language = self.rec_language_var.get()
        word = self.rec_word_var.get()
        
        # Show videos for specific word folder
        word_dir = data_dir / language / word
        
        if word_dir.exists():
            videos = sorted(word_dir.glob("*.mp4"))
            for video in videos:
                self.video_listbox.insert(tk.END, video.name)
            
            if len(videos) > 0:
                self.rec_status_var.set(f"Found {len(videos)} videos for '{word}' in {language}")
            else:
                self.rec_status_var.set(f"No videos found for '{word}' in {language}")
        else:
            # If word folder doesn't exist, show all videos in language folder (backward compatibility)
            lang_dir = data_dir / language
            if lang_dir.exists():
                # Check for nested structure
                subdirs = [d for d in lang_dir.iterdir() if d.is_dir()]
                if subdirs:
                    # Show summary of all words
                    total_videos = 0
                    for subdir in subdirs:
                        videos = list(subdir.glob("*.mp4"))
                        total_videos += len(videos)
                        for video in videos:
                            self.video_listbox.insert(tk.END, f"{subdir.name}/{video.name}")
                    self.rec_status_var.set(f"Found {total_videos} videos across {len(subdirs)} words in {language}")
                else:
                    # Old flat structure
                    videos = sorted(lang_dir.glob("*.mp4"))
                    for video in videos:
                        self.video_listbox.insert(tk.END, video.name)
                    self.rec_status_var.set(f"Found {len(videos)} videos in {language} (flat structure)")
            else:
                self.rec_status_var.set(f"Language folder not found: {language}")
    
    # Copy all other methods from original train_gui.py
    def setup_data_panel(self, parent):
        """Setup data configuration panel"""
        frame = ttk.LabelFrame(parent, text="Data Configuration", padding="10")
        frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        frame.columnconfigure(1, weight=1)
        
        # Language selection
        ttk.Label(frame, text="Language:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.language_var = tk.StringVar(value="english")
        language_combo = ttk.Combobox(
            frame,
            textvariable=self.language_var,
            values=["english", "hindi", "kannada", "all"],
            state="readonly"
        )
        language_combo.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Video directory
        ttk.Label(frame, text="Video Directory:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.video_dir_var = tk.StringVar(value=self.config['paths']['video_dir'])
        ttk.Entry(frame, textvariable=self.video_dir_var, state="readonly").grid(
            row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E)
        )
        ttk.Button(frame, text="Browse", command=self.browse_videos).grid(
            row=1, column=2, padx=5, pady=5
        )
        
        # Buttons frame
        buttons_frame = ttk.Frame(frame)
        buttons_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        ttk.Button(buttons_frame, text="🔍 Scan Dataset", command=self.scan_dataset).grid(
            row=0, column=0, padx=5
        )
        
        ttk.Button(buttons_frame, text="⚙️ Preprocess Data", command=self.preprocess_dataset).grid(
            row=0, column=1, padx=5
        )
        
        # Dataset info
        self.dataset_info_var = tk.StringVar(value="No dataset loaded")
        ttk.Label(frame, textvariable=self.dataset_info_var, foreground="blue").grid(
            row=3, column=0, columnspan=3, pady=5
        )
        
        # Preprocessing status
        self.preprocessing_status_var = tk.StringVar(value="")
        self.preprocessing_status_label = ttk.Label(
            frame,
            textvariable=self.preprocessing_status_var,
            foreground="green",
            font=("Arial", 9)
        )
        self.preprocessing_status_label.grid(row=4, column=0, columnspan=3, pady=5)
        
        # Preprocessing progress bar
        self.preprocess_progress_var = tk.DoubleVar(value=0)
        self.preprocess_progress_bar = ttk.Progressbar(
            frame,
            variable=self.preprocess_progress_var,
            maximum=100,
            mode='determinate'
        )
        self.preprocess_progress_bar.grid(row=5, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
    
    def setup_training_panel(self, parent):
        """Setup training configuration panel"""
        frame = ttk.LabelFrame(parent, text="Training Configuration", padding="10")
        frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Create two columns
        left_frame = ttk.Frame(frame)
        left_frame.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E, tk.N))
        
        right_frame = ttk.Frame(frame)
        right_frame.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E, tk.N))
        
        # Left column
        ttk.Label(left_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(left_frame, from_=1, to=1000, textvariable=self.epochs_var, width=10).grid(
            row=0, column=1, padx=5, pady=5
        )
        
        ttk.Label(left_frame, text="Batch Size:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.batch_size_var = tk.IntVar(value=8)
        ttk.Spinbox(left_frame, from_=1, to=64, textvariable=self.batch_size_var, width=10).grid(
            row=1, column=1, padx=5, pady=5
        )
        
        ttk.Label(left_frame, text="Learning Rate:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(left_frame, textvariable=self.lr_var, width=12).grid(
            row=2, column=1, padx=5, pady=5
        )
        
        # Right column
        ttk.Label(right_frame, text="Device:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.device_var = tk.StringVar(value="CPU")
        device_combo = ttk.Combobox(
            right_frame,
            textvariable=self.device_var,
            values=["CPU", "GPU"],
            state="readonly",
            width=10
        )
        device_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # GPU Status indicator
        gpu_status_text = "❌ Not available" if not self.gpu_available['available'] else f"✓ {self.gpu_available['count']} GPU(s)"
        gpu_status_color = "red" if not self.gpu_available['available'] else "green"
        self.gpu_status_label = ttk.Label(
            right_frame,
            text=f"GPU: {gpu_status_text}",
            foreground=gpu_status_color,
            font=("Arial", 9)
        )
        self.gpu_status_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(right_frame, text="Use Attention:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.attention_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right_frame, variable=self.attention_var).grid(
            row=1, column=1, padx=5, pady=5, sticky=tk.W
        )
        
        # Training buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.train_btn = ttk.Button(btn_frame, text="▶ Start Training", command=self.start_training)
        self.train_btn.grid(row=0, column=0, padx=5)
        
        self.pause_btn = ttk.Button(btn_frame, text="⏸ Pause", command=self.pause_training, state="disabled")
        self.pause_btn.grid(row=0, column=1, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self.stop_training, state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=5)
        
        # Progress bar
        progress_frame = ttk.Frame(frame)
        progress_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        progress_frame.columnconfigure(0, weight=1)
        
        ttk.Label(progress_frame, text="Training Progress:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=400
        )
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.progress_label = ttk.Label(progress_frame, text="0%", font=("Arial", 9))
        self.progress_label.grid(row=1, column=1, padx=5)
    
    def setup_metrics_panel(self, parent):
        """Setup metrics display panel"""
        frame = ttk.LabelFrame(parent, text="Training Metrics", padding="10")
        frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 4), dpi=80)
        
        # Loss plot
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True)
        
        # Accuracy plot
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.grid(True)
        
        self.fig.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Metrics labels - Row 1
        metrics_frame = ttk.Frame(frame)
        metrics_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.epoch_label = ttk.Label(metrics_frame, text="Epoch: 0/0", font=("Arial", 10, "bold"))
        self.epoch_label.grid(row=0, column=0, padx=10)
        
        self.time_remaining_label = ttk.Label(metrics_frame, text="Time Remaining: --:--:--", font=("Arial", 10, "bold"), foreground="blue")
        self.time_remaining_label.grid(row=0, column=1, padx=10)
        
        self.elapsed_time_label = ttk.Label(metrics_frame, text="Elapsed: 00:00:00", font=("Arial", 10))
        self.elapsed_time_label.grid(row=0, column=2, padx=10)
        
        # Metrics labels - Row 2
        metrics_frame2 = ttk.Frame(frame)
        metrics_frame2.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        
        self.loss_label = ttk.Label(metrics_frame2, text="Loss: N/A", font=("Arial", 10, "bold"), foreground="red")
        self.loss_label.grid(row=0, column=0, padx=10)
        
        self.acc_label = ttk.Label(metrics_frame2, text="Accuracy: N/A", font=("Arial", 10, "bold"), foreground="green")
        self.acc_label.grid(row=0, column=1, padx=10)
        
        self.bias_label = ttk.Label(metrics_frame2, text="Bias: N/A", font=("Arial", 10))
        self.bias_label.grid(row=0, column=2, padx=10)
        
        self.variance_label = ttk.Label(metrics_frame2, text="Variance: N/A", font=("Arial", 10))
        self.variance_label.grid(row=0, column=3, padx=10)
        
        self.epoch_time_label = ttk.Label(metrics_frame2, text="Epoch Time: --:--", font=("Arial", 10))
        self.epoch_time_label.grid(row=0, column=4, padx=10)
    
    def setup_console_panel(self, parent):
        """Setup console output panel"""
        frame = ttk.LabelFrame(parent, text="Console Output", padding="10")
        frame.grid(row=3, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        
        self.console = scrolledtext.ScrolledText(frame, height=8, wrap=tk.WORD)
        self.console.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def show_gpu_status_warning(self):
        """Show GPU status warning dialog if there are issues"""
        if self.gpu_available['warning']:
            # Create info box with warning
            response = messagebox.askquestion(
                "GPU Status Warning",
                f"{self.gpu_available['warning']}\n\n"
                "Do you want to continue?\n\n"
                "• Click 'Yes' to continue with current setup\n"
                "• Click 'No' to exit and fix GPU setup",
                icon='warning'
            )
            
            if response == 'no':
                self.log("Application closed by user to fix GPU setup")
                self.root.quit()
        elif self.gpu_available['available']:
            # Show success message briefly
            self.log(f"✓ GPU ready: {', '.join(self.gpu_available['devices'])}")
    
    def initialize_system(self):
        """Initialize the system"""
        self.log("System initializing...")
        self.log(f"Python version: {sys.version}")
        self.log(f"TensorFlow version: {tf.__version__}")
        self.log(f"TensorFlow built with CUDA: {self.gpu_available['cuda_built']}")
        
        # Display GPU status
        if self.gpu_available['available']:
            self.log(f"✓ GPU available: {self.gpu_available['count']} device(s)")
            for device in self.gpu_available['devices']:
                self.log(f"  - {device}")
        else:
            self.log("⚠ No GPU detected - training will use CPU")
            if not self.gpu_available['cuda_built']:
                self.log("⚠ TensorFlow not built with CUDA support")
                self.log("  Run: python verify_gpu.py for detailed diagnosis")
        
        self.log("✓ System ready")
    
    def browse_videos(self):
        """Browse for video directory"""
        directory = filedialog.askdirectory(title="Select Video Directory")
        if directory:
            self.video_dir_var.set(directory)
    
    def scan_dataset(self):
        """Scan and load dataset"""
        self.log("Scanning dataset...")
        
        try:
            language = self.language_var.get()
            video_dir = self.video_dir_var.get()
            
            # Determine languages to scan
            if language == "all":
                languages = ["english", "hindi", "kannada"]
            else:
                languages = [language]
            
            # Initialize data loader
            self.data_loader = DataLoader(
                data_dir=video_dir,
                preprocessed_dir=self.config['paths']['preprocessed_dir'],
                languages=languages
            )
            
            # Scan dataset
            self.data_loader.scan_dataset()
            
            # Get info
            total_videos = len(self.data_loader.video_paths)
            num_classes = len(self.data_loader.label_to_idx)
            class_names = list(self.data_loader.label_to_idx.keys())
            
            info_text = f"Found {total_videos} videos, {num_classes} classes ({', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''})"
            self.dataset_info_var.set(info_text)
            self.log(f"✓ {info_text}")
            
            if total_videos == 0:
                messagebox.showwarning("Warning", "No videos found! Please record some videos first.")
            
        except Exception as e:
            self.log(f"✗ Error scanning dataset: {str(e)}")
            messagebox.showerror("Error", f"Failed to scan dataset: {str(e)}")
    
    def preprocess_dataset(self):
        """Preprocess all videos and extract geometric features"""
        if self.data_loader is None or len(self.data_loader.video_paths) == 0:
            messagebox.showwarning("Warning", "Please scan dataset first!")
            return
        
        # Confirm preprocessing
        total_videos = len(self.data_loader.video_paths)
        response = messagebox.askyesno(
            "Confirm Preprocessing",
            f"This will preprocess {total_videos} videos and extract lip geometry features.\n\n"
            "This may take several minutes depending on the number of videos.\n\n"
            "Continue?"
        )
        
        if not response:
            return
        
        # Start preprocessing in separate thread
        preprocessing_thread = threading.Thread(
            target=self._preprocess_thread,
            daemon=True
        )
        preprocessing_thread.start()
    
    def _preprocess_thread(self):
        """Thread for preprocessing videos"""
        try:
            self.preprocessing_status_var.set("Initializing preprocessor...")
            self.log("\n" + "="*60)
            self.log("PREPROCESSING STARTED")
            self.log("="*60)
            
            # Initialize preprocessor
            shape_predictor_path = self.config['paths'].get(
                'shape_predictor',
                './models/shape_predictor_68_face_landmarks.dat'
            )
            
            preprocessor = VideoPreprocessor(
                shape_predictor_path=shape_predictor_path,
                target_size=(100, 100),
                sequence_length=75,
                target_fps=25,
                augment=False
            )
            
            self.log("✓ Preprocessor initialized")
            
            # Get list of videos to process
            total_videos = len(self.data_loader.video_paths)
            preprocessed_count = 0
            failed_count = 0
            
            preprocessed_dir = Path(self.config['paths']['preprocessed_dir'])
            preprocessed_dir.mkdir(parents=True, exist_ok=True)
            
            self.log(f"Processing {total_videos} videos...")
            
            for i, (video_path, label_idx) in enumerate(zip(
                self.data_loader.video_paths,
                self.data_loader.labels
            )):
                # Update progress
                progress_percent = (i / total_videos) * 100
                self.preprocess_progress_var.set(progress_percent)
                self.preprocessing_status_var.set(
                    f"Processing video {i+1}/{total_videos}: {Path(video_path).name}"
                )
                
                # Generate output path
                label = self.data_loader.idx_to_label[label_idx]
                language, word = label.split('_', 1)
                
                output_dir = preprocessed_dir / language
                output_dir.mkdir(parents=True, exist_ok=True)
                
                video_name = Path(video_path).stem
                output_path = output_dir / f"{video_name}.npy"
                
                # Skip if already preprocessed
                if output_path.exists():
                    self.log(f"  Skipped (already exists): {video_name}")
                    preprocessed_count += 1
                    continue
                
                try:
                    # Process video and extract geometric features
                    sequence = preprocessor.process_video(video_path)
                    
                    # Save to disk
                    np.save(output_path, sequence)
                    
                    preprocessed_count += 1
                    self.log(f"  ✓ Processed: {video_name} → {sequence.shape}")
                    
                except Exception as e:
                    failed_count += 1
                    self.log(f"  ✗ Failed: {video_name} - {str(e)}")
            
            # Update progress to 100%
            self.preprocess_progress_var.set(100)
            
            # Final status
            self.preprocessing_status_var.set(
                f"Completed: {preprocessed_count}/{total_videos} processed, {failed_count} failed"
            )
            
            self.log("="*60)
            self.log(f"PREPROCESSING COMPLETED")
            self.log(f"  Total: {total_videos}")
            self.log(f"  Processed: {preprocessed_count}")
            self.log(f"  Failed: {failed_count}")
            self.log("="*60)
            
            messagebox.showinfo(
                "Preprocessing Complete",
                f"Successfully preprocessed {preprocessed_count} videos!\n\n"
                f"Failed: {failed_count}\n\n"
                "Geometric features extracted and saved."
            )
            
        except Exception as e:
            self.log(f"✗ Preprocessing failed: {str(e)}")
            self.preprocessing_status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Preprocessing failed:\n{str(e)}")
    
    def start_training(self):
        """Start model training"""
        if self.is_training:
            messagebox.showwarning("Warning", "Training already in progress")
            return
        
        if self.data_loader is None or len(self.data_loader.video_paths) == 0:
            messagebox.showwarning("Warning", "Please scan dataset first and ensure videos are available")
            return
        
        # Start training in thread
        self.is_training = True
        self.train_btn.config(state="disabled")
        self.pause_btn.config(state="normal")
        self.stop_btn.config(state="normal")
        
        self.training_thread = threading.Thread(target=self._training_thread, daemon=True)
        self.training_thread.start()
    
    def _training_thread(self):
        """Training thread"""
        try:
            # Reset time tracking
            self.training_start_time = None
            self.epoch_start_time = None
            self.epoch_times = []
            self.epochs_completed = 0
            self.train_losses = []
            self.val_losses = []
            self.train_accs = []
            self.val_accs = []
            self.bias_values = []
            self.variance_values = []
            
            self.log("\n" + "="*60)
            self.log("TRAINING STARTED")
            self.log("="*60)
            
            # Get parameters
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            learning_rate = self.lr_var.get()
            device = self.device_var.get()
            use_attention = self.attention_var.get()
            
            # Initialize preprocessor
            self.log("Initializing preprocessor...")
            shape_predictor_path = self.config['paths'].get('shape_predictor', './models/shape_predictor_68_face_landmarks.dat')
            self.preprocessor = VideoPreprocessor(
                shape_predictor_path=shape_predictor_path,
                target_size=(100, 100),
                sequence_length=75,
                target_fps=25,
                augment=True
            )
            
            # Split dataset
            self.log("Splitting dataset...")
            train_data, val_data = self.data_loader.split_dataset()
            
            self.log(f"Train: {len(train_data[0])}, Val: {len(val_data[0])}")
            
            # Create data generators
            self.log("Creating data generators...")
            train_gen = self.data_loader.create_data_generator(
                train_data[0],
                train_data[1],
                self.preprocessor,
                augment=True
            )
            
            val_gen = self.data_loader.create_data_generator(
                val_data[0],
                val_data[1],
                self.preprocessor,
                augment=False
            )
            
            # Detect number of features from first video
            self.log("Detecting feature dimensions from first video...")
            try:
                first_video = train_data[0][0]
                sample_sequence = self.preprocessor.process_video(first_video)
                num_features = sample_sequence.shape[1]
                self.log(f"Detected {num_features} features per frame")
            except Exception as e:
                self.log(f"Warning: Could not detect features, using default. Error: {e}")
                num_features = None
            
            # Build model
            self.log("Building model...")
            num_classes = len(self.data_loader.label_to_idx)
            self.model = LipReadingModel(
                num_classes=num_classes,
                sequence_length=75,
                num_features=num_features,
                frame_height=100,
                frame_width=100,
                channels=3
            )
            
            if use_attention:
                self.model.build_model_with_attention(num_features=num_features)
            else:
                self.model.build_model(num_features=num_features)
            
            # Compile model
            self.log("Compiling model...")
            self.model.compile_model(learning_rate=learning_rate, device=device)
            
            # Get callbacks
            callbacks = self.model.get_callbacks(
                checkpoint_path='./models/best_model.h5',
                log_dir='./logs/tensorboard'
            )
            
            # Add GUI callback
            gui_callback = GUICallback(self)
            callbacks.append(gui_callback)
            
            # Train
            self.log(f"Training for {epochs} epochs...")
            history = self.model.train(
                train_gen,
                val_gen,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            
            self.log("="*60)
            self.log("TRAINING COMPLETED SUCCESSFULLY")
            self.log("="*60)
            
            messagebox.showinfo("Success", "Training completed successfully!")
            
        except Exception as e:
            self.log(f"\n✗ Training failed: {str(e)}")
            messagebox.showerror("Error", f"Training failed:\n{str(e)}")
        
        finally:
            self.is_training = False
            self.train_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
    
    def pause_training(self):
        """Pause training"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶ Resume")
            self.log("⏸ Training paused")
        else:
            self.pause_btn.config(text="⏸ Pause")
            self.log("▶ Training resumed")
    
    def stop_training(self):
        """Stop training"""
        if messagebox.askyesno("Confirm", "Are you sure you want to stop training?"):
            self.is_training = False
            self.log("⏹ Training stopped by user")
    
    def update_metrics(self, epoch, logs):
        """Update training metrics with real-time information"""
        if logs is None or len(logs) == 0:
            return
        
        self.epochs_completed = epoch + 1
        
        # Calculate epoch time
        if self.epoch_start_time:
            epoch_duration = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_duration)
            epoch_time_str = self._format_time(epoch_duration)
        else:
            epoch_time_str = "--:--"
        
        # Calculate elapsed time
        if self.training_start_time:
            elapsed_time = time.time() - self.training_start_time
            elapsed_time_str = self._format_time(elapsed_time)
        else:
            elapsed_time_str = "00:00:00"
        
        # Calculate estimated time remaining
        if len(self.epoch_times) > 0:
            avg_epoch_time = np.mean(self.epoch_times[-5:])  # Use last 5 epochs for better estimate
            remaining_epochs = self.epochs_var.get() - self.epochs_completed
            estimated_remaining = avg_epoch_time * remaining_epochs
            time_remaining_str = self._format_time(estimated_remaining)
        else:
            time_remaining_str = "--:--:--"
        
        # Extract metrics
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        # Store metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        # Calculate bias and variance
        bias = train_loss - val_loss
        variance = np.std(self.val_losses[-10:]) if len(self.val_losses) >= 10 else 0
        
        self.bias_values.append(bias)
        self.variance_values.append(variance)
        
        # Update plots
        self.ax1.clear()
        self.ax1.plot(self.train_losses, label='Train Loss', color='blue', linewidth=2)
        self.ax1.plot(self.val_losses, label='Val Loss', color='red', linewidth=2)
        self.ax1.set_title('Loss', fontweight='bold')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.clear()
        self.ax2.plot(self.train_accs, label='Train Acc', color='blue', linewidth=2)
        self.ax2.plot(self.val_accs, label='Val Acc', color='red', linewidth=2)
        self.ax2.set_title('Accuracy', fontweight='bold')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
        # Update labels with real-time information
        self.epoch_label.config(text=f"Epoch: {self.epochs_completed}/{self.epochs_var.get()}")
        self.time_remaining_label.config(text=f"Time Remaining: {time_remaining_str}")
        self.elapsed_time_label.config(text=f"Elapsed: {elapsed_time_str}")
        self.epoch_time_label.config(text=f"Epoch Time: {epoch_time_str}")
        
        self.loss_label.config(text=f"Val Loss: {val_loss:.4f}")
        self.acc_label.config(text=f"Val Accuracy: {val_acc*100:.2f}%")
        self.bias_label.config(text=f"Bias: {bias:.4f}")
        self.variance_label.config(text=f"Variance: {variance:.4f}")
        
        # Update progress bar
        progress_percent = (self.epochs_completed / self.epochs_var.get()) * 100
        self.progress_var.set(progress_percent)
        self.progress_label.config(text=f"{progress_percent:.1f}%")
        
        # Log with detailed information
        self.log(f"Epoch {self.epochs_completed}/{self.epochs_var.get()} | "
                f"Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | "
                f"Time: {epoch_time_str} | Remaining: {time_remaining_str}")
    
    def _format_time(self, seconds):
        """Format seconds into HH:MM:SS or MM:SS"""
        if seconds < 0:
            return "--:--:--"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def log(self, message):
        """Log message to console"""
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.root.update_idletasks()


class GUICallback(keras.callbacks.Callback):
    """Custom callback for GUI updates with real-time progress"""
    
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self.batch_count = 0
        self.total_batches = 0
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        self.gui.training_start_time = time.time()
        self.gui.log("Training started...")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        self.gui.epoch_start_time = time.time()
        self.batch_count = 0
        self.gui.log(f"\nStarting Epoch {epoch + 1}/{self.gui.epochs_var.get()}...")
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch - update progress in real-time"""
        self.batch_count += 1
        
        # Update every 5 batches to avoid too frequent updates
        if self.batch_count % 5 == 0 and logs:
            batch_loss = logs.get('loss', 0)
            batch_acc = logs.get('accuracy', 0)
            
            # Update status in console (overwrite last line)
            status_msg = f"  Batch {self.batch_count}: Loss={batch_loss:.4f}, Acc={batch_acc*100:.2f}%"
            
            # For GUI update without creating too many log entries
            if self.batch_count % 10 == 0:
                self.gui.console.insert(tk.END, status_msg + "\n")
                self.gui.console.see(tk.END)
                self.gui.root.update_idletasks()
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        self.gui.update_metrics(epoch, logs)
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        if self.gui.training_start_time:
            total_time = time.time() - self.gui.training_start_time
            self.gui.log(f"\nTraining completed in {self.gui._format_time(total_time)}")


def main():
    root = tk.Tk()
    app = TrainingGUIWithRecording(root)
    root.mainloop()


if __name__ == "__main__":
    main()
