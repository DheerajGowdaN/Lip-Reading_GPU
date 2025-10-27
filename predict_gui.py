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
        self.root.geometry("1280x720")
        
        # Initialize variables
        self.config = load_config()
        self.model = None
        self.preprocessor = None
        self.class_mapping = None
        self.idx_to_label = {}
        
        # Multi-model system for automatic language detection
        self.models = {}  # Dictionary: {'hindi': model, 'kannada': model}
        self.model_mappings = {}  # Dictionary: {'hindi': idx_to_label, 'kannada': idx_to_label}
        self.detected_language = "Unknown"  # Currently detected language
        self.auto_detect_enabled = False  # Whether auto-detection is active
        
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
        self.stable_top_predictions = []  # Only updated when prediction stabilizes
        
        # Prediction stabilization
        self.prediction_history = deque(maxlen=15)  # Last 15 predictions for maximum stability
        self.stable_prediction = "No prediction"
        self.stable_prediction_display = "No prediction"
        self.stable_confidence = 0.0
        self.prediction_change_threshold = 0.80  # Very high threshold (80%) to change prediction
        self.stability_count = 0  # Consecutive same predictions
        self.stability_required = 10  # Need 10 consecutive confirmations (~2-3 seconds) predictions to change
        
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
        
        # Main container with reduced padding
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)  # Video gets more space
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Video feed
        self.setup_video_panel(main_frame)
        
        # Right panel - Controls and results
        self.setup_control_panel(main_frame)
    
    def setup_video_panel(self, parent):
        """Setup video display panel"""
        frame = ttk.LabelFrame(parent, text="Video Feed", padding="5")
        frame.grid(row=0, column=0, padx=3, pady=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        
        # Video display
        self.video_label = ttk.Label(frame, text="Camera not started", 
                                     background="black", foreground="white",
                                     font=("Arial", 12))
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # FPS and status - more compact
        status_frame = ttk.Frame(frame)
        status_frame.grid(row=1, column=0, pady=3, sticky=(tk.W, tk.E))
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0", font=("Arial", 9))
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready", 
                                      font=("Arial", 9), foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=5)
    
    def setup_control_panel(self, parent):
        """Setup control and results panel"""
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=1, padx=3, pady=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame.rowconfigure(2, weight=1)  # Prediction results expands
        
        
        # Model selection - ultra compact
        model_frame = ttk.LabelFrame(frame, text="Model", padding="2")
        model_frame.grid(row=0, column=0, pady=1, sticky=(tk.W, tk.E))
        
        # Auto-detection checkbox - shorter text
        self.auto_detect_var = tk.BooleanVar(value=False)
        auto_check = ttk.Checkbutton(model_frame, text="🌐 Auto Detection",
                                     variable=self.auto_detect_var,
                                     command=self.toggle_auto_detection)
        auto_check.grid(row=0, column=0, columnspan=3, pady=0, sticky=tk.W)
        
        # Model path - compact horizontal layout
        path_frame = ttk.Frame(model_frame)
        path_frame.grid(row=1, column=0, columnspan=3, pady=1, sticky=(tk.W, tk.E))
        
        self.model_path_var = tk.StringVar(value="./models/best_model_hindi.h5")
        model_entry = ttk.Entry(path_frame, textvariable=self.model_path_var, width=20, font=("Arial", 8))
        model_entry.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(path_frame, text="Browse", command=self.browse_model, width=7)
        browse_btn.pack(side=tk.LEFT, padx=1)
        
        # Model status info - minimal
        self.model_info_var = tk.StringVar(value="No model loaded")
        info_label = ttk.Label(model_frame, textvariable=self.model_info_var, 
                              foreground="blue", wraplength=320, font=("Arial", 7))
        info_label.grid(row=2, column=0, columnspan=3, pady=0)
        
        # Load all models button (for auto-detection)
        self.load_all_btn = ttk.Button(model_frame, text="🔄 Load All Language Models", 
                                       command=self.load_all_models, state=tk.DISABLED)
        self.load_all_btn.grid(row=3, column=0, columnspan=3, pady=1)
        
        # Camera controls - ultra compact
        camera_frame = ttk.LabelFrame(frame, text="Camera", padding="2")
        camera_frame.grid(row=1, column=0, pady=1, sticky=(tk.W, tk.E))
        
        # Buttons in horizontal layout
        btn_frame = ttk.Frame(camera_frame)
        btn_frame.pack(pady=1, fill=tk.X)
        
        self.start_camera_btn = ttk.Button(btn_frame, text="Start", 
                                          command=self.start_camera, width=12)
        self.start_camera_btn.pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)
        
        self.stop_camera_btn = ttk.Button(btn_frame, text="Stop", 
                                         command=self.stop_camera, width=12,
                                         state=tk.DISABLED)
        self.stop_camera_btn.pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)
        
        # Settings in compact layout
        settings_frame = ttk.Frame(camera_frame)
        settings_frame.pack(pady=1, fill=tk.X)
        
        ttk.Label(settings_frame, text="Camera ID:", font=("Arial", 8)).pack(side=tk.LEFT, padx=1)
        self.camera_id_var = tk.IntVar(value=0)
        ttk.Spinbox(settings_frame, from_=0, to=5, textvariable=self.camera_id_var, 
                   width=5).pack(side=tk.LEFT, padx=1)
        
        # Confidence threshold - more compact
        thresh_frame = ttk.Frame(camera_frame)
        thresh_frame.pack(pady=1, fill=tk.X)
        
        ttk.Label(thresh_frame, text="Confidence:", font=("Arial", 8)).pack(side=tk.LEFT, padx=1)
        self.confidence_threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = ttk.Scale(thresh_frame, from_=0.0, to=1.0, 
                                   variable=self.confidence_threshold_var,
                                   orient=tk.HORIZONTAL)
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        self.threshold_label = ttk.Label(thresh_frame, text="0.50", font=("Arial", 8), width=4)
        self.threshold_label.pack(side=tk.LEFT, padx=1)
        
        # Update threshold label
        def update_threshold_label(val):
            self.threshold_label.config(text=f"{float(val):.2f}")
        
        threshold_scale.config(command=update_threshold_label)
        
        # Prediction results - optimized to use available space
        results_frame = ttk.LabelFrame(frame, text="Prediction", padding="2")
        results_frame.grid(row=2, column=0, pady=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Current prediction
        self.prediction_label = ttk.Label(results_frame, 
                                         text="No prediction",
                                         font=("Arial", 14, "bold"),
                                         foreground="green",
                                         wraplength=320)
        self.prediction_label.pack(pady=2)
        
        # Confidence
        self.confidence_label = ttk.Label(results_frame, 
                                         text="Confidence: 0.0%",
                                         font=("Arial", 9))
        self.confidence_label.pack(pady=1)
        
        # Detected language (for auto-detection mode)
        self.language_label = ttk.Label(results_frame,
                                       text="",
                                       font=("Arial", 8, "italic"),
                                       foreground="blue")
        self.language_label.pack(pady=0)
        
        # Top 3 predictions
        ttk.Label(results_frame, text="Top 3 Predictions:", 
                 font=("Arial", 8, "bold")).pack(pady=1)
        
        self.top_predictions_text = tk.Text(results_frame, height=3, width=38,
                                           font=("Arial", 8), wrap=tk.WORD)
        self.top_predictions_text.pack(pady=1, fill=tk.BOTH, expand=True)
        self.top_predictions_text.config(state=tk.DISABLED)
        
        # Recording controls - minimal
        recording_frame = ttk.LabelFrame(frame, text="Recording", padding="2")
        recording_frame.grid(row=3, column=0, pady=1, sticky=(tk.W, tk.E))
        
        self.record_btn = ttk.Button(recording_frame, text="Record Video", 
                                    command=self.toggle_recording)
        self.record_btn.pack(pady=1, fill=tk.X)
        
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
    
    def toggle_auto_detection(self):
        """Toggle automatic language detection"""
        if self.auto_detect_var.get():
            self.auto_detect_enabled = True
            self.load_all_btn.config(state=tk.NORMAL)
            print("\n✓ Auto-detection mode enabled")
            print("  Load all language models using the button below\n")
        else:
            self.auto_detect_enabled = False
            self.load_all_btn.config(state=tk.DISABLED)
            self.detected_language = "Unknown"
            print("\n✓ Auto-detection mode disabled")
            print("  Use single model mode\n")
    
    def load_all_models(self):
        """Load all available language models for auto-detection"""
        try:
            models_dir = Path('./models')
            if not models_dir.exists():
                messagebox.showerror("Error", "Models directory not found!")
                return
            
            # Find all language-specific model files
            model_files = list(models_dir.glob('best_model_*.h5'))
            
            if not model_files:
                messagebox.showwarning("Warning", 
                                     "No language-specific models found!\n\n"
                                     "Looking for: best_model_hindi.h5, best_model_kannada.h5, etc.\n"
                                     "Train separate models first.")
                return
            
            self.models = {}
            self.model_mappings = {}
            loaded_languages = []
            
            for model_file in model_files:
                # Extract language from filename (e.g., best_model_hindi.h5 -> hindi)
                lang = model_file.stem.replace('best_model_', '')
                
                # Skip multi-language models for now, load at the end if no language-specific models
                if lang == 'multi':
                    continue
                
                try:
                    # Load class mapping
                    mapping_file = models_dir / f'class_mapping_{lang}.json'
                    if not mapping_file.exists():
                        print(f"⚠ Skipping {lang}: class_mapping_{lang}.json not found")
                        continue
                    
                    import json
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        mapping = json.load(f)
                    
                    idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}
                    num_classes = len(idx_to_label)
                    
                    # Initialize model
                    model = LipReadingModel(
                        num_classes=num_classes,
                        sequence_length=self.config['model']['sequence_length'],
                        frame_height=self.config['model']['frame_height'],
                        frame_width=self.config['model']['frame_width']
                    )
                    
                    model.load_model(str(model_file))
                    
                    # Store model and mapping
                    self.models[lang] = model
                    self.model_mappings[lang] = idx_to_label
                    loaded_languages.append(f"{lang.upper()} ({num_classes} classes)")
                    
                    print(f"✓ Loaded {lang.upper()} model: {num_classes} classes")
                
                except Exception as e:
                    print(f"✗ Error loading {lang} model: {e}")
            
            # If no language-specific models found, try loading multi-language model
            if not self.models:
                print("ℹ No language-specific models found. Attempting to load multi-language model...")
                multi_model = models_dir / 'best_model_multi.h5'
                multi_mapping = models_dir / 'class_mapping_multi.json'
                
                if multi_model.exists() and multi_mapping.exists():
                    try:
                        import json
                        with open(multi_mapping, 'r', encoding='utf-8') as f:
                            mapping = json.load(f)
                        
                        idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}
                        num_classes = len(idx_to_label)
                        
                        # Initialize model
                        model = LipReadingModel(
                            num_classes=num_classes,
                            sequence_length=self.config['model']['sequence_length'],
                            frame_height=self.config['model']['frame_height'],
                            frame_width=self.config['model']['frame_width']
                        )
                        
                        model.load_model(str(multi_model))
                        
                        # Store as 'multi' language
                        self.models['multi'] = model
                        self.model_mappings['multi'] = idx_to_label
                        
                        # Detect languages from mapping
                        languages = mapping.get('languages', ['multi'])
                        lang_display = '+'.join([l.upper() for l in languages])
                        loaded_languages.append(f"MULTI ({lang_display}, {num_classes} classes)")
                        
                        print(f"✓ Loaded MULTI-LANGUAGE model: {num_classes} classes")
                    
                    except Exception as e:
                        print(f"✗ Error loading multi-language model: {e}")
            
            if not self.models:
                messagebox.showerror("Error", 
                                   "No models could be loaded!\n\n"
                                   "Make sure you have:\n"
                                   "• best_model_hindi.h5 + class_mapping_hindi.json\n"
                                   "• best_model_kannada.h5 + class_mapping_kannada.json\n"
                                   "OR\n"
                                   "• best_model_multi.h5 + class_mapping_multi.json\n\n"
                                   "Train models first using train_gui_with_recording.py")
                return
            
            # Update GUI
            info_text = f"✓ Auto-Detection Active\n" \
                       f"Loaded: {', '.join(loaded_languages)}\n" \
                       f"Models: {len(self.models)}"
            self.model_info_var.set(info_text)
            
            # Clear single model reference
            self.model = None
            self.idx_to_label = {}
            
            messagebox.showinfo("Success", 
                              f"Loaded {len(self.models)} language models:\n" + 
                              "\n".join(loaded_languages) +
                              "\n\nSystem will automatically detect language!")
            
            print(f"\n✓ Auto-detection ready with {len(self.models)} languages\n")
        
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
    
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
            
            # Load class mapping - try to find matching language-specific file first
            model_name = model_path.stem  # e.g., "best_model_hindi" or "best_model"
            
            # Try language-specific mapping first (e.g., class_mapping_hindi.json)
            if '_' in model_name:
                lang_suffix = model_name.split('_', 2)[-1]  # Extract language part
                mapping_path = model_path.parent / f'class_mapping_{lang_suffix}.json'
            else:
                mapping_path = None
            
            # Fall back to generic class_mapping.json if language-specific not found
            if mapping_path is None or not mapping_path.exists():
                mapping_path = model_path.parent / 'class_mapping.json'
            
            if not mapping_path.exists():
                messagebox.showerror("Error", 
                                   f"Class mapping not found!\n"
                                   f"Looking for: {mapping_path.name}\n"
                                   f"Please ensure it's in the models directory.")
                return
            
            import json
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            
            self.idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}
            num_classes = len(self.idx_to_label)
            
            # Get language info if available
            languages = mapping.get('languages', ['unknown'])
            lang_info = ', '.join(languages).upper() if languages else 'Unknown'
            
            print(f"✓ Loaded class mapping: {mapping_path.name}")
            print(f"  Languages: {lang_info}")
            
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
            info_text = f"✓ Model loaded\nLanguages: {lang_info}\nClasses: {num_classes}\n" \
                       f"Parameters: {info['total_parameters']:,}\n" \
                       f"Size: {info['model_size_mb']:.2f} MB"
            self.model_info_var.set(info_text)
            
            print(f"✓ Model loaded successfully")
            print(f"  Classes: {num_classes}")
            print(f"  Parameters: {info['total_parameters']:,}\n")
            
            messagebox.showinfo("Success", f"Model loaded successfully!\nLanguages: {lang_info}")
        
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def start_camera(self):
        """Start camera capture"""
        # Check if models are loaded (single or multiple)
        if self.auto_detect_enabled:
            if not self.models:
                messagebox.showwarning("Warning", "Please load language models first!\n"
                                     "Click 'Load All Language Models' button.")
                return
        else:
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
            
            # Prediction text removed from camera interface
            # All prediction results are shown in the GUI panel on the right side
            
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
        # Check if we have buffer ready
        if len(self.lip_buffer) < 75:
            return
        
        # Route to appropriate prediction method
        if self.auto_detect_enabled and self.models:
            self._predict_with_auto_detection()
        elif self.model is not None:
            self._predict_with_single_model()
    
    def _predict_with_auto_detection(self):
        """Run all models and pick the best prediction"""
        try:
            # Prepare sequence once (will be reused for all models)
            sequence = np.array(list(self.lip_buffer))
            
            # Apply preprocessing (smoothing, outlier removal)
            try:
                window_length = min(11, len(sequence) if len(sequence) % 2 == 1 else len(sequence) - 1)
                if window_length >= 5:
                    sequence = savgol_filter(sequence, window_length=window_length, 
                                            polyorder=3, axis=0)
            except Exception as e:
                pass
            
            # Remove outliers
            feature_magnitudes = np.linalg.norm(sequence, axis=1)
            z_scores = np.abs((feature_magnitudes - np.mean(feature_magnitudes)) / 
                            (np.std(feature_magnitudes) + 1e-8))
            outlier_mask = z_scores > 3.0
            if np.any(outlier_mask):
                for i in np.where(outlier_mask)[0]:
                    if i > 0 and i < len(sequence) - 1:
                        sequence[i] = (sequence[i-1] + sequence[i+1]) / 2
            
            # Add temporal features and normalize
            sequence = self.preprocessor._add_temporal_features(sequence)
            sequence = self.preprocessor._normalize_features(sequence)
            sequence = np.expand_dims(sequence, axis=0)
            
            # Run ALL models and collect results
            all_predictions = []
            
            print(f"\n[AUTO-DETECT] Running {len(self.models)} models...")
            
            for lang, model in self.models.items():
                try:
                    predictions = model.predict(sequence)
                    if not isinstance(predictions, np.ndarray):
                        predictions = np.array(predictions)
                    
                    # Get best prediction from this model
                    top_idx = np.argmax(predictions)
                    confidence = float(predictions[top_idx])
                    word_label = self.model_mappings[lang][top_idx]
                    
                    # Extract word (remove language prefix if present)
                    if '_' in word_label:
                        word = word_label.split('_', 1)[1]
                    else:
                        word = word_label
                    
                    all_predictions.append({
                        'language': lang,
                        'word': word,
                        'word_label': word_label,
                        'confidence': confidence,
                        'all_probs': predictions
                    })
                    
                    print(f"  {lang.upper():8s}: {word} ({confidence:.1%})")
                
                except Exception as e:
                    print(f"  {lang.upper():8s}: Error - {e}")
            
            if not all_predictions:
                return
            
            # Pick the prediction with HIGHEST confidence across ALL models
            best = max(all_predictions, key=lambda x: x['confidence'])
            
            self.detected_language = best['language']
            self.prediction_confidence = best['confidence']
            
            # Format for display
            current_pred = f"{best['word']} [{best['language'].upper()}]"
            
            print(f"[AUTO-DETECT] ✓ WINNER: {current_pred} at {self.prediction_confidence:.1%}\n")
            
            # Update top predictions from best model
            top_indices = np.argsort(best['all_probs'])[-3:][::-1]
            self.top_predictions = [
                (self.model_mappings[best['language']][idx], 
                 float(best['all_probs'][idx]))
                for idx in top_indices
            ]
            
            # Add to prediction history
            if self.prediction_confidence >= self.confidence_threshold_var.get():
                self.prediction_history.append({
                    'prediction': current_pred,
                    'word': best['word'],
                    'confidence': self.prediction_confidence
                })
            else:
                self.prediction_history.append({
                    'prediction': 'Low confidence',
                    'word': None,
                    'confidence': self.prediction_confidence
                })
            
            # Simple stabilization for auto-detection
            if len(self.prediction_history) >= 5:
                recent = list(self.prediction_history)[-5:]
                pred_counts = {}
                for p in recent:
                    pred = p['prediction']
                    pred_counts[pred] = pred_counts.get(pred, 0) + 1
                
                most_common = max(pred_counts.items(), key=lambda x: x[1])
                if most_common[1] >= 3:  # At least 3 out of 5
                    self.stable_prediction = most_common[0]
                    self.stable_confidence = self.prediction_confidence
                    self.stable_top_predictions = self.top_predictions.copy()
                    
                    # Romanize for display
                    if '[' in most_common[0]:
                        word_part = most_common[0].split('[')[0].strip()
                        lang_part = most_common[0].split('[')[1].strip(']')
                        word_rom = {
                            'ನಮಸ್ಕಾರ': 'Namaskara',
                            'ರಾಮ': 'Rama',
                            'तुम्हारा': 'Tumhara',
                            'पिता': 'Pitha',
                            'hello': 'Hello',
                        }.get(word_part, word_part)
                        self.stable_prediction_display = f"{word_rom} [{lang_part}]"
                    else:
                        self.stable_prediction_display = most_common[0]
            
            self.current_prediction = self.stable_prediction
            self.current_prediction_display = self.stable_prediction_display
        
        except Exception as e:
            print(f"[ERROR] Auto-detection failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _predict_with_single_model(self):
        """Original single model prediction"""
        try:
            sequence = np.array(list(self.lip_buffer))
            
            # Preprocessing
            try:
                window_length = min(11, len(sequence) if len(sequence) % 2 == 1 else len(sequence) - 1)
                if window_length >= 5:
                    sequence = savgol_filter(sequence, window_length=window_length, polyorder=3, axis=0)
            except Exception as e:
                pass
            
            # Remove outliers
            feature_magnitudes = np.linalg.norm(sequence, axis=1)
            z_scores = np.abs((feature_magnitudes - np.mean(feature_magnitudes)) / 
                            (np.std(feature_magnitudes) + 1e-8))
            outlier_mask = z_scores > 3.0
            if np.any(outlier_mask):
                for i in np.where(outlier_mask)[0]:
                    if i > 0 and i < len(sequence) - 1:
                        sequence[i] = (sequence[i-1] + sequence[i+1]) / 2
                    elif i == 0 and len(sequence) > 1:
                        sequence[i] = sequence[i+1]
                    elif i == len(sequence) - 1 and len(sequence) > 1:
                        sequence[i] = sequence[i-1]
            
            # Add temporal features and normalize
            sequence = self.preprocessor._add_temporal_features(sequence)
            sequence = self.preprocessor._normalize_features(sequence)
            sequence = np.expand_dims(sequence, axis=0)
            
            # Predict
            predictions = self.model.predict(sequence)
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            
            # Get top predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_idx = top_indices[0]
            self.prediction_confidence = float(predictions[top_idx])
            
            # Store top 3
            self.top_predictions = [
                (self.idx_to_label[idx], float(predictions[idx]))
                for idx in top_indices
            ]
            
            # Format prediction
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
            if len(self.prediction_history) >= 10:
                # Get most recent predictions
                recent_predictions = list(self.prediction_history)[-10:]
                
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
                
                if frequency >= 8 and avg_confidence >= self.prediction_change_threshold:
                    should_update = True
                    should_update = True
                
                if should_update and frequent_pred != "Low confidence":
                    # Check if it's actually different from current stable prediction
                    if frequent_pred != self.stable_prediction:
                        self.stability_count += 1
                        if self.stability_count >= self.stability_required:
                            # Update stable prediction
                            self.stable_prediction = frequent_pred
                            self.stable_confidence = avg_confidence
                            self.stability_count = 0
                            
                            # Update stable top predictions (only when prediction stabilizes)
                            self.stable_top_predictions = self.top_predictions.copy()
                            
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
                
                # Get the current size of the video label
                label_width = self.video_label.winfo_width()
                label_height = self.video_label.winfo_height()
                
                # Only resize if label has been drawn (width > 1)
                if label_width > 1 and label_height > 1:
                    # Calculate aspect ratios
                    img_aspect = img.width / img.height
                    label_aspect = label_width / label_height
                    
                    # Fit to fill the label while maintaining aspect ratio
                    if img_aspect > label_aspect:
                        # Image is wider - fit to height
                        new_height = label_height
                        new_width = int(new_height * img_aspect)
                    else:
                        # Image is taller - fit to width
                        new_width = label_width
                        new_height = int(new_width / img_aspect)
                    
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    # Default size before label is drawn
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image=img)
                
                # Update label
                self.video_label.config(image=photo, text='')
                self.video_label.image = photo  # Keep reference
            
            # Update prediction display - ONLY show stable predictions
            self.prediction_label.config(text=self.stable_prediction)
            self.confidence_label.config(
                text=f"Confidence: {self.stable_confidence*100:.1f}%"
            )
            
            # Update detected language (for auto-detection mode)
            if self.auto_detect_enabled and self.detected_language != "Unknown":
                self.language_label.config(
                    text=f"🌐 Detected: {self.detected_language.upper()}"
                )
            else:
                self.language_label.config(text="")
            
            # Update top predictions - ONLY show stable predictions
            if self.stable_top_predictions:
                self.top_predictions_text.config(state=tk.NORMAL)
                self.top_predictions_text.delete(1.0, tk.END)
                
                for i, (label, conf) in enumerate(self.stable_top_predictions, 1):
                    # Handle both formats: "language_word" and "word"
                    if '_' in label:
                        language, word = label.split('_', 1)
                        text = f"{i}. {word} ({language})\n   {conf*100:.1f}%\n"
                    else:
                        text = f"{i}. {label}\n   {conf*100:.1f}%\n"
                    
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
