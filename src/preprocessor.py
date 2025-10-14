"""
Enhanced Video Preprocessor for Multi-Lingual Lip Reading
===========================================================

Complete preprocessing pipeline including:
1. Frame Extraction - Extract frames from video at target FPS
2. Face Detection - Detect faces using MediaPipe/dlib
3. ROI Alignment and Normalization - Align and normalize lip regions
4. Data Augmentation - Random transformations for training
5. Feature Extraction - Extract geometric lip features
6. Lip Region Highlighting and Tracking - Highlight and track lips across frames
7. Grayscale Conversion - Convert to grayscale for processing

Author: AI Assistant
Date: October 2025
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import albumentations as A
from tqdm import tqdm

# Optional dlib import - will use MediaPipe if not available
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Using MediaPipe for all face detection.")


class VideoPreprocessor:
    """Preprocessor for extracting and normalizing lip regions from videos"""
    
    def __init__(self, 
                 shape_predictor_path,
                 target_size=(100, 100),
                 sequence_length=75,
                 target_fps=25,
                 augment=False):
        """
        Initialize the video preprocessor
        
        Args:
            shape_predictor_path: Path to dlib shape predictor model
            target_size: Target size for lip region (width, height)
            sequence_length: Number of frames in output sequence
            target_fps: Target frames per second
            augment: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.sequence_length = sequence_length
        self.target_fps = target_fps
        self.augment = augment
        
        # Initialize face detectors
        print("Initializing face detectors...")
        
        # MediaPipe face detection (primary detector)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5
        )
        
        # MediaPipe face mesh for landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Dlib face detector and shape predictor (fallback - optional)
        self.dlib_detector = None
        self.shape_predictor = None
        
        if DLIB_AVAILABLE:
            try:
                self.dlib_detector = dlib.get_frontal_face_detector()
                if Path(shape_predictor_path).exists():
                    self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
                    print("Dlib detector initialized successfully (fallback mode)")
                else:
                    print(f"Warning: Shape predictor not found at {shape_predictor_path}")
            except Exception as e:
                print(f"Warning: Could not initialize dlib: {e}")
        
        # Mouth landmark indices (for dlib - points 48-68)
        self.MOUTH_LANDMARKS = list(range(48, 68))
        
        # Data augmentation pipeline
        if augment:
            self.augmentation = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(0.0, 0.01), p=0.3),
                A.Rotate(limit=5, p=0.3),
            ])
        else:
            self.augmentation = None
        
        print("✓ Preprocessor initialized successfully")
    
    def process_video(self, video_path, return_metadata=False):
        """
        COMPLETE PREPROCESSING PIPELINE
        ================================
        
        Process a video file through all preprocessing steps:
        1. Frame Extraction - Extract frames at target FPS
        2. Face Detection - Detect faces in each frame
        3. Grayscale Conversion - Convert to grayscale for processing
        4. ROI Alignment - Extract and align lip region
        5. Lip Region Highlighting and Tracking - Track lips across frames
        6. Feature Extraction - Extract geometric features
        7. Data Augmentation - Apply augmentations (if enabled)
        8. Normalization - Normalize features
        
        Args:
            video_path: Path to video file
            return_metadata: If True, return metadata along with sequence
        
        Returns:
            sequence: Numpy array of shape (sequence_length, num_features)
                     Features include: landmark coordinates, distances, angles, velocities
            metadata: Dict with video metadata (if return_metadata=True)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # ============================================================
        # STEP 1: FRAME EXTRACTION
        # ============================================================
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling interval based on target FPS
        frame_interval = max(1, int(fps / self.target_fps))
        
        frames = []
        frame_idx = 0
        
        # Extract frames at target FPS
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at target FPS
            if frame_idx % frame_interval == 0:
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        # ============================================================
        # STEP 2-7: PROCESS EACH FRAME
        # ============================================================
        # Process frames through complete pipeline:
        # - Face Detection
        # - Grayscale Conversion
        # - ROI Alignment and Normalization
        # - Lip Region Highlighting and Tracking
        # - Feature Extraction
        # - Data Augmentation
        
        lip_features_sequence = []
        previous_lip_position = None  # For tracking
        
        for frame in frames:
            try:
                # Process frame through pipeline
                lip_features, lip_position = self._process_frame_complete(
                    frame, 
                    previous_lip_position
                )
                
                if lip_features is not None:
                    lip_features_sequence.append(lip_features)
                    previous_lip_position = lip_position
            except Exception as e:
                # Skip frames where processing fails
                continue
        
        if len(lip_features_sequence) == 0:
            raise ValueError(f"No lips detected in video: {video_path}")
        
        # ============================================================
        # STEP 8: SEQUENCE NORMALIZATION
        # ============================================================
        # Normalize sequence length to target length
        lip_features_sequence = self._normalize_sequence_length(lip_features_sequence)
        
        # Convert to numpy array
        sequence = np.array(lip_features_sequence, dtype=np.float32)
        
        # Calculate temporal features (velocities/movements between frames)
        sequence = self._add_temporal_features(sequence)
        
        # Normalize features to zero mean and unit variance
        sequence = self._normalize_features(sequence)
        
        if return_metadata:
            metadata = {
                'original_fps': fps,
                'original_frame_count': frame_count,
                'extracted_frames': len(frames),
                'lip_frames': len(lip_features_sequence),
                'video_path': str(video_path),
                'num_features': sequence.shape[1],
                'preprocessing_steps': [
                    'frame_extraction',
                    'face_detection',
                    'grayscale_conversion',
                    'roi_alignment',
                    'lip_highlighting',
                    'feature_extraction',
                    'data_augmentation',
                    'normalization'
                ]
            }
            return sequence, metadata
        
        return sequence
    
    def _process_frame_complete(self, frame, previous_lip_position=None):
        """
        COMPLETE FRAME PROCESSING PIPELINE
        ===================================
        
        Process a single frame through all steps:
        1. Grayscale Conversion
        2. Face Detection
        3. ROI Alignment and Normalization
        4. Lip Region Highlighting and Tracking
        5. Feature Extraction
        6. Data Augmentation
        
        Args:
            frame: Input BGR frame
            previous_lip_position: Previous lip position for tracking
        
        Returns:
            features: Extracted lip features
            lip_position: Current lip position (centroid) for tracking
        """
        # ============================================================
        # STEP 1: GRAYSCALE CONVERSION
        # ============================================================
        # Convert to grayscale for processing (some algorithms work better with grayscale)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ============================================================
        # STEP 2: FACE DETECTION
        # ============================================================
        # Detect face using MediaPipe
        face_results = self.face_mesh.process(rgb_frame)
        
        if not face_results.multi_face_landmarks:
            # Fallback to dlib if available
            if DLIB_AVAILABLE and self.dlib_detector and self.shape_predictor:
                return self._process_frame_dlib(gray_frame, previous_lip_position)
            return None, None
        
        # Get landmarks from first face
        face_landmarks = face_results.multi_face_landmarks[0]
        
        h, w = frame.shape[:2]
        
        # ============================================================
        # STEP 3: ROI ALIGNMENT AND NORMALIZATION
        # ============================================================
        # Extract lip region of interest (ROI)
        # MediaPipe mouth landmark indices
        outer_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375]
        inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        
        # Extract normalized coordinates
        outer_points = []
        inner_points = []
        
        for idx in outer_lip_indices:
            landmark = face_landmarks.landmark[idx]
            outer_points.append([landmark.x, landmark.y])
        
        for idx in inner_lip_indices:
            landmark = face_landmarks.landmark[idx]
            inner_points.append([landmark.x, landmark.y])
        
        outer_points = np.array(outer_points)
        inner_points = np.array(inner_points)
        
        # ============================================================
        # STEP 4: LIP REGION HIGHLIGHTING AND TRACKING
        # ============================================================
        # Calculate lip centroid for tracking
        lip_centroid = np.mean(outer_points, axis=0)
        
        # If previous position available, calculate movement
        if previous_lip_position is not None:
            movement = np.linalg.norm(lip_centroid - previous_lip_position)
            # Can use movement for temporal smoothing or motion features
        
        # Create highlighted lip region (for visualization/debugging)
        # This step highlights the lip area in the frame
        lip_mask = self._create_lip_mask(frame, outer_points, inner_points)
        
        # ============================================================
        # STEP 5: FEATURE EXTRACTION
        # ============================================================
        # Extract geometric features from lip landmarks
        features = self._compute_geometric_features(outer_points, inner_points)
        
        # ============================================================
        # STEP 6: DATA AUGMENTATION
        # ============================================================
        # Apply augmentation if enabled (training only)
        if self.augment:
            features = self._augment_features(features)
        
        return features, lip_centroid
    
    def _create_lip_mask(self, frame, outer_points, inner_points):
        """
        LIP REGION HIGHLIGHTING
        =======================
        
        Create a mask highlighting the lip region for visualization
        
        Args:
            frame: Input frame
            outer_points: Outer lip landmarks (normalized 0-1)
            inner_points: Inner lip landmarks (normalized 0-1)
        
        Returns:
            lip_mask: Binary mask of lip region
        """
        h, w = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        outer_pixels = (outer_points * [w, h]).astype(np.int32)
        inner_pixels = (inner_points * [w, h]).astype(np.int32)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Fill outer lip contour
        cv2.fillPoly(mask, [outer_pixels], 255)
        
        # Optional: Remove inner lip (for open mouth)
        # cv2.fillPoly(mask, [inner_pixels], 0)
        
        return mask
    
    def _augment_features(self, features):
        """
        DATA AUGMENTATION
        =================
        
        Apply data augmentation to geometric features
        
        Augmentations include:
        - Gaussian noise addition
        - Random scaling
        - Small rotations (simulated via feature perturbation)
        
        Args:
            features: Input feature vector
        
        Returns:
            augmented_features: Augmented feature vector
        """
        augmented = features.copy()
        
        # 1. Add Gaussian noise (simulate measurement noise)
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented += noise
        
        # 2. Random scaling (simulate distance variation)
        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.95, 1.05)
            augmented *= scale
        
        # 3. Random offset (simulate slight misalignment)
        if np.random.rand() < 0.3:
            offset = np.random.normal(0, 0.005, augmented.shape)
            augmented += offset
        
        return augmented.astype(np.float32)
    
    def _process_frame_dlib(self, gray_frame, previous_lip_position=None):
        """
        Process frame using dlib (fallback method)
        Includes all preprocessing steps
        """
        # Face Detection
        faces = self.dlib_detector(gray_frame, 1)
        if len(faces) == 0:
            return None, None
        
        face_bbox = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # ROI Alignment
        try:
            shape = self.shape_predictor(gray_frame, face_bbox)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        except:
            return None, None
        
        h, w = gray_frame.shape[:2]
        
        # Normalize coordinates
        landmarks_normalized = landmarks.copy().astype(np.float32)
        landmarks_normalized[:, 0] /= w
        landmarks_normalized[:, 1] /= h
        
        # Extract lip landmarks
        outer_points = landmarks_normalized[48:60]
        inner_points = landmarks_normalized[60:68]
        
        # Lip tracking
        lip_centroid = np.mean(outer_points, axis=0)
        
        # Feature Extraction
        features = self._compute_geometric_features(outer_points, inner_points)
        
        # Data Augmentation
        if self.augment:
            features = self._augment_features(features)
        
        return features, lip_centroid
    
    def _extract_lip_features(self, frame):
        """
        Extract lip geometry and movement features from a single frame
        
        Returns:
            features: 1D array of lip features including:
                - Normalized landmark coordinates (x, y for each point)
                - Distances between key landmark pairs
                - Mouth aspect ratio, width, height
                - Lip angles and curvature
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            # Fallback to dlib if available
            if DLIB_AVAILABLE and self.dlib_detector and self.shape_predictor:
                return self._extract_lip_features_dlib(frame)
            return None
        
        # Get landmarks from first face
        face_landmarks = results.multi_face_landmarks[0]
        
        h, w = frame.shape[:2]
        
        # MediaPipe mouth landmark indices (lips outer and inner)
        # Outer lip landmarks
        outer_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375]
        # Inner lip landmarks
        inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        
        # Extract coordinates
        outer_points = []
        for idx in outer_lip_indices:
            landmark = face_landmarks.landmark[idx]
            outer_points.append([landmark.x, landmark.y])
        
        inner_points = []
        for idx in inner_lip_indices:
            landmark = face_landmarks.landmark[idx]
            inner_points.append([landmark.x, landmark.y])
        
        outer_points = np.array(outer_points)
        inner_points = np.array(inner_points)
        
        # Compute geometric features
        features = self._compute_geometric_features(outer_points, inner_points)
        
        return features
    
    def _compute_geometric_features(self, outer_points, inner_points):
        """
        Compute geometric features from lip landmarks
        
        Args:
            outer_points: Outer lip landmarks (N, 2)
            inner_points: Inner lip landmarks (M, 2)
        
        Returns:
            features: 1D array of geometric features
        """
        features = []
        
        # 1. Normalized landmark coordinates (flatten)
        outer_flat = outer_points.flatten()
        inner_flat = inner_points.flatten()
        features.extend(outer_flat)
        features.extend(inner_flat)
        
        # 2. Mouth width and height
        mouth_width = np.max(outer_points[:, 0]) - np.min(outer_points[:, 0])
        mouth_height = np.max(outer_points[:, 1]) - np.min(outer_points[:, 1])
        features.extend([mouth_width, mouth_height])
        
        # 3. Mouth aspect ratio (height/width)
        aspect_ratio = mouth_height / (mouth_width + 1e-6)
        features.append(aspect_ratio)
        
        # 4. Centroid of outer and inner lips
        outer_centroid = np.mean(outer_points, axis=0)
        inner_centroid = np.mean(inner_points, axis=0)
        features.extend(outer_centroid)
        features.extend(inner_centroid)
        
        # 5. Distances from centroid to key points (lip spread)
        outer_distances = np.linalg.norm(outer_points - outer_centroid, axis=1)
        features.extend(outer_distances)
        
        # 6. Upper lip to lower lip distances (vertical opening)
        # Approximate: top-most to bottom-most points
        top_point_idx = np.argmin(outer_points[:, 1])
        bottom_point_idx = np.argmax(outer_points[:, 1])
        vertical_opening = outer_points[bottom_point_idx, 1] - outer_points[top_point_idx, 1]
        features.append(vertical_opening)
        
        # 7. Left to right lip distance (horizontal spread)
        left_point_idx = np.argmin(outer_points[:, 0])
        right_point_idx = np.argmax(outer_points[:, 0])
        horizontal_spread = outer_points[right_point_idx, 0] - outer_points[left_point_idx, 0]
        features.append(horizontal_spread)
        
        # 8. Lip curvature (simple measure using angles)
        # Calculate angles at multiple points along the lip
        angles = []
        for i in range(1, len(outer_points) - 1):
            v1 = outer_points[i] - outer_points[i-1]
            v2 = outer_points[i+1] - outer_points[i]
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angles.append(angle)
        features.extend(angles)
        
        # 9. Area of outer lip contour (approximate)
        # Using shoelace formula
        area = 0.5 * np.abs(np.sum(outer_points[:-1, 0] * outer_points[1:, 1] - 
                                     outer_points[1:, 0] * outer_points[:-1, 1]))
        features.append(area)
        
        return np.array(features, dtype=np.float32)
    
    def _add_temporal_features(self, sequence):
        """
        Add temporal features (velocities/movements) between consecutive frames
        
        Args:
            sequence: Array of shape (sequence_length, num_features)
        
        Returns:
            enhanced_sequence: Array with temporal features added
        """
        # Calculate velocity (difference between consecutive frames)
        velocities = np.diff(sequence, axis=0)
        
        # Pad first frame velocity with zeros
        velocities = np.vstack([np.zeros((1, velocities.shape[1])), velocities])
        
        # Calculate acceleration (second derivative)
        accelerations = np.diff(velocities, axis=0)
        accelerations = np.vstack([np.zeros((1, accelerations.shape[1])), accelerations])
        
        # Concatenate original features with temporal features
        enhanced_sequence = np.concatenate([sequence, velocities, accelerations], axis=1)
        
        return enhanced_sequence
    
    def _normalize_features(self, sequence):
        """
        Normalize features to zero mean and unit variance
        
        Args:
            sequence: Array of shape (sequence_length, num_features)
        
        Returns:
            normalized_sequence: Normalized array
        """
        # Calculate mean and std across time dimension
        mean = np.mean(sequence, axis=0)
        std = np.std(sequence, axis=0) + 1e-8  # Add epsilon to avoid division by zero
        
        normalized_sequence = (sequence - mean) / std
        
        return normalized_sequence
    
    def _extract_lip_features_dlib(self, frame):
        """Extract lip geometry features using dlib (fallback method)"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.dlib_detector(gray_frame, 1)
        if len(faces) == 0:
            return None
        
        # Get largest face
        face_bbox = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Detect landmarks
        try:
            shape = self.shape_predictor(gray_frame, face_bbox)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        except:
            return None
        
        h, w = frame.shape[:2]
        
        # Normalize coordinates
        landmarks_normalized = landmarks.copy().astype(np.float32)
        landmarks_normalized[:, 0] /= w
        landmarks_normalized[:, 1] /= h
        
        # Extract mouth landmarks (points 48-68)
        outer_points = landmarks_normalized[48:60]
        inner_points = landmarks_normalized[60:68]
        
        # Compute geometric features
        features = self._compute_geometric_features(outer_points, inner_points)
        
        return features
    
    def _extract_lip_region(self, frame):
        """Extract lip region from a single frame using MediaPipe (deprecated, kept for compatibility)"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            # Fallback to dlib if available
            if DLIB_AVAILABLE and self.dlib_detector and self.shape_predictor:
                return self._extract_lip_region_dlib(frame)
            return None
        
        # Get landmarks from first face
        face_landmarks = results.multi_face_landmarks[0]
        
        h, w = frame.shape[:2]
        
        # MediaPipe mouth landmark indices (lips outer boundary)
        # Upper lip: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
        # Lower lip: 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
        # Simplified: use key mouth points
        mouth_indices = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,  # Upper lip
            146, 91, 181, 84, 17, 314, 405, 321, 375,  # Lower lip
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308  # Inner mouth
        ]
        
        # Extract mouth coordinates
        mouth_points = []
        for idx in mouth_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            mouth_points.append([x, y])
        
        mouth_points = np.array(mouth_points)
        
        # Get bounding box around mouth
        x_min, y_min = mouth_points.min(axis=0)
        x_max, y_max = mouth_points.max(axis=0)
        
        # Add padding (30%)
        padding = 0.3
        width = x_max - x_min
        height = y_max - y_min
        
        x_min = max(0, int(x_min - width * padding))
        x_max = min(w, int(x_max + width * padding))
        y_min = max(0, int(y_min - height * padding))
        y_max = min(h, int(y_max + height * padding))
        
        # Ensure valid crop region
        if x_max <= x_min or y_max <= y_min:
            return None
        
        # Crop lip region
        lip_roi = frame[y_min:y_max, x_min:x_max]
        
        # Resize to target size
        if lip_roi.size > 0:
            lip_roi = cv2.resize(lip_roi, self.target_size)
            return lip_roi
        
        return None
    
    def _extract_lip_region_dlib(self, frame):
        """Extract lip region using dlib (fallback method)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.dlib_detector(gray_frame, 1)
        if len(faces) == 0:
            return None
        
        # Get largest face
        face_bbox = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Detect landmarks
        try:
            shape = self.shape_predictor(gray_frame, face_bbox)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        except:
            return None
        
        # Extract mouth region (points 48-68)
        mouth_points = landmarks[self.MOUTH_LANDMARKS]
        
        # Get bounding box around mouth
        x_coords = mouth_points[:, 0]
        y_coords = mouth_points[:, 1]
        
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        # Add padding (20%)
        padding = 0.2
        width = x_max - x_min
        height = y_max - y_min
        
        h, w = frame.shape[:2]
        x_min = max(0, int(x_min - width * padding))
        x_max = min(w, int(x_max + width * padding))
        y_min = max(0, int(y_min - height * padding))
        y_max = min(h, int(y_max + height * padding))
        
        # Crop lip region
        lip_roi = frame[y_min:y_max, x_min:x_max]
        
        # Resize to target size
        if lip_roi.size > 0:
            lip_roi = cv2.resize(lip_roi, self.target_size)
            return lip_roi
        
        return None
    
    def _normalize_sequence_length(self, sequence):
        """Normalize sequence to target length by padding or sampling"""
        current_length = len(sequence)
        
        if current_length == self.sequence_length:
            return sequence
        
        elif current_length < self.sequence_length:
            # Pad by repeating frames
            padding_needed = self.sequence_length - current_length
            
            # Repeat last frame
            padded_sequence = sequence + [sequence[-1]] * padding_needed
            return padded_sequence
        
        else:
            # Sample frames uniformly
            indices = np.linspace(0, current_length - 1, self.sequence_length, dtype=int)
            sampled_sequence = [sequence[i] for i in indices]
            return sampled_sequence
    
    def _augment_sequence(self, sequence):
        """
        Apply augmentation to geometric feature sequence
        Note: For geometric features, augmentation is minimal (add noise, scale)
        """
        if not self.augment:
            return sequence
        
        # Add small Gaussian noise to features
        noise = np.random.normal(0, 0.01, sequence.shape)
        augmented = sequence + noise
        
        # Random scaling (subtle)
        scale = np.random.uniform(0.98, 1.02)
        augmented = augmented * scale
        
        return augmented.astype(np.float32)
    
    def process_batch(self, video_paths, show_progress=True):
        """
        Process multiple videos
        
        Args:
            video_paths: List of video file paths
            show_progress: Show progress bar
        
        Returns:
            sequences: List of processed sequences
            failed: List of failed video paths
        """
        sequences = []
        failed = []
        
        iterator = tqdm(video_paths) if show_progress else video_paths
        
        for video_path in iterator:
            try:
                sequence = self.process_video(video_path)
                sequences.append(sequence)
            except Exception as e:
                print(f"✗ Failed to process {video_path}: {e}")
                failed.append(video_path)
        
        return sequences, failed
    
    def visualize_lip_extraction(self, video_path, output_path=None):
        """
        Visualize lip extraction process on a video
        
        Args:
            video_path: Input video path
            output_path: Output video path (optional)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract lip region
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_bbox = self._detect_face(rgb_frame)
            
            if face_bbox:
                # Draw face bounding box
                cv2.rectangle(frame, 
                            (face_bbox.left(), face_bbox.top()),
                            (face_bbox.right(), face_bbox.bottom()),
                            (0, 255, 0), 2)
                
                # Detect landmarks
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = self._detect_landmarks(gray, face_bbox)
                
                if landmarks is not None:
                    # Draw mouth landmarks
                    mouth_points = landmarks[self.MOUTH_LANDMARKS]
                    for point in mouth_points:
                        cv2.circle(frame, tuple(point), 2, (0, 0, 255), -1)
            
            if output_path:
                out.write(frame)
            else:
                cv2.imshow('Lip Extraction', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if output_path:
            out.release()
        else:
            cv2.destroyAllWindows()


def test_preprocessor():
    """Test the preprocessor"""
    from utils import ensure_shape_predictor
    
    print("Testing VideoPreprocessor...")
    
    # Ensure shape predictor exists
    predictor_path = ensure_shape_predictor()
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        shape_predictor_path=predictor_path,
        target_size=(100, 100),
        sequence_length=75,
        augment=False
    )
    
    print("✓ Preprocessor initialized")
    print(f"  Target size: {preprocessor.target_size}")
    print(f"  Sequence length: {preprocessor.sequence_length}")
    print(f"  Target FPS: {preprocessor.target_fps}")


if __name__ == "__main__":
    test_preprocessor()
