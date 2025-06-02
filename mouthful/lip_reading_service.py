"""
Lip Reading Module for Mouthful
Implements visual speech recognition from lip movements
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from collections import deque
import json

class TemporalCNN(nn.Module):
    """Temporal Convolutional Network for lip reading"""
    
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 512):
        super(TemporalCNN, self).__init__()
        
        # Visual frontend - CNN for processing lip images
        self.visual_frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
        # Temporal modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, channels, time, height, width)
        batch_size, channels, time_steps, height, width = x.size()
        
        # Visual feature extraction
        x = self.visual_frontend(x)  # (batch_size, 512, time, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch_size, 512, time)
        
        # Temporal modeling
        x = self.temporal_conv(x)  # (batch_size, hidden_dim, time)
        
        # Global average pooling over time
        x = torch.mean(x, dim=2)  # (batch_size, hidden_dim)
        
        # Classification
        x = self.classifier(x)  # (batch_size, vocab_size)
        
        return x

class LipReadingService:
    """Service for lip reading from video sequences"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = {}
        self.reverse_vocab = {}
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        
        # Load model and vocabulary
        self._load_model()
        self._load_vocabulary()
        
        # Processing parameters
        self.mouth_roi_size = (112, 112)
        self.sequence_length = 16  # frames
        self.frame_buffer = deque(maxlen=self.sequence_length)
    
    def _load_model(self):
        """Load the pre-trained lip reading model"""
        model_path = self.config.get("model_path", "models/lip_reading/model.pth")
        
        if os.path.exists(model_path):
            # Load saved model
            checkpoint = torch.load(model_path, map_location=self.device)
            vocab_size = checkpoint.get("vocab_size", 1000)
            hidden_dim = checkpoint.get("hidden_dim", 512)
            
            self.model = TemporalCNN(vocab_size=vocab_size, hidden_dim=hidden_dim)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Loaded lip reading model from {model_path}")
        else:
            # Initialize with pre-trained weights (simplified for demo)
            print(f"Model file {model_path} not found. Initializing new model...")
            self.model = TemporalCNN(vocab_size=1000, hidden_dim=512)
            self.model.to(self.device)
            self.model.eval()
            
            # Create model directory and save initial model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "vocab_size": 1000,
                "hidden_dim": 512
            }, model_path)
    
    def _load_vocabulary(self):
        """Load vocabulary mapping"""
        vocab_path = "models/lip_reading/vocab.json"
        
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
        else:
            # Create basic vocabulary for demo
            words = [
                "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
                "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
                "at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
                "but", "not", "what", "all", "were", "we", "when", "your", "can", "said",
                "there", "each", "which", "she", "do", "how", "their", "if", "will", "up"
            ]
            
            self.vocab = {word: idx for idx, word in enumerate(words)}
            self.vocab["<UNK>"] = len(words)
            self.vocab["<PAD>"] = len(words) + 1
            
            # Save vocabulary
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            with open(vocab_path, 'w') as f:
                json.dump(self.vocab, f, indent=2)
        
        # Create reverse vocabulary
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
    
    def extract_mouth_roi(self, image: np.ndarray, landmarks: Dict) -> Optional[np.ndarray]:
        """Extract mouth region of interest from face landmarks"""
        if "bottom_lip" not in landmarks or "top_lip" not in landmarks:
            return None
        
        # Get mouth landmarks
        mouth_points = landmarks["bottom_lip"] + landmarks["top_lip"]
        mouth_points = np.array(mouth_points)
        
        if len(mouth_points) == 0:
            return None
        
        # Calculate bounding box around mouth
        x_min, y_min = np.min(mouth_points, axis=0)
        x_max, y_max = np.max(mouth_points, axis=0)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Extract mouth region
        mouth_roi = image[y_min:y_max, x_min:x_max]
        
        if mouth_roi.size == 0:
            return None
        
        # Resize to standard size
        mouth_roi = cv2.resize(mouth_roi, self.mouth_roi_size)
        
        # Convert to grayscale if needed
        if len(mouth_roi.shape) == 3:
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        mouth_roi = mouth_roi.astype(np.float32) / 255.0
        
        return mouth_roi
    
    def preprocess_frame(self, mouth_roi: np.ndarray) -> np.ndarray:
        """Preprocess mouth ROI for model input"""
        # Ensure proper shape and normalization
        if mouth_roi is None:
            return None
        
        # Normalize to [-1, 1]
        mouth_roi = (mouth_roi - 0.5) * 2.0
        
        return mouth_roi
    
    def predict_sequence(self, mouth_sequence: List[np.ndarray]) -> Tuple[str, float]:
        """Predict words from a sequence of mouth ROIs"""
        if len(mouth_sequence) < self.sequence_length:
            # Pad sequence if too short
            while len(mouth_sequence) < self.sequence_length:
                mouth_sequence.append(np.zeros(self.mouth_roi_size, dtype=np.float32))
        
        # Take last sequence_length frames
        mouth_sequence = mouth_sequence[-self.sequence_length:]
        
        # Stack frames into tensor
        sequence_tensor = np.stack(mouth_sequence, axis=0)  # (time, height, width)
        sequence_tensor = np.expand_dims(sequence_tensor, axis=0)  # Add channel dim
        sequence_tensor = np.expand_dims(sequence_tensor, axis=0)  # Add batch dim
        
        # Convert to torch tensor
        input_tensor = torch.from_numpy(sequence_tensor).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
        
        # Convert prediction to word
        predicted_word = self.reverse_vocab.get(predicted_idx, "<UNK>")
        
        return predicted_word, confidence
    
    def process_frame(self, image: np.ndarray, landmarks: Dict) -> Optional[Tuple[str, float]]:
        """Process a single frame and return prediction if sequence is complete"""
        # Extract mouth ROI
        mouth_roi = self.extract_mouth_roi(image, landmarks)
        
        if mouth_roi is None:
            return None
        
        # Preprocess frame
        processed_roi = self.preprocess_frame(mouth_roi)
        
        if processed_roi is None:
            return None
        
        # Add to buffer
        self.frame_buffer.append(processed_roi)
        
        # If buffer is full, make prediction
        if len(self.frame_buffer) == self.sequence_length:
            word, confidence = self.predict_sequence(list(self.frame_buffer))
            
            # Only return prediction if confidence is above threshold
            if confidence >= self.confidence_threshold:
                return word, confidence
        
        return None
    
    def process_video_sequence(self, frames: List[np.ndarray], landmarks_list: List[Dict]) -> List[Tuple[str, float]]:
        """Process a sequence of video frames"""
        predictions = []
        self.frame_buffer.clear()
        
        for frame, landmarks in zip(frames, landmarks_list):
            result = self.process_frame(frame, landmarks)
            if result:
                predictions.append(result)
        
        return predictions
    
    def reset_buffer(self):
        """Reset the frame buffer"""
        self.frame_buffer.clear()