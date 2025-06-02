"""
Face Recognition Module for Mouthful
Handles face detection, recognition, and identity verification
"""

import cv2
import face_recognition
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os
from dataclasses import dataclass

@dataclass
class FaceMatch:
    """Represents a face match with confidence score"""
    person_id: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    encoding: np.ndarray
    name: Optional[str] = None

class FaceRecognitionService:
    """Service for face detection and recognition"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.known_faces = {}  # person_id -> {"name": str, "encodings": List[np.ndarray]}
        self.tolerance = config.get("tolerance", 0.6)
        self.model = config.get("model", "hog")
        
        # Load known faces database
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load known faces from the database"""
        db_path = os.path.join(self.config.get("model_location", "models/face_recognition"), "known_faces.json")
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                data = json.load(f)
                for person_id, person_data in data.items():
                    self.known_faces[person_id] = {
                        "name": person_data["name"],
                        "encodings": [np.array(enc) for enc in person_data["encodings"]]
                    }
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image
        Returns list of bounding boxes (top, right, bottom, left)
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        face_locations = face_recognition.face_locations(rgb_image, model=self.model)
        return face_locations
    
    def get_face_encodings(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Get face encodings for detected faces"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        return encodings
    
    def recognize_faces(self, image: np.ndarray) -> List[FaceMatch]:
        """
        Recognize faces in an image
        Returns list of face matches with confidence scores
        """
        face_locations = self.detect_faces(image)
        face_encodings = self.get_face_encodings(image, face_locations)
        
        matches = []
        
        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            best_match = self._find_best_match(face_encoding)
            
            if best_match:
                person_id, confidence, name = best_match
                match = FaceMatch(
                    person_id=person_id,
                    confidence=confidence,
                    bounding_box=face_location,
                    encoding=face_encoding,
                    name=name
                )
                matches.append(match)
            else:
                # Unknown face - generate new ID
                unknown_id = f"unknown_{len(matches)}_{hash(str(face_encoding[:5]))}"
                match = FaceMatch(
                    person_id=unknown_id,
                    confidence=0.0,
                    bounding_box=face_location,
                    encoding=face_encoding,
                    name="Unknown"
                )
                matches.append(match)
        
        return matches
    
    def _find_best_match(self, face_encoding: np.ndarray) -> Optional[Tuple[str, float, str]]:
        """Find the best match for a face encoding"""
        best_person_id = None
        best_confidence = 0.0
        best_name = None
        
        for person_id, person_data in self.known_faces.items():
            distances = face_recognition.face_distance(person_data["encodings"], face_encoding)
            min_distance = np.min(distances)
            
            # Convert distance to confidence (lower distance = higher confidence)
            confidence = max(0, 1 - min_distance)
            
            if min_distance <= self.tolerance and confidence > best_confidence:
                best_person_id = person_id
                best_confidence = confidence
                best_name = person_data["name"]
        
        if best_person_id:
            return best_person_id, best_confidence, best_name
        return None
    
    def add_person(self, person_id: str, name: str, images: List[np.ndarray]):
        """Add a new person to the known faces database"""
        encodings = []
        
        for image in images:
            face_locations = self.detect_faces(image)
            if face_locations:
                face_encodings = self.get_face_encodings(image, face_locations)
                if face_encodings:
                    encodings.extend(face_encodings)
        
        if encodings:
            self.known_faces[person_id] = {
                "name": name,
                "encodings": encodings
            }
            self._save_known_faces()
            return True
        return False
    
    def _save_known_faces(self):
        """Save known faces database to disk"""
        db_path = os.path.join(self.config.get("model_location", "models/face_recognition"), "known_faces.json")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for person_id, person_data in self.known_faces.items():
            serializable_data[person_id] = {
                "name": person_data["name"],
                "encodings": [encoding.tolist() for encoding in person_data["encodings"]]
            }
        
        with open(db_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def extract_face_region(self, image: np.ndarray, bounding_box: Tuple[int, int, int, int], 
                           padding: int = 20) -> np.ndarray:
        """Extract face region from image with padding"""
        top, right, bottom, left = bounding_box
        
        # Add padding
        height, width = image.shape[:2]
        top = max(0, top - padding)
        right = min(width, right + padding)
        bottom = min(height, bottom + padding)
        left = max(0, left - padding)
        
        face_region = image[top:bottom, left:right]
        return face_region
    
    def get_face_landmarks(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """Get facial landmarks for lip reading"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        landmarks_list = face_recognition.face_landmarks(rgb_image, face_locations)
        return landmarks_list