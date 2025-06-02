"""
Main Processing Pipeline for Mouthful
Orchestrates the entire lip reading and voice cloning workflow
"""

import cv2
import numpy as np
import json
import os
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

from .face_recognition_service import FaceRecognitionService, FaceMatch
from .lip_reading_service import LipReadingService
from .voice_cloning_service import VoiceCloningService
from .social_media_search import SocialMediaSearchService, IdentityResolutionService
from .audio_crawler import AudioCrawlerService

@dataclass
class ProcessingResult:
    """Result of processing a video frame or sequence"""
    timestamp: float
    detected_faces: List[FaceMatch]
    lip_reading_results: List[Tuple[str, float]]  # (word, confidence)
    synthesized_audio: Optional[str]  # Path to generated audio
    identity_info: Optional[Dict]
    processing_time: float

class MouthfulProcessor:
    """Main processor that orchestrates all services"""
    
    def __init__(self, config_path: str = "config.json"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize services
        print("Initializing Mouthful services...")
        
        self.face_recognition = FaceRecognitionService(self.config["models"]["face_recognition"])
        self.lip_reading = LipReadingService(self.config["models"]["lip_reading"])
        self.voice_cloning = VoiceCloningService(self.config["models"]["voice_cloning"])
        self.social_media_search = SocialMediaSearchService(self.config["social_media"])
        self.audio_crawler = AudioCrawlerService(self.config["crawling"])
        self.identity_resolver = IdentityResolutionService(self.social_media_search)
        
        # Processing parameters
        self.processing_delay = self.config["processing"]["processing_delay"]
        self.video_fps = self.config["processing"]["video_fps"]
        self.batch_size = self.config["processing"]["batch_size"]
        
        # Frame buffer for live processing
        self.frame_buffer = deque(maxlen=self.processing_delay * self.video_fps)
        self.processing_queue = deque()
        
        # Cache for identified persons
        self.known_persons = {}  # person_id -> identity_info
        
        print("Mouthful processor initialized successfully")
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None) -> List[ProcessingResult]:
        """Process a video file and return results"""
        print(f"Processing video file: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        results = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / self.video_fps
                result = self.process_frame(frame, timestamp)
                
                if result:
                    results.append(result)
                
                frame_count += 1
                
                # Progress logging
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...")
        
        finally:
            cap.release()
        
        print(f"Completed processing {frame_count} frames")
        
        # Save results if output path specified
        if output_path:
            self._save_results(results, output_path)
        
        return results
    
    def process_live_stream(self, stream_url: str, output_callback: Optional[callable] = None):
        """Process live video stream with delay"""
        print(f"Starting live stream processing: {stream_url}")
        print(f"Processing delay: {self.processing_delay} seconds")
        
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise ValueError(f"Could not open stream: {stream_url}")
        
        # Start processing thread
        processing_thread = threading.Thread(target=self._process_delayed_frames, args=(output_callback,))
        processing_thread.daemon = True
        processing_thread.start()
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Stream ended or connection lost")
                    break
                
                current_time = time.time()
                timestamp = current_time - start_time
                
                # Add frame to buffer
                self.frame_buffer.append((frame.copy(), timestamp))
                
                frame_count += 1
                
                # Progress logging
                if frame_count % 100 == 0:
                    print(f"Buffered {frame_count} frames, buffer size: {len(self.frame_buffer)}")
        
        finally:
            cap.release()
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[ProcessingResult]:
        """Process a single frame"""
        start_time = time.time()
        
        # Detect faces
        face_matches = self.face_recognition.recognize_faces(frame)
        
        if not face_matches:
            return None
        
        lip_reading_results = []
        synthesized_audio = None
        identity_info = None
        
        # Process each detected face
        for face_match in face_matches:
            try:
                # Get facial landmarks for lip reading
                face_landmarks = self.face_recognition.get_face_landmarks(
                    frame, [face_match.bounding_box]
                )
                
                if face_landmarks:
                    # Perform lip reading
                    lip_result = self.lip_reading.process_frame(frame, face_landmarks[0])
                    if lip_result:
                        lip_reading_results.append(lip_result)
                        
                        # If we have a confident lip reading result, try to synthesize speech
                        word, confidence = lip_result
                        if confidence > 0.8:
                            # Check if we know this person's voice
                            if self.voice_cloning.has_voice_profile(face_match.person_id):
                                audio_path = self.voice_cloning.synthesize_speech(word, face_match.person_id)
                                if audio_path:
                                    synthesized_audio = audio_path
                            else:
                                # Try to identify person and find their voice
                                identity_info = self._identify_and_train_voice(face_match, frame)
                
            except Exception as e:
                print(f"Error processing face {face_match.person_id}: {e}")
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            timestamp=timestamp,
            detected_faces=face_matches,
            lip_reading_results=lip_reading_results,
            synthesized_audio=synthesized_audio,
            identity_info=identity_info,
            processing_time=processing_time
        )
    
    def _process_delayed_frames(self, output_callback: Optional[callable] = None):
        """Process frames with delay for live streams"""
        while True:
            if len(self.frame_buffer) >= self.processing_delay * self.video_fps:
                # Get the oldest frame (delayed frame)
                frame, timestamp = self.frame_buffer[0]
                
                result = self.process_frame(frame, timestamp)
                
                if result and output_callback:
                    output_callback(result)
                
                if result:
                    print(f"Processed delayed frame at {timestamp:.1f}s")
            
            time.sleep(1.0 / self.video_fps)  # Match video framerate
    
    def _identify_and_train_voice(self, face_match: FaceMatch, frame: np.ndarray) -> Optional[Dict]:
        """Identify person and train voice profile"""
        person_id = face_match.person_id
        
        # Check if we already processed this person
        if person_id in self.known_persons:
            return self.known_persons[person_id]
        
        print(f"Identifying unknown person: {person_id}")
        
        try:
            # Search for identity using face encoding
            identity_info = self.identity_resolver.resolve_identity(face_match.encoding)
            
            if identity_info and identity_info["verified"]:
                print(f"Identified person: {identity_info['primary_name']}")
                
                # Search for audio samples
                audio_sources = self.audio_crawler.search_audio_by_person(
                    identity_info["primary_name"],
                    additional_info={"sources": identity_info["sources"]}
                )
                
                if audio_sources:
                    print(f"Found {len(audio_sources)} potential audio sources")
                    
                    # Download and process audio samples
                    audio_files = self.audio_crawler.batch_download_audio(
                        audio_sources, max_downloads=5
                    )
                    
                    if audio_files:
                        # Train voice profile
                        success = self.voice_cloning.add_person_voice(person_id, audio_files)
                        
                        if success:
                            print(f"Successfully trained voice profile for {identity_info['primary_name']}")
                            identity_info["voice_trained"] = True
                        else:
                            print(f"Failed to train voice profile for {identity_info['primary_name']}")
                            identity_info["voice_trained"] = False
                        
                        # Clean up temporary files
                        self.audio_crawler.cleanup_temp_files(audio_files)
                    
                    else:
                        print(f"No audio files downloaded for {identity_info['primary_name']}")
                        identity_info["voice_trained"] = False
                
                else:
                    print(f"No audio sources found for {identity_info['primary_name']}")
                    identity_info["voice_trained"] = False
                
                # Cache the identity info
                self.known_persons[person_id] = identity_info
                
                # Update face recognition database with name
                if identity_info["primary_name"]:
                    face_region = self.face_recognition.extract_face_region(frame, face_match.bounding_box)
                    self.face_recognition.add_person(
                        person_id, 
                        identity_info["primary_name"], 
                        [face_region]
                    )
                
                return identity_info
            
            else:
                print(f"Could not verify identity for person {person_id}")
                return None
                
        except Exception as e:
            print(f"Error identifying person {person_id}: {e}")
            return None
    
    def _save_results(self, results: List[ProcessingResult], output_path: str):
        """Save processing results to file"""
        try:
            # Convert results to serializable format
            serializable_results = []
            
            for result in results:
                serializable_result = {
                    "timestamp": result.timestamp,
                    "detected_faces": [
                        {
                            "person_id": face.person_id,
                            "confidence": face.confidence,
                            "bounding_box": face.bounding_box,
                            "name": face.name
                        }
                        for face in result.detected_faces
                    ],
                    "lip_reading_results": result.lip_reading_results,
                    "synthesized_audio": result.synthesized_audio,
                    "identity_info": result.identity_info,
                    "processing_time": result.processing_time
                }
                serializable_results.append(serializable_result)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"Results saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        stats = {
            "known_persons": len(self.known_persons),
            "voice_profiles": len(self.voice_cloning.list_voices()),
            "face_database_size": len(self.face_recognition.known_faces),
            "buffer_size": len(self.frame_buffer),
            "cache_info": {
                "social_media_searches": len(self.social_media_search.get_cached_searches()),
                "audio_cache_dir": self.audio_crawler.cache_dir
            }
        }
        return stats
    
    def reset_buffers(self):
        """Reset all processing buffers"""
        self.frame_buffer.clear()
        self.processing_queue.clear()
        self.lip_reading.reset_buffer()
        print("Processing buffers reset")
    
    def add_known_person(self, person_id: str, name: str, images: List[np.ndarray], 
                        audio_files: Optional[List[str]] = None) -> bool:
        """Manually add a known person to the system"""
        try:
            # Add to face recognition
            face_success = self.face_recognition.add_person(person_id, name, images)
            
            voice_success = True
            if audio_files:
                voice_success = self.voice_cloning.add_person_voice(person_id, audio_files)
            
            if face_success:
                self.known_persons[person_id] = {
                    "primary_name": name,
                    "confidence": 1.0,
                    "verified": True,
                    "voice_trained": voice_success,
                    "sources": ["manual"],
                    "manual_entry": True
                }
                
                print(f"Successfully added person: {name}")
                return True
            else:
                print(f"Failed to add person: {name}")
                return False
                
        except Exception as e:
            print(f"Error adding person {name}: {e}")
            return False

class LiveStreamProcessor:
    """Specialized processor for live streaming scenarios"""
    
    def __init__(self, processor: MouthfulProcessor):
        self.processor = processor
        self.output_handlers = []
        self.is_running = False
    
    def add_output_handler(self, handler: callable):
        """Add output handler for processing results"""
        self.output_handlers.append(handler)
    
    def start_processing(self, stream_url: str):
        """Start live stream processing"""
        self.is_running = True
        
        def output_callback(result: ProcessingResult):
            for handler in self.output_handlers:
                try:
                    handler(result)
                except Exception as e:
                    print(f"Error in output handler: {e}")
        
        self.processor.process_live_stream(stream_url, output_callback)
    
    def stop_processing(self):
        """Stop live stream processing"""
        self.is_running = False
        print("Live stream processing stopped")