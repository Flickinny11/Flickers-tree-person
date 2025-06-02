#!/usr/bin/env python3
"""
Simplified Demo for Mouthful
Works with basic Python libraries to demonstrate core concepts
"""

import os
import sys
import json
import time
import random
from pathlib import Path

def create_mock_models():
    """Create mock model files and data"""
    print("Creating mock models and data...")
    
    # Create directories
    os.makedirs("models/face_recognition", exist_ok=True)
    os.makedirs("models/lip_reading", exist_ok=True)
    os.makedirs("models/voice_cloning", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Create mock face database
    face_db = {
        "person_001": {
            "name": "John Doe",
            "encodings": [[0.1, 0.2, 0.3, 0.4, 0.5] * 20]  # Mock 100-dimensional encoding
        },
        "person_002": {
            "name": "Jane Smith", 
            "encodings": [[0.2, 0.3, 0.4, 0.5, 0.6] * 20]
        }
    }
    
    with open("models/face_recognition/known_faces.json", "w") as f:
        json.dump(face_db, f, indent=2)
    
    # Create mock vocabulary
    vocab = {
        "hello": 0, "world": 1, "test": 2, "demo": 3, "the": 4,
        "and": 5, "to": 6, "of": 7, "a": 8, "in": 9, "is": 10,
        "it": 11, "you": 12, "that": 13, "he": 14, "was": 15,
        "<UNK>": 998, "<PAD>": 999
    }
    
    with open("models/lip_reading/vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    
    # Create mock voice profiles
    voice_profiles = {
        "person_001": {
            "fundamental_freq": 120.0,
            "formants": [800, 1200, 2400, 3600],
            "voice_quality": "normal",
            "speaking_rate": 0.5,
            "sample_count": 3
        },
        "person_002": {
            "fundamental_freq": 200.0,
            "formants": [900, 1400, 2600, 3800],
            "voice_quality": "normal", 
            "speaking_rate": 0.6,
            "sample_count": 2
        }
    }
    
    with open("models/voice_cloning/voice_profiles.json", "w") as f:
        json.dump(voice_profiles, f, indent=2)
    
    print("Mock models created successfully")

class MockFaceRecognition:
    """Mock face recognition for demo"""
    
    def __init__(self):
        self.known_faces = {}
        self._load_known_faces()
    
    def _load_known_faces(self):
        try:
            with open("models/face_recognition/known_faces.json", "r") as f:
                self.known_faces = json.load(f)
        except FileNotFoundError:
            self.known_faces = {}
    
    def detect_faces_in_frame(self, frame_data):
        """Simulate face detection"""
        # Mock detection: randomly detect 0-2 faces
        import random
        num_faces = random.choice([0, 1, 1, 2])  # Bias towards 1 face
        
        faces = []
        for i in range(num_faces):
            face = {
                "person_id": random.choice(list(self.known_faces.keys()) + ["unknown"]),
                "confidence": random.uniform(0.7, 0.95),
                "bounding_box": [50 + i*100, 50, 150 + i*100, 150],
                "name": None
            }
            
            if face["person_id"] in self.known_faces:
                face["name"] = self.known_faces[face["person_id"]]["name"]
            
            faces.append(face)
        
        return faces

class MockLipReading:
    """Mock lip reading for demo"""
    
    def __init__(self):
        self.vocab = {}
        self._load_vocabulary()
        self.frame_buffer = []
    
    def _load_vocabulary(self):
        try:
            with open("models/lip_reading/vocab.json", "r") as f:
                self.vocab = json.load(f)
        except FileNotFoundError:
            self.vocab = {"hello": 0, "world": 1, "test": 2}
    
    def process_frame(self, frame_data, face_landmarks):
        """Simulate lip reading"""
        import random
        
        # Add frame to buffer
        self.frame_buffer.append(frame_data)
        
        # Process every 10 frames
        if len(self.frame_buffer) >= 10:
            self.frame_buffer = self.frame_buffer[-10:]  # Keep last 10
            
            # Simulate lip reading result
            if random.random() < 0.3:  # 30% chance of detecting speech
                words = list(self.vocab.keys())[:15]  # Use first 15 words
                word = random.choice(words)
                confidence = random.uniform(0.6, 0.9)
                return word, confidence
        
        return None

class MockVoiceCloning:
    """Mock voice cloning for demo"""
    
    def __init__(self):
        self.voice_profiles = {}
        self._load_voice_profiles()
    
    def _load_voice_profiles(self):
        try:
            with open("models/voice_cloning/voice_profiles.json", "r") as f:
                self.voice_profiles = json.load(f)
        except FileNotFoundError:
            self.voice_profiles = {}
    
    def synthesize_speech(self, text, person_id):
        """Simulate speech synthesis"""
        if person_id not in self.voice_profiles:
            return None
        
        # Create mock audio file path
        audio_path = f"temp/synthesized_{person_id}_{hash(text)}.wav"
        
        # Simulate creating audio file
        with open(audio_path, "w") as f:
            f.write(f"# Mock audio file for '{text}' in {person_id}'s voice\n")
            f.write(f"# Voice profile: {self.voice_profiles[person_id]}\n")
        
        print(f"Synthesized: '{text}' -> {audio_path}")
        return audio_path
    
    def has_voice_profile(self, person_id):
        return person_id in self.voice_profiles

class MockProcessor:
    """Mock main processor"""
    
    def __init__(self):
        self.face_recognition = MockFaceRecognition()
        self.lip_reading = MockLipReading()
        self.voice_cloning = MockVoiceCloning()
        self.known_persons = {}
    
    def process_frame(self, frame_number):
        """Process a single frame"""
        start_time = time.time()
        
        # Mock frame data
        frame_data = {"frame": frame_number, "timestamp": frame_number * 0.04}  # 25fps
        
        # Detect faces
        faces = self.face_recognition.detect_faces_in_frame(frame_data)
        
        # Process lip reading for each face
        lip_results = []
        synthesized_audio = None
        
        for face in faces:
            # Mock face landmarks
            face_landmarks = {"mouth": "mock_landmarks"}
            
            # Try lip reading
            lip_result = self.lip_reading.process_frame(frame_data, face_landmarks)
            
            if lip_result:
                word, confidence = lip_result
                lip_results.append((word, confidence))
                
                # Try voice synthesis
                if confidence > 0.8 and self.voice_cloning.has_voice_profile(face["person_id"]):
                    audio_path = self.voice_cloning.synthesize_speech(word, face["person_id"])
                    if audio_path:
                        synthesized_audio = audio_path
        
        processing_time = time.time() - start_time
        
        return {
            "frame": frame_number,
            "timestamp": frame_data["timestamp"],
            "faces": faces,
            "lip_reading": lip_results,
            "synthesized_audio": synthesized_audio,
            "processing_time": processing_time
        }
    
    def process_video_sequence(self, num_frames=100):
        """Process a sequence of frames"""
        print(f"Processing {num_frames} frames...")
        
        results = []
        for frame_num in range(num_frames):
            result = self.process_frame(frame_num)
            results.append(result)
            
            # Print progress
            if (frame_num + 1) % 20 == 0:
                print(f"Processed {frame_num + 1}/{num_frames} frames...")
        
        return results
    
    def get_statistics(self):
        """Get system statistics"""
        return {
            "known_faces": len(self.face_recognition.known_faces),
            "voice_profiles": len(self.voice_cloning.voice_profiles),
            "vocabulary_size": len(self.lip_reading.vocab),
            "known_persons": len(self.known_persons)
        }

def demo_face_recognition():
    """Demo face recognition functionality"""
    print("\n=== Face Recognition Demo ===")
    
    face_rec = MockFaceRecognition()
    
    print(f"Known faces: {len(face_rec.known_faces)}")
    for person_id, data in face_rec.known_faces.items():
        print(f"  {person_id}: {data['name']}")
    
    # Simulate detecting faces in frames
    for frame_num in range(5):
        faces = face_rec.detect_faces_in_frame({"frame": frame_num})
        print(f"Frame {frame_num}: Detected {len(faces)} faces")
        for face in faces:
            name = face['name'] or face['person_id']
            print(f"  - {name} (confidence: {face['confidence']:.2f})")

def demo_lip_reading():
    """Demo lip reading functionality"""
    print("\n=== Lip Reading Demo ===")
    
    lip_reader = MockLipReading()
    
    print(f"Vocabulary size: {len(lip_reader.vocab)}")
    print(f"Sample words: {list(lip_reader.vocab.keys())[:10]}")
    
    # Simulate processing frames
    for frame_num in range(25):  # Process 1 second at 25fps
        result = lip_reader.process_frame({"frame": frame_num}, {"mouth": "landmarks"})
        if result:
            word, confidence = result
            print(f"Frame {frame_num}: Detected word '{word}' (confidence: {confidence:.2f})")

def demo_voice_synthesis():
    """Demo voice synthesis functionality"""
    print("\n=== Voice Synthesis Demo ===")
    
    voice_cloner = MockVoiceCloning()
    
    print(f"Available voices: {len(voice_cloner.voice_profiles)}")
    for person_id, profile in voice_cloner.voice_profiles.items():
        print(f"  {person_id}: f0={profile['fundamental_freq']}Hz, rate={profile['speaking_rate']}")
    
    # Test synthesis
    test_phrases = ["Hello world", "This is a test", "Mouthful demo"]
    for person_id in voice_cloner.voice_profiles.keys():
        for phrase in test_phrases[:2]:  # Test first 2 phrases
            audio_path = voice_cloner.synthesize_speech(phrase, person_id)
            if audio_path:
                print(f"Generated: {phrase} -> {audio_path}")

def demo_full_processing():
    """Demo the full processing pipeline"""
    print("\n=== Full Processing Pipeline Demo ===")
    
    processor = MockProcessor()
    
    # Show initial statistics
    print("Initial system statistics:")
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Process video sequence
    results = processor.process_video_sequence(50)  # Process 2 seconds
    
    # Analyze results
    total_faces = sum(len(r["faces"]) for r in results)
    total_lip_readings = sum(len(r["lip_reading"]) for r in results)
    total_audio_generated = sum(1 for r in results if r["synthesized_audio"])
    avg_processing_time = sum(r["processing_time"] for r in results) / len(results)
    
    print(f"\nProcessing Results:")
    print(f"  Total frames processed: {len(results)}")
    print(f"  Total faces detected: {total_faces}")
    print(f"  Total lip reading results: {total_lip_readings}")
    print(f"  Audio files generated: {total_audio_generated}")
    print(f"  Average processing time: {avg_processing_time:.4f}s per frame")
    
    # Show sample results
    print(f"\nSample results (first 5 frames with activity):")
    active_results = [r for r in results if r["faces"] or r["lip_reading"]][:5]
    
    for result in active_results:
        print(f"Frame {result['frame']} ({result['timestamp']:.2f}s):")
        for face in result["faces"]:
            name = face["name"] or face["person_id"]
            print(f"  Face: {name} (confidence: {face['confidence']:.2f})")
        
        for word, confidence in result["lip_reading"]:
            print(f"  Lip reading: '{word}' (confidence: {confidence:.2f})")
        
        if result["synthesized_audio"]:
            print(f"  Generated audio: {result['synthesized_audio']}")
    
    # Save results
    output_file = "output/demo_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

def main():
    """Main demo function"""
    print("Mouthful - Simplified Demo")
    print("=" * 40)
    
    # Create mock models and data
    create_mock_models()
    
    try:
        # Run individual component demos
        demo_face_recognition()
        demo_lip_reading() 
        demo_voice_synthesis()
        
        # Run full pipeline demo
        demo_full_processing()
        
        print("\n" + "=" * 40)
        print("Demo completed successfully!")
        print("\nGenerated files:")
        print("  - models/: Mock AI models and data")
        print("  - temp/: Generated audio files")
        print("  - output/demo_results.json: Processing results")
        
        print("\nNext steps:")
        print("1. Install full dependencies: pip install -r requirements.txt")
        print("2. Run real API server: python main.py")
        print("3. Open web interface: web_interface.html")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()