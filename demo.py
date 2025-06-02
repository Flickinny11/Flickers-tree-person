#!/usr/bin/env python3
"""
Demo script for Mouthful service
Shows basic functionality of the lip reading and voice cloning system
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mouthful import MouthfulProcessor

def create_demo_video():
    """Create a simple demo video with a face"""
    print("Creating demo video...")
    
    # Create a simple video with a moving circle (representing a face)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_video.mp4', fourcc, 25.0, (640, 480))
    
    for frame_num in range(125):  # 5 seconds at 25 fps
        # Create a black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a simple face-like circle
        center_x = 320 + int(50 * np.sin(frame_num * 0.1))
        center_y = 240
        
        # Face circle
        cv2.circle(frame, (center_x, center_y), 100, (255, 255, 255), -1)
        
        # Eyes
        cv2.circle(frame, (center_x - 30, center_y - 20), 10, (0, 0, 0), -1)
        cv2.circle(frame, (center_x + 30, center_y - 20), 10, (0, 0, 0), -1)
        
        # Mouth (animated)
        mouth_width = 30 + int(10 * np.sin(frame_num * 0.5))
        cv2.ellipse(frame, (center_x, center_y + 30), (mouth_width, 15), 0, 0, 180, (0, 0, 0), -1)
        
        # Add frame number text
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("Demo video created: demo_video.mp4")

def demo_basic_functionality():
    """Demonstrate basic functionality of Mouthful"""
    print("\n=== Mouthful Demo ===")
    
    # Check if configuration exists
    if not os.path.exists("config.json"):
        print("Configuration file not found. Please ensure config.json exists.")
        return
    
    try:
        # Initialize the processor
        print("Initializing Mouthful processor...")
        processor = MouthfulProcessor("config.json")
        
        # Display system statistics
        print("\nSystem Statistics:")
        stats = processor.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Create demo video if it doesn't exist
        if not os.path.exists("demo_video.mp4"):
            create_demo_video()
        
        # Process demo video
        print("\nProcessing demo video...")
        results = processor.process_video_file("demo_video.mp4", "demo_results.json")
        
        print(f"Processing completed. {len(results)} frames processed.")
        
        # Display sample results
        if results:
            print("\nSample processing results:")
            for i, result in enumerate(results[:3]):  # Show first 3 results
                print(f"Frame {i}:")
                print(f"  Timestamp: {result.timestamp:.2f}s")
                print(f"  Detected faces: {len(result.detected_faces)}")
                print(f"  Lip reading results: {len(result.lip_reading_results)}")
                print(f"  Processing time: {result.processing_time:.3f}s")
                
                if result.detected_faces:
                    for j, face in enumerate(result.detected_faces):
                        print(f"    Face {j}: {face.name or face.person_id} (confidence: {face.confidence:.2f})")
                
                if result.lip_reading_results:
                    for word, confidence in result.lip_reading_results:
                        print(f"    Lip reading: '{word}' (confidence: {confidence:.2f})")
        
        # Test voice synthesis if we have voice profiles
        voice_profiles = processor.voice_cloning.list_voices()
        if voice_profiles:
            print(f"\nTesting voice synthesis with {len(voice_profiles)} available voices...")
            test_text = "Hello, this is a test of voice synthesis."
            
            for voice_id in voice_profiles[:2]:  # Test first 2 voices
                print(f"Synthesizing with voice: {voice_id}")
                audio_path = processor.voice_cloning.synthesize_speech(test_text, voice_id)
                if audio_path:
                    print(f"  Generated audio: {audio_path}")
                else:
                    print(f"  Failed to generate audio for {voice_id}")
        else:
            print("\nNo voice profiles available for synthesis testing.")
        
        # Demonstrate adding a person manually
        print("\nDemonstrating manual person addition...")
        
        # Create a simple demo face image
        demo_image = np.ones((200, 200, 3), dtype=np.uint8) * 128  # Gray image
        cv2.rectangle(demo_image, (50, 50), (150, 150), (255, 255, 255), -1)  # White square face
        cv2.circle(demo_image, (80, 80), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(demo_image, (120, 80), 5, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(demo_image, (90, 110), (110, 120), (0, 0, 0), -1)  # Mouth
        
        success = processor.add_known_person("demo_person", "Demo Person", [demo_image])
        if success:
            print("Successfully added demo person to the system")
        else:
            print("Failed to add demo person")
        
        # Final statistics
        print("\nFinal system statistics:")
        final_stats = processor.get_statistics()
        for key, value in final_stats.items():
            print(f"  {key}: {value}")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()

def demo_api_usage():
    """Demonstrate API usage examples"""
    print("\n=== API Usage Examples ===")
    
    print("To start the API server:")
    print("  python main.py")
    print()
    
    print("Example API calls:")
    print("1. Upload and process video:")
    print("   curl -X POST 'http://localhost:8000/upload/video' -F 'file=@video.mp4'")
    print()
    
    print("2. Add a person:")
    print("   curl -X POST 'http://localhost:8000/persons/add' \\")
    print("        -F 'name=John Doe' \\")
    print("        -F 'images=@photo1.jpg' \\")
    print("        -F 'images=@photo2.jpg' \\")
    print("        -F 'audio_files=@voice1.wav'")
    print()
    
    print("3. List all persons:")
    print("   curl 'http://localhost:8000/persons'")
    print()
    
    print("4. Synthesize speech:")
    print("   curl -X POST 'http://localhost:8000/synthesize/person_123?text=Hello%20world'")
    print()
    
    print("5. Get system statistics:")
    print("   curl 'http://localhost:8000/stats'")
    print()
    
    print("6. Start live stream processing:")
    print("   curl -X POST 'http://localhost:8000/process/live' \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"video_url\": \"rtmp://stream-url\", \"delay_seconds\": 180}'")

if __name__ == "__main__":
    print("Mouthful - Lip Reading with Voice Cloning Demo")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    
    # Check if running from correct directory
    if not os.path.exists("config.json"):
        print("Warning: config.json not found. Make sure you're running from the project root directory.")
    
    try:
        # Run basic functionality demo
        demo_basic_functionality()
        
        # Show API usage examples
        demo_api_usage()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDemo finished. Check the generated files:")
    print("  - demo_video.mp4: Generated demo video")
    print("  - demo_results.json: Processing results")
    print("  - temp/: Temporary files directory")
    print("  - models/: AI models directory")