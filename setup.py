#!/usr/bin/env python3
"""
Setup script for Mouthful
Downloads required models and sets up the environment
"""

import os
import json
import urllib.request
import zipfile
import sys
from pathlib import Path

def create_directories():
    """Create required directories"""
    directories = [
        "models/face_recognition",
        "models/lip_reading", 
        "models/voice_cloning",
        "temp",
        "output",
        "cache/social_media",
        "cache/audio_sources"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Dependencies installed successfully")
        else:
            print(f"Error installing dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        return False
    
    return True

def download_models():
    """Download or create placeholder models"""
    print("Setting up AI models...")
    
    # Create placeholder model files for demo
    models_info = {
        "face_recognition": {
            "path": "models/face_recognition/known_faces.json",
            "content": {}
        },
        "lip_reading": {
            "path": "models/lip_reading/vocab.json", 
            "content": {
                "hello": 0, "world": 1, "test": 2, "demo": 3,
                "the": 4, "and": 5, "to": 6, "of": 7,
                "<UNK>": 998, "<PAD>": 999
            }
        },
        "voice_cloning": {
            "path": "models/voice_cloning/voice_profiles.json",
            "content": {}
        }
    }
    
    for model_name, model_info in models_info.items():
        model_path = model_info["path"]
        
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            with open(model_path, 'w') as f:
                json.dump(model_info["content"], f, indent=2)
            
            print(f"Created placeholder model: {model_path}")
        else:
            print(f"Model already exists: {model_path}")

def check_system_requirements():
    """Check system requirements"""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        return False
    
    print(f"Python version: {sys.version}")
    
    # Check for OpenCV
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("Warning: OpenCV not installed. Will be installed with dependencies.")
    
    # Check for PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("Warning: PyTorch not installed. Will be installed with dependencies.")
    
    # Check available disk space
    import shutil
    free_space_gb = shutil.disk_usage('.').free / (1024**3)
    print(f"Available disk space: {free_space_gb:.1f} GB")
    
    if free_space_gb < 5:
        print("Warning: Less than 5GB of free space available. May need more for models.")
    
    return True

def setup_configuration():
    """Verify and update configuration"""
    print("Checking configuration...")
    
    config_path = "config.json"
    if os.path.exists(config_path):
        print("Configuration file already exists")
        
        # Verify configuration structure
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            required_sections = ["models", "processing", "crawling", "social_media", "api", "storage"]
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                print(f"Warning: Missing configuration sections: {missing_sections}")
            else:
                print("Configuration structure is valid")
                
        except Exception as e:
            print(f"Error reading configuration: {e}")
            return False
    else:
        print("Configuration file not found, please ensure config.json exists")
        return False
    
    return True

def test_installation():
    """Test the installation"""
    print("Testing installation...")
    
    try:
        # Test imports
        from mouthful import MouthfulProcessor
        print("✓ Mouthful package imports successfully")
        
        # Test processor initialization
        if os.path.exists("config.json"):
            processor = MouthfulProcessor("config.json")
            print("✓ Processor initializes successfully")
            
            # Test basic functionality
            stats = processor.get_statistics()
            print(f"✓ System statistics: {stats}")
            
        else:
            print("⚠ Cannot test processor without config.json")
        
        # Test API imports
        try:
            from main import app
            print("✓ API application imports successfully")
        except Exception as e:
            print(f"⚠ API import warning: {e}")
        
        print("Installation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup function"""
    print("Mouthful Setup Script")
    print("=" * 30)
    
    # Check system requirements
    if not check_system_requirements():
        print("System requirements check failed")
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("Dependency installation failed")
        return False
    
    # Setup models
    download_models()
    
    # Verify configuration
    if not setup_configuration():
        print("Configuration setup failed")
        return False
    
    # Test installation
    if not test_installation():
        print("Installation test failed")
        return False
    
    print("\n" + "=" * 50)
    print("Mouthful setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: python demo.py")
    print("2. Start the API server: python main.py")
    print("3. Check the README.md for more information")
    print("\nAPI will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)