"""
Mouthful Package
Lip Reading with Voice Cloning Service
"""

from .processor import MouthfulProcessor, LiveStreamProcessor
from .face_recognition_service import FaceRecognitionService, FaceMatch
from .lip_reading_service import LipReadingService
from .voice_cloning_service import VoiceCloningService
from .social_media_search import SocialMediaSearchService, IdentityResolutionService
from .audio_crawler import AudioCrawlerService

__version__ = "1.0.0"
__author__ = "Mouthful Development Team"
__description__ = "AI-powered lip reading with voice cloning"

__all__ = [
    "MouthfulProcessor",
    "LiveStreamProcessor", 
    "FaceRecognitionService",
    "FaceMatch",
    "LipReadingService",
    "VoiceCloningService",
    "SocialMediaSearchService",
    "IdentityResolutionService",
    "AudioCrawlerService"
]