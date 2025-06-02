"""
Audio Crawler Module for Mouthful
Searches and downloads audio samples from the web for voice training
"""

import requests
import os
import time
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse, quote
import tempfile
from dataclasses import dataclass
import librosa
import soundfile as sf

@dataclass
class AudioSource:
    """Represents an audio source found online"""
    url: str
    title: str
    source_type: str  # 'video', 'podcast', 'interview', 'speech'
    duration: float
    quality_score: float
    metadata: Dict

class AudioCrawlerService:
    """Service for finding and downloading audio samples"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_search_results = config.get("max_search_results", 50)
        self.max_audio_samples = config.get("max_audio_samples", 10)
        self.timeout = config.get("timeout_seconds", 30)
        self.user_agents = config.get("user_agents", [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        ])
        
        self.cache_dir = "cache/audio_sources"
        self.temp_dir = "temp/audio"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Audio quality thresholds
        self.min_duration = 5.0  # seconds
        self.max_duration = 300.0  # 5 minutes
        self.min_sample_rate = 16000
        self.preferred_formats = ['wav', 'mp3', 'flac', 'm4a']
        
        # Common audio hosting platforms and their patterns
        self.audio_platforms = {
            "youtube": {
                "search_url": "https://www.googleapis.com/youtube/v3/search",
                "extract_pattern": r"youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
                "quality_score": 0.8
            },
            "soundcloud": {
                "search_url": "https://api.soundcloud.com/tracks",
                "extract_pattern": r"soundcloud\.com/([^/]+)/([^/]+)",
                "quality_score": 0.9
            },
            "podcast_platforms": {
                "quality_score": 0.95
            }
        }
    
    def search_audio_by_person(self, person_name: str, additional_info: Optional[Dict] = None) -> List[AudioSource]:
        """Search for audio samples featuring a specific person"""
        print(f"Searching for audio samples of: {person_name}")
        
        # Check cache first
        cache_key = hashlib.md5(f"{person_name}_{str(additional_info)}".encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"search_{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if time.time() - cached_data.get("timestamp", 0) < 7200:  # 2 hour cache
                        print("Using cached audio search results")
                        return [AudioSource(**source) for source in cached_data["sources"]]
            except Exception as e:
                print(f"Error loading audio cache: {e}")
        
        # Build search queries
        search_queries = self._build_search_queries(person_name, additional_info)
        
        all_sources = []
        
        # Search different platforms
        for query in search_queries:
            try:
                # YouTube search
                youtube_sources = self._search_youtube(query)
                all_sources.extend(youtube_sources)
                
                # Podcast search
                podcast_sources = self._search_podcasts(query)
                all_sources.extend(podcast_sources)
                
                # News/Interview search
                news_sources = self._search_news_audio(query)
                all_sources.extend(news_sources)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error searching with query '{query}': {e}")
        
        # Remove duplicates and sort by quality
        unique_sources = self._deduplicate_sources(all_sources)
        unique_sources.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Limit results
        unique_sources = unique_sources[:self.max_search_results]
        
        # Cache results
        try:
            cache_data = {
                "timestamp": time.time(),
                "sources": [self._source_to_dict(source) for source in unique_sources]
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Error saving audio cache: {e}")
        
        return unique_sources
    
    def _build_search_queries(self, person_name: str, additional_info: Optional[Dict] = None) -> List[str]:
        """Build search queries for finding audio content"""
        queries = []
        
        # Basic queries
        queries.extend([
            f'"{person_name}" interview',
            f'"{person_name}" speech',
            f'"{person_name}" podcast',
            f'"{person_name}" audio',
            f'"{person_name}" voice'
        ])
        
        # Add context-specific queries if additional info available
        if additional_info:
            if "profession" in additional_info:
                profession = additional_info["profession"]
                queries.extend([
                    f'"{person_name}" {profession}',
                    f'{profession} "{person_name}"'
                ])
            
            if "company" in additional_info:
                company = additional_info["company"]
                queries.extend([
                    f'"{person_name}" {company}',
                    f'{company} "{person_name}"'
                ])
            
            if "topics" in additional_info:
                for topic in additional_info["topics"][:3]:  # Limit to top 3 topics
                    queries.append(f'"{person_name}" {topic}')
        
        return queries[:10]  # Limit total queries
    
    def _search_youtube(self, query: str) -> List[AudioSource]:
        """Search YouTube for audio content (mock implementation)"""
        # In a real implementation, you would use YouTube Data API
        print(f"Searching YouTube for: {query}")
        
        # Mock results
        mock_sources = [
            AudioSource(
                url="https://youtube.com/watch?v=example1",
                title=f"{query} - Interview",
                source_type="video",
                duration=120.0,
                quality_score=0.8,
                metadata={
                    "platform": "youtube",
                    "views": 10000,
                    "upload_date": "2023-01-15"
                }
            ),
            AudioSource(
                url="https://youtube.com/watch?v=example2",
                title=f"{query} - Conference Talk",
                source_type="speech",
                duration=300.0,
                quality_score=0.85,
                metadata={
                    "platform": "youtube",
                    "views": 5000,
                    "upload_date": "2023-03-20"
                }
            )
        ]
        
        return mock_sources
    
    def _search_podcasts(self, query: str) -> List[AudioSource]:
        """Search podcast platforms for audio content"""
        print(f"Searching podcasts for: {query}")
        
        # Mock podcast results
        mock_sources = [
            AudioSource(
                url="https://podcast.example.com/episode123",
                title=f"Podcast Episode featuring {query}",
                source_type="podcast",
                duration=1800.0,  # 30 minutes
                quality_score=0.95,
                metadata={
                    "platform": "podcast",
                    "show_name": "Tech Talks",
                    "episode_number": 123,
                    "publish_date": "2023-02-10"
                }
            )
        ]
        
        return mock_sources
    
    def _search_news_audio(self, query: str) -> List[AudioSource]:
        """Search news sites for audio interviews/reports"""
        print(f"Searching news audio for: {query}")
        
        # Mock news audio results
        mock_sources = [
            AudioSource(
                url="https://news.example.com/audio/interview456",
                title=f"News Interview with {query}",
                source_type="interview",
                duration=600.0,  # 10 minutes
                quality_score=0.9,
                metadata={
                    "platform": "news",
                    "outlet": "Example News",
                    "publish_date": "2023-04-05"
                }
            )
        ]
        
        return mock_sources
    
    def _deduplicate_sources(self, sources: List[AudioSource]) -> List[AudioSource]:
        """Remove duplicate audio sources"""
        seen_urls = set()
        unique_sources = []
        
        for source in sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)
        
        return unique_sources
    
    def _source_to_dict(self, source: AudioSource) -> Dict:
        """Convert AudioSource to dictionary for caching"""
        return {
            "url": source.url,
            "title": source.title,
            "source_type": source.source_type,
            "duration": source.duration,
            "quality_score": source.quality_score,
            "metadata": source.metadata
        }
    
    def download_audio(self, source: AudioSource, output_dir: Optional[str] = None) -> Optional[str]:
        """Download audio from source URL"""
        if output_dir is None:
            output_dir = self.temp_dir
        
        try:
            print(f"Downloading audio from: {source.url}")
            
            # Generate filename
            safe_title = "".join(c for c in source.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
            filename = f"{safe_title}_{hashlib.md5(source.url.encode()).hexdigest()[:8]}"
            
            # For demonstration, create a mock audio file
            output_path = os.path.join(output_dir, f"{filename}.wav")
            
            # In a real implementation, you would:
            # 1. Download the audio/video file
            # 2. Extract audio if it's a video
            # 3. Convert to standard format
            # 4. Validate audio quality
            
            # Create mock audio file for demo
            self._create_mock_audio(output_path, source.duration)
            
            # Validate downloaded audio
            if self._validate_audio_quality(output_path):
                print(f"Successfully downloaded audio to: {output_path}")
                return output_path
            else:
                print(f"Downloaded audio failed quality validation")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return None
                
        except Exception as e:
            print(f"Error downloading audio from {source.url}: {e}")
            return None
    
    def _create_mock_audio(self, output_path: str, duration: float):
        """Create mock audio file for demonstration"""
        # Generate simple synthetic speech-like audio
        sample_rate = 22050
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        
        # Create speech-like signal with formants
        fundamental_freq = 150  # Hz
        signal = np.sin(2 * np.pi * fundamental_freq * t)
        
        # Add formants (speech characteristics)
        signal += 0.5 * np.sin(2 * np.pi * fundamental_freq * 2 * t)  # 2nd harmonic
        signal += 0.3 * np.sin(2 * np.pi * fundamental_freq * 3 * t)  # 3rd harmonic
        
        # Add some noise and modulation
        signal += 0.1 * np.random.normal(0, 1, len(signal))
        signal *= (1 + 0.2 * np.sin(2 * np.pi * 5 * t))  # Amplitude modulation
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.7
        
        # Save as WAV
        sf.write(output_path, signal, sample_rate)
    
    def _validate_audio_quality(self, audio_path: str) -> bool:
        """Validate audio quality for voice training"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path)
            duration = len(audio) / sr
            
            # Check duration
            if duration < self.min_duration or duration > self.max_duration:
                print(f"Audio duration {duration:.1f}s outside acceptable range")
                return False
            
            # Check sample rate
            if sr < self.min_sample_rate:
                print(f"Sample rate {sr} too low")
                return False
            
            # Check for silence
            rms_energy = librosa.feature.rms(y=audio)[0]
            avg_energy = np.mean(rms_energy)
            
            if avg_energy < 0.01:  # Very quiet
                print("Audio appears to be too quiet or silent")
                return False
            
            # Check for clipping
            if np.max(np.abs(audio)) > 0.95:
                print("Audio appears to be clipped")
                return False
            
            print(f"Audio validation passed: {duration:.1f}s, {sr}Hz, energy={avg_energy:.4f}")
            return True
            
        except Exception as e:
            print(f"Error validating audio {audio_path}: {e}")
            return False
    
    def batch_download_audio(self, sources: List[AudioSource], max_downloads: Optional[int] = None) -> List[str]:
        """Download multiple audio sources"""
        if max_downloads is None:
            max_downloads = self.max_audio_samples
        
        downloaded_files = []
        download_count = 0
        
        for source in sources:
            if download_count >= max_downloads:
                break
            
            audio_path = self.download_audio(source)
            if audio_path:
                downloaded_files.append(audio_path)
                download_count += 1
            
            # Rate limiting
            time.sleep(2)
        
        print(f"Downloaded {len(downloaded_files)} audio files")
        return downloaded_files
    
    def cleanup_temp_files(self, files: List[str]):
        """Clean up temporary audio files"""
        for file_path in files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing temp file {file_path}: {e}")
    
    def get_audio_metadata(self, audio_path: str) -> Dict:
        """Extract metadata from audio file"""
        try:
            audio, sr = librosa.load(audio_path)
            duration = len(audio) / sr
            
            # Basic audio analysis
            rms_energy = librosa.feature.rms(y=audio)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            metadata = {
                "duration": float(duration),
                "sample_rate": int(sr),
                "avg_energy": float(np.mean(rms_energy)),
                "avg_spectral_centroid": float(np.mean(spectral_centroid)),
                "avg_zero_crossing_rate": float(np.mean(zero_crossing_rate)),
                "max_amplitude": float(np.max(np.abs(audio))),
                "file_size": os.path.getsize(audio_path)
            }
            
            return metadata
            
        except Exception as e:
            print(f"Error extracting metadata from {audio_path}: {e}")
            return {}

# Import numpy for audio processing
import numpy as np