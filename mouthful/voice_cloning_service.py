"""
Voice Cloning Service for Mouthful
Implements voice synthesis and cloning from audio samples
"""

import torch
import numpy as np
import soundfile as sf
import librosa
from typing import List, Dict, Optional, Tuple
import os
import json
from pathlib import Path
import tempfile
import subprocess

class VoiceCloning:
    """Simple voice cloning implementation using basic TTS"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 22050
        self.voice_profiles = {}  # person_id -> voice_profile
        
        # Create models directory
        models_dir = config.get("model_path", "models/voice_cloning")
        os.makedirs(models_dir, exist_ok=True)
        
        # Load existing voice profiles
        self._load_voice_profiles()
        
        # Initialize simple TTS (for demo purposes)
        self._init_tts()
    
    def _init_tts(self):
        """Initialize Text-to-Speech system"""
        # For this demo, we'll use a simple approach
        # In production, you would use models like Tacotron2, WaveNet, etc.
        try:
            # Try to use espeak if available
            result = subprocess.run(['which', 'espeak'], capture_output=True)
            self.tts_available = result.returncode == 0
        except:
            self.tts_available = False
        
        print(f"TTS System initialized. espeak available: {self.tts_available}")
    
    def _load_voice_profiles(self):
        """Load existing voice profiles from disk"""
        profiles_path = os.path.join(self.config.get("model_path", "models/voice_cloning"), "voice_profiles.json")
        
        if os.path.exists(profiles_path):
            with open(profiles_path, 'r') as f:
                profiles_data = json.load(f)
                
            for person_id, profile_data in profiles_data.items():
                self.voice_profiles[person_id] = {
                    "fundamental_freq": profile_data["fundamental_freq"],
                    "formants": profile_data["formants"],
                    "voice_quality": profile_data["voice_quality"],
                    "speaking_rate": profile_data["speaking_rate"],
                    "sample_count": profile_data["sample_count"]
                }
        
        print(f"Loaded {len(self.voice_profiles)} voice profiles")
    
    def _save_voice_profiles(self):
        """Save voice profiles to disk"""
        profiles_path = os.path.join(self.config.get("model_path", "models/voice_cloning"), "voice_profiles.json")
        
        with open(profiles_path, 'w') as f:
            json.dump(self.voice_profiles, f, indent=2)
    
    def extract_voice_features(self, audio_path: str) -> Optional[Dict]:
        """Extract voice characteristics from audio sample"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if len(audio) < self.sample_rate:  # Less than 1 second
                return None
            
            # Extract fundamental frequency (pitch)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # Calculate average F0 (removing NaN values)
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) == 0:
                return None
                
            avg_f0 = np.mean(f0_clean)
            f0_std = np.std(f0_clean)
            
            # Extract spectral features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            
            # Estimate formants (simplified)
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Estimate first two formants by finding peaks in average spectrum
            avg_magnitude = np.mean(magnitude, axis=1)
            
            # Find peaks in the spectrum (simplified formant estimation)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(avg_magnitude, height=np.max(avg_magnitude) * 0.1)
            
            formant_freqs = freqs[peaks][:4] if len(peaks) >= 4 else list(freqs[peaks]) + [0] * (4 - len(peaks))
            
            # Calculate speaking rate (very basic)
            # Count zero crossings as a proxy for speaking rate
            zero_crossings = librosa.zero_crossings(audio, pad=False)
            speaking_rate = np.sum(zero_crossings) / len(audio) * sr
            
            features = {
                "fundamental_freq": float(avg_f0),
                "f0_std": float(f0_std),
                "formants": [float(f) for f in formant_freqs],
                "mfcc_mean": np.mean(mfccs, axis=1).tolist(),
                "spectral_centroid": float(np.mean(spectral_centroid)),
                "spectral_rolloff": float(np.mean(spectral_rolloff)),
                "speaking_rate": float(speaking_rate),
                "voice_quality": "normal"  # Would need more sophisticated analysis
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting voice features from {audio_path}: {e}")
            return None
    
    def train_voice_profile(self, person_id: str, audio_samples: List[str]) -> bool:
        """Train a voice profile from multiple audio samples"""
        features_list = []
        
        for audio_path in audio_samples:
            features = self.extract_voice_features(audio_path)
            if features:
                features_list.append(features)
        
        if not features_list:
            print(f"No valid audio features extracted for person {person_id}")
            return False
        
        # Average features across all samples
        avg_features = {
            "fundamental_freq": np.mean([f["fundamental_freq"] for f in features_list]),
            "formants": np.mean([f["formants"] for f in features_list], axis=0).tolist(),
            "voice_quality": features_list[0]["voice_quality"],  # Use first sample's quality
            "speaking_rate": np.mean([f["speaking_rate"] for f in features_list]),
            "sample_count": len(features_list)
        }
        
        # Store voice profile
        self.voice_profiles[person_id] = avg_features
        self._save_voice_profiles()
        
        print(f"Trained voice profile for {person_id} using {len(features_list)} samples")
        return True
    
    def synthesize_speech(self, text: str, person_id: str) -> Optional[str]:
        """Synthesize speech with the voice profile of a specific person"""
        if person_id not in self.voice_profiles:
            print(f"No voice profile found for person {person_id}")
            return None
        
        voice_profile = self.voice_profiles[person_id]
        
        # Create temporary output file
        output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = output_file.name
        output_file.close()
        
        try:
            if self.tts_available:
                # Use espeak with voice modifications
                cmd = [
                    'espeak',
                    '-s', str(int(voice_profile["speaking_rate"] * 150)),  # Speaking rate
                    '-p', str(int(voice_profile["fundamental_freq"] / 2)),  # Pitch
                    '-w', output_path,  # Output to WAV file
                    text
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    # Apply voice profile modifications
                    self._apply_voice_profile(output_path, voice_profile)
                    return output_path
                else:
                    print(f"espeak failed: {result.stderr}")
                    
            else:
                # Fallback: Create a simple synthesized audio (for demo)
                self._create_simple_synthesis(text, output_path, voice_profile)
                return output_path
                
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
            
        # Clean up if failed
        if os.path.exists(output_path):
            os.unlink(output_path)
        
        return None
    
    def _apply_voice_profile(self, audio_path: str, voice_profile: Dict):
        """Apply voice profile characteristics to synthesized audio"""
        try:
            # Load the synthesized audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Apply pitch shifting to match fundamental frequency
            target_f0 = voice_profile["fundamental_freq"]
            
            # Simple pitch shifting (in production, use more sophisticated methods)
            if target_f0 > 150:  # Higher pitch
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
            elif target_f0 < 100:  # Lower pitch
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
            
            # Apply formant shifting (simplified)
            # In production, you would use more sophisticated formant manipulation
            
            # Save modified audio
            sf.write(audio_path, audio, sr)
            
        except Exception as e:
            print(f"Error applying voice profile: {e}")
    
    def _create_simple_synthesis(self, text: str, output_path: str, voice_profile: Dict):
        """Create simple synthesized audio (fallback method)"""
        # This is a very basic implementation for demo purposes
        # In production, you would use sophisticated TTS models
        
        # Generate a simple tone sequence based on text length
        duration = len(text) * 0.1  # 0.1 seconds per character
        
        # Create a simple tone at the person's fundamental frequency
        f0 = voice_profile["fundamental_freq"]
        t = np.linspace(0, duration, int(duration * self.sample_rate), False)
        
        # Create a simple synthesized voice with harmonics
        audio = np.sin(2 * np.pi * f0 * t)  # Fundamental
        audio += 0.5 * np.sin(2 * np.pi * f0 * 2 * t)  # Second harmonic
        audio += 0.3 * np.sin(2 * np.pi * f0 * 3 * t)  # Third harmonic
        
        # Add some formant-like filtering (very simplified)
        from scipy.signal import butter, filtfilt
        
        formants = voice_profile["formants"]
        for formant in formants[:2]:  # Use first two formants
            if formant > 0:
                # Create bandpass filter around formant frequency
                nyquist = self.sample_rate // 2
                low = max(50, formant - 200) / nyquist
                high = min(nyquist - 1, formant + 200) / nyquist
                
                if low < high:
                    b, a = butter(2, [low, high], btype='band')
                    formant_signal = filtfilt(b, a, audio)
                    audio += 0.3 * formant_signal
        
        # Normalize and add some envelope
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        # Add simple envelope
        envelope = np.ones_like(audio)
        fade_samples = int(0.1 * self.sample_rate)  # 0.1 second fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio *= envelope
        
        # Save audio
        sf.write(output_path, audio, self.sample_rate)
    
    def get_voice_profile(self, person_id: str) -> Optional[Dict]:
        """Get voice profile for a person"""
        return self.voice_profiles.get(person_id)
    
    def list_voice_profiles(self) -> List[str]:
        """List all available voice profiles"""
        return list(self.voice_profiles.keys())

class VoiceCloningService:
    """Main service for voice cloning operations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.voice_cloning = VoiceCloning(config)
        self.min_audio_duration = config.get("min_audio_duration", 5.0)
    
    def add_person_voice(self, person_id: str, audio_files: List[str]) -> bool:
        """Add voice samples for a person"""
        # Filter audio files by minimum duration
        valid_audio_files = []
        
        for audio_file in audio_files:
            try:
                duration = librosa.get_duration(path=audio_file)
                if duration >= self.min_audio_duration:
                    valid_audio_files.append(audio_file)
                else:
                    print(f"Audio file {audio_file} too short ({duration:.1f}s), skipping")
            except Exception as e:
                print(f"Error checking duration of {audio_file}: {e}")
        
        if not valid_audio_files:
            print(f"No valid audio files for person {person_id}")
            return False
        
        return self.voice_cloning.train_voice_profile(person_id, valid_audio_files)
    
    def synthesize_speech(self, text: str, person_id: str) -> Optional[str]:
        """Synthesize speech for a person"""
        return self.voice_cloning.synthesize_speech(text, person_id)
    
    def has_voice_profile(self, person_id: str) -> bool:
        """Check if voice profile exists for person"""
        return person_id in self.voice_cloning.voice_profiles
    
    def get_voice_info(self, person_id: str) -> Optional[Dict]:
        """Get voice profile information"""
        return self.voice_cloning.get_voice_profile(person_id)
    
    def list_voices(self) -> List[str]:
        """List all available voice profiles"""
        return self.voice_cloning.list_voice_profiles()