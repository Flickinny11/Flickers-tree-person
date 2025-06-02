"""
Social Media Search Module for Mouthful
Searches social media platforms to identify persons from face images
"""

import requests
import time
import json
from typing import List, Dict, Optional, Tuple
import os
from urllib.parse import urljoin, urlparse
import hashlib
import base64
from dataclasses import dataclass
import numpy as np

@dataclass
class SocialMediaProfile:
    """Represents a social media profile match"""
    platform: str
    profile_url: str
    username: str
    display_name: str
    confidence: float
    profile_image_url: Optional[str] = None
    additional_info: Optional[Dict] = None

class SocialMediaSearchService:
    """Service for searching social media platforms for identity matches"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.platforms = config.get("platforms", ["twitter", "facebook", "instagram", "linkedin"])
        self.max_profiles_per_platform = config.get("max_profiles_per_platform", 5)
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.cache_dir = "cache/social_media"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Rate limiting
        self.request_delays = {
            "twitter": 1.0,
            "facebook": 2.0,
            "instagram": 1.5,
            "linkedin": 1.0
        }
        self.last_request_time = {}
    
    def search_by_face_encoding(self, face_encoding: np.ndarray, max_results: int = 20) -> List[SocialMediaProfile]:
        """
        Search social media platforms using face encoding
        Note: This is a simplified implementation for demonstration
        """
        all_profiles = []
        
        # Generate a hash for caching
        encoding_hash = hashlib.md5(face_encoding.tobytes()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"search_{encoding_hash}.json")
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if time.time() - cached_data.get("timestamp", 0) < 3600:  # 1 hour cache
                        print("Using cached social media search results")
                        return [SocialMediaProfile(**profile) for profile in cached_data["profiles"]]
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Search each platform
        for platform in self.platforms:
            try:
                profiles = self._search_platform(platform, face_encoding)
                all_profiles.extend(profiles)
                
                # Rate limiting
                if platform in self.request_delays:
                    time.sleep(self.request_delays[platform])
                    
            except Exception as e:
                print(f"Error searching {platform}: {e}")
        
        # Sort by confidence and limit results
        all_profiles.sort(key=lambda x: x.confidence, reverse=True)
        all_profiles = all_profiles[:max_results]
        
        # Cache results
        try:
            cache_data = {
                "timestamp": time.time(),
                "profiles": [self._profile_to_dict(profile) for profile in all_profiles]
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
        
        return all_profiles
    
    def _search_platform(self, platform: str, face_encoding: np.ndarray) -> List[SocialMediaProfile]:
        """Search a specific platform (simplified implementation)"""
        # In a real implementation, this would use platform-specific APIs
        # or scraping methods. For this demo, we'll return mock data.
        
        print(f"Searching {platform} for face matches...")
        
        # Simulate some matches with varying confidence
        mock_profiles = []
        
        if platform == "twitter":
            mock_profiles = [
                SocialMediaProfile(
                    platform="twitter",
                    profile_url="https://twitter.com/example_user1",
                    username="example_user1",
                    display_name="John Example",
                    confidence=0.85,
                    profile_image_url="https://example.com/avatar1.jpg"
                ),
                SocialMediaProfile(
                    platform="twitter",
                    profile_url="https://twitter.com/example_user2",
                    username="example_user2", 
                    display_name="Jane Sample",
                    confidence=0.72,
                    profile_image_url="https://example.com/avatar2.jpg"
                )
            ]
        elif platform == "facebook":
            mock_profiles = [
                SocialMediaProfile(
                    platform="facebook",
                    profile_url="https://facebook.com/example.profile",
                    username="example.profile",
                    display_name="Example Person",
                    confidence=0.78,
                    profile_image_url="https://example.com/fb_avatar.jpg"
                )
            ]
        elif platform == "instagram":
            mock_profiles = [
                SocialMediaProfile(
                    platform="instagram",
                    profile_url="https://instagram.com/example_ig",
                    username="example_ig",
                    display_name="Example Instagram",
                    confidence=0.68,
                    profile_image_url="https://example.com/ig_avatar.jpg"
                )
            ]
        elif platform == "linkedin":
            mock_profiles = [
                SocialMediaProfile(
                    platform="linkedin",
                    profile_url="https://linkedin.com/in/example-professional",
                    username="example-professional",
                    display_name="Example Professional",
                    confidence=0.83,
                    profile_image_url="https://example.com/li_avatar.jpg"
                )
            ]
        
        # Filter by confidence threshold
        return [p for p in mock_profiles if p.confidence >= self.confidence_threshold]
    
    def _profile_to_dict(self, profile: SocialMediaProfile) -> Dict:
        """Convert profile to dictionary for caching"""
        return {
            "platform": profile.platform,
            "profile_url": profile.profile_url,
            "username": profile.username,
            "display_name": profile.display_name,
            "confidence": profile.confidence,
            "profile_image_url": profile.profile_image_url,
            "additional_info": profile.additional_info
        }
    
    def verify_profile_match(self, profile: SocialMediaProfile, face_encoding: np.ndarray) -> float:
        """
        Verify if a social media profile matches the face encoding
        Returns updated confidence score
        """
        # In a real implementation, this would download the profile image
        # and compare it with the face encoding
        
        try:
            if profile.profile_image_url:
                # Mock verification - in reality, download image and compare faces
                print(f"Verifying profile image for {profile.username} on {profile.platform}")
                
                # Simulate face comparison
                # In real implementation: download image, detect face, compare encodings
                verification_confidence = min(1.0, profile.confidence + 0.1)
                
                return verification_confidence
            else:
                return profile.confidence
                
        except Exception as e:
            print(f"Error verifying profile {profile.username}: {e}")
            return profile.confidence * 0.8  # Reduce confidence if verification fails
    
    def extract_profile_info(self, profile: SocialMediaProfile) -> Dict:
        """Extract additional information from social media profile"""
        # In a real implementation, this would scrape or use APIs to get:
        # - Bio/description
        # - Posts/content
        # - Associated media
        # - Contact information
        
        mock_info = {
            "bio": f"Profile information for {profile.display_name}",
            "follower_count": 1000,
            "following_count": 500,
            "post_count": 250,
            "verified": False,
            "location": "Unknown",
            "website": None,
            "joined_date": "2020-01-01"
        }
        
        return mock_info
    
    def search_by_name(self, name: str, additional_info: Optional[Dict] = None) -> List[SocialMediaProfile]:
        """Search social media platforms by name"""
        all_profiles = []
        
        for platform in self.platforms:
            try:
                profiles = self._search_platform_by_name(platform, name, additional_info)
                all_profiles.extend(profiles)
                
                # Rate limiting
                if platform in self.request_delays:
                    time.sleep(self.request_delays[platform])
                    
            except Exception as e:
                print(f"Error searching {platform} by name: {e}")
        
        return all_profiles
    
    def _search_platform_by_name(self, platform: str, name: str, additional_info: Optional[Dict] = None) -> List[SocialMediaProfile]:
        """Search a platform by name (simplified implementation)"""
        print(f"Searching {platform} for name: {name}")
        
        # Mock search results
        mock_profiles = [
            SocialMediaProfile(
                platform=platform,
                profile_url=f"https://{platform}.com/{name.lower().replace(' ', '_')}",
                username=name.lower().replace(' ', '_'),
                display_name=name,
                confidence=0.7,
                profile_image_url=f"https://example.com/{platform}_avatar.jpg"
            )
        ]
        
        return mock_profiles
    
    def get_cached_searches(self) -> List[str]:
        """Get list of cached searches"""
        cache_files = []
        try:
            for file in os.listdir(self.cache_dir):
                if file.startswith("search_") and file.endswith(".json"):
                    cache_files.append(file)
        except Exception as e:
            print(f"Error listing cache files: {e}")
        
        return cache_files
    
    def clear_cache(self):
        """Clear search cache"""
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith(".json"):
                    os.remove(os.path.join(self.cache_dir, file))
            print("Social media search cache cleared")
        except Exception as e:
            print(f"Error clearing cache: {e}")

class IdentityResolutionService:
    """Service for resolving identities from multiple sources"""
    
    def __init__(self, social_media_service: SocialMediaSearchService):
        self.social_media_service = social_media_service
    
    def resolve_identity(self, face_encoding: np.ndarray, name: Optional[str] = None) -> Optional[Dict]:
        """
        Resolve identity from face encoding and optional name
        Returns consolidated identity information
        """
        identity_info = {
            "confidence": 0.0,
            "primary_name": None,
            "alternative_names": [],
            "social_profiles": [],
            "verified": False,
            "sources": []
        }
        
        # Search by face encoding
        face_profiles = self.social_media_service.search_by_face_encoding(face_encoding)
        
        # Search by name if provided
        name_profiles = []
        if name:
            name_profiles = self.social_media_service.search_by_name(name)
        
        # Combine and deduplicate profiles
        all_profiles = face_profiles + name_profiles
        unique_profiles = self._deduplicate_profiles(all_profiles)
        
        if not unique_profiles:
            return None
        
        # Calculate overall confidence
        identity_info["confidence"] = max(p.confidence for p in unique_profiles)
        identity_info["social_profiles"] = unique_profiles
        
        # Extract names
        names = [p.display_name for p in unique_profiles if p.display_name]
        if names:
            # Use most confident profile's name as primary
            best_profile = max(unique_profiles, key=lambda x: x.confidence)
            identity_info["primary_name"] = best_profile.display_name
            identity_info["alternative_names"] = list(set(names) - {best_profile.display_name})
        
        # Mark as verified if confidence is high enough
        identity_info["verified"] = identity_info["confidence"] >= 0.85
        
        # Track sources
        identity_info["sources"] = list(set(p.platform for p in unique_profiles))
        
        return identity_info
    
    def _deduplicate_profiles(self, profiles: List[SocialMediaProfile]) -> List[SocialMediaProfile]:
        """Remove duplicate profiles based on username and platform"""
        seen = set()
        unique_profiles = []
        
        for profile in profiles:
            key = (profile.platform, profile.username)
            if key not in seen:
                seen.add(key)
                unique_profiles.append(profile)
        
        return unique_profiles