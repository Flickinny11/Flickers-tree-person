# Mouthful - Live Lip Reading with Voice Cloning

Mouthful is an advanced AI service that provides real-time lip reading with voice cloning capabilities. It identifies people in videos through facial recognition, finds their audio samples online, clones their voice, and synthesizes speech from lip movements.

## Features

- **Real-time Lip Reading**: Advanced deep learning models for lip movement analysis
- **Face Recognition**: Biometric face scanning and identity matching
- **Voice Cloning**: AI-powered voice synthesis from discovered audio samples
- **Social Media Integration**: Automated identity verification through social media scanning
- **Audio Discovery**: Web crawling to find audio samples for voice training
- **Live Video Processing**: Support for live feeds with processing delay

## Architecture

The system consists of several integrated modules:

1. **Face Detection & Recognition**: Uses state-of-the-art face recognition models
2. **Lip Reading Engine**: Deep learning models for visual speech recognition
3. **Voice Cloning Service**: Real-time voice synthesis and cloning
4. **Identity Search**: Social media and web scraping for identity verification
5. **Audio Crawler**: Automated discovery of audio samples for voice training
6. **Processing Pipeline**: Orchestrates the entire workflow

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web API
```bash
python main.py
```

### Process Video File
```python
from mouthful import MouthfulProcessor

processor = MouthfulProcessor()
result = processor.process_video("input_video.mp4")
```

### Live Stream Processing
```python
processor = MouthfulProcessor()
processor.process_live_stream("rtmp://stream-url", delay_seconds=180)
```

## Configuration

Edit `config.json` to customize:
- Model paths and configurations
- API endpoints and keys
- Processing parameters
- Output settings

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for educational and research purposes. Ensure compliance with privacy laws and obtain proper consent before processing personal data.