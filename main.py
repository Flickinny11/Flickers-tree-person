"""
Main application entry point for Mouthful
FastAPI-based REST API for the lip reading service
"""

import os
import json
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import tempfile
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from mouthful.processor import MouthfulProcessor, LiveStreamProcessor, ProcessingResult

# Pydantic models for API
class ProcessVideoRequest(BaseModel):
    video_url: Optional[str] = None
    process_live: bool = False
    delay_seconds: Optional[int] = None

class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # 'queued', 'processing', 'completed', 'failed'
    progress: float
    results: Optional[List[Dict]] = None
    error: Optional[str] = None

class PersonInfo(BaseModel):
    person_id: str
    name: str
    confidence: float
    voice_trained: bool

class AddPersonRequest(BaseModel):
    name: str
    person_id: Optional[str] = None

# Global variables
app = FastAPI(
    title="Mouthful API",
    description="Lip Reading with Voice Cloning Service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instance
processor: Optional[MouthfulProcessor] = None
live_processor: Optional[LiveStreamProcessor] = None
processing_tasks: Dict[str, ProcessingStatus] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the processor on startup"""
    global processor, live_processor
    
    print("Starting Mouthful API...")
    
    try:
        # Load configuration
        config_path = "config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found")
        
        # Initialize processor
        processor = MouthfulProcessor(config_path)
        live_processor = LiveStreamProcessor(processor)
        
        print("Mouthful API started successfully")
        
    except Exception as e:
        print(f"Error starting Mouthful API: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Mouthful API",
        "version": "1.0.0",
        "description": "Lip Reading with Voice Cloning Service",
        "status": "running",
        "endpoints": {
            "process_video": "/process/video",
            "process_live": "/process/live",
            "upload_video": "/upload/video",
            "add_person": "/persons/add",
            "list_persons": "/persons",
            "statistics": "/stats",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    stats = processor.get_statistics()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "statistics": stats
    }

@app.post("/upload/video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a video file"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate task ID
    task_id = f"video_{int(time.time())}_{hash(file.filename)}"
    
    # Create processing status
    processing_tasks[task_id] = ProcessingStatus(
        task_id=task_id,
        status="queued",
        progress=0.0
    )
    
    # Save uploaded file
    try:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, f"{task_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Start background processing
        background_tasks.add_task(process_video_background, task_id, file_path)
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": f"Video upload successful. Processing started.",
            "filename": file.filename
        }
        
    except Exception as e:
        processing_tasks[task_id].status = "failed"
        processing_tasks[task_id].error = str(e)
        raise HTTPException(status_code=500, detail=f"Error processing video: {e}")

async def process_video_background(task_id: str, video_path: str):
    """Background task for video processing"""
    try:
        # Update status
        processing_tasks[task_id].status = "processing"
        processing_tasks[task_id].progress = 0.1
        
        # Process video
        results = processor.process_video_file(video_path)
        
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
        
        # Update status
        processing_tasks[task_id].status = "completed"
        processing_tasks[task_id].progress = 1.0
        processing_tasks[task_id].results = serializable_results
        
        # Clean up temporary file
        if os.path.exists(video_path):
            os.remove(video_path)
        
    except Exception as e:
        processing_tasks[task_id].status = "failed"
        processing_tasks[task_id].error = str(e)
        print(f"Error in background processing for task {task_id}: {e}")

@app.get("/process/status/{task_id}")
async def get_processing_status(task_id: str):
    """Get processing status for a task"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return processing_tasks[task_id]

@app.post("/process/live")
async def start_live_processing(request: ProcessVideoRequest):
    """Start live stream processing"""
    if processor is None or live_processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    if not request.video_url:
        raise HTTPException(status_code=400, detail="video_url is required for live processing")
    
    try:
        # Configure delay if specified
        if request.delay_seconds:
            processor.processing_delay = request.delay_seconds
        
        # Start live processing in background
        asyncio.create_task(start_live_stream(request.video_url))
        
        return {
            "status": "started",
            "stream_url": request.video_url,
            "delay_seconds": processor.processing_delay,
            "message": "Live stream processing started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting live processing: {e}")

async def start_live_stream(stream_url: str):
    """Start live stream processing in background"""
    try:
        live_processor.start_processing(stream_url)
    except Exception as e:
        print(f"Error in live stream processing: {e}")

@app.post("/process/live/stop")
async def stop_live_processing():
    """Stop live stream processing"""
    if live_processor is None:
        raise HTTPException(status_code=503, detail="Live processor not initialized")
    
    live_processor.stop_processing()
    
    return {
        "status": "stopped",
        "message": "Live stream processing stopped"
    }

@app.post("/persons/add")
async def add_person(request: AddPersonRequest, 
                    images: List[UploadFile] = File(...),
                    audio_files: Optional[List[UploadFile]] = File(None)):
    """Add a new person to the system"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    try:
        # Generate person ID if not provided
        person_id = request.person_id or f"person_{int(time.time())}_{hash(request.name)}"
        
        # Process uploaded images
        image_arrays = []
        for image_file in images:
            if not image_file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {image_file.filename} is not an image")
            
            # Read image
            content = await image_file.read()
            
            # Save temporarily and load with OpenCV
            temp_path = f"temp/image_{int(time.time())}_{image_file.filename}"
            os.makedirs("temp", exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(content)
            
            import cv2
            image = cv2.imread(temp_path)
            if image is not None:
                image_arrays.append(image)
            
            # Clean up
            os.remove(temp_path)
        
        # Process audio files if provided
        audio_paths = []
        if audio_files:
            for audio_file in audio_files:
                # Save audio file temporarily
                temp_audio_path = f"temp/audio_{int(time.time())}_{audio_file.filename}"
                
                content = await audio_file.read()
                with open(temp_audio_path, "wb") as f:
                    f.write(content)
                
                audio_paths.append(temp_audio_path)
        
        # Add person to system
        success = processor.add_known_person(person_id, request.name, image_arrays, audio_paths)
        
        # Clean up temporary audio files
        for audio_path in audio_paths:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
        if success:
            return {
                "person_id": person_id,
                "name": request.name,
                "status": "added",
                "voice_trained": bool(audio_paths),
                "message": f"Person {request.name} added successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add person")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding person: {e}")

@app.get("/persons")
async def list_persons():
    """List all known persons in the system"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        persons = []
        
        # Get persons from face recognition database
        for person_id, person_data in processor.face_recognition.known_faces.items():
            voice_trained = processor.voice_cloning.has_voice_profile(person_id)
            
            person_info = PersonInfo(
                person_id=person_id,
                name=person_data["name"],
                confidence=1.0,  # Known persons have high confidence
                voice_trained=voice_trained
            )
            persons.append(person_info)
        
        # Add persons from known_persons cache
        for person_id, identity_info in processor.known_persons.items():
            if person_id not in processor.face_recognition.known_faces:
                person_info = PersonInfo(
                    person_id=person_id,
                    name=identity_info.get("primary_name", "Unknown"),
                    confidence=identity_info.get("confidence", 0.0),
                    voice_trained=identity_info.get("voice_trained", False)
                )
                persons.append(person_info)
        
        return {
            "total_persons": len(persons),
            "persons": persons
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing persons: {e}")

@app.get("/persons/{person_id}")
async def get_person_info(person_id: str):
    """Get detailed information about a specific person"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        # Check face recognition database
        if person_id in processor.face_recognition.known_faces:
            person_data = processor.face_recognition.known_faces[person_id]
            voice_info = processor.voice_cloning.get_voice_info(person_id)
            
            return {
                "person_id": person_id,
                "name": person_data["name"],
                "face_encodings_count": len(person_data["encodings"]),
                "voice_trained": voice_info is not None,
                "voice_info": voice_info,
                "source": "face_database"
            }
        
        # Check known persons cache
        if person_id in processor.known_persons:
            identity_info = processor.known_persons[person_id]
            voice_info = processor.voice_cloning.get_voice_info(person_id)
            
            return {
                "person_id": person_id,
                "identity_info": identity_info,
                "voice_info": voice_info,
                "source": "identity_resolution"
            }
        
        raise HTTPException(status_code=404, detail="Person not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting person info: {e}")

@app.post("/synthesize/{person_id}")
async def synthesize_speech(person_id: str, text: str):
    """Synthesize speech for a specific person"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        if not processor.voice_cloning.has_voice_profile(person_id):
            raise HTTPException(status_code=404, detail="Voice profile not found for this person")
        
        audio_path = processor.voice_cloning.synthesize_speech(text, person_id)
        
        if audio_path and os.path.exists(audio_path):
            return FileResponse(
                audio_path,
                media_type="audio/wav",
                filename=f"synthesized_{person_id}.wav"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error synthesizing speech: {e}")

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        stats = processor.get_statistics()
        
        # Add API-specific stats
        stats["api"] = {
            "active_tasks": len([t for t in processing_tasks.values() if t.status == "processing"]),
            "total_tasks": len(processing_tasks),
            "live_processing_active": live_processor.is_running if live_processor else False
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {e}")

@app.post("/reset")
async def reset_system():
    """Reset processing buffers and cache"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        processor.reset_buffers()
        
        # Clear completed tasks
        global processing_tasks
        processing_tasks = {
            k: v for k, v in processing_tasks.items() 
            if v.status in ["queued", "processing"]
        }
        
        return {
            "status": "reset",
            "message": "System buffers and cache reset successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting system: {e}")

if __name__ == "__main__":
    # Load configuration
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            api_config = config.get("api", {})
    else:
        api_config = {}
    
    # Run the API server
    uvicorn.run(
        app,
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        debug=api_config.get("debug", False)
    )