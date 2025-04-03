# Enhanced Face Recognition System

A robust facial recognition application that works effectively with few training images. This system uses state-of-the-art face detection and recognition libraries to provide reliable facial recognition capabilities with minimal setup.

## Features

- **Face detection** using dlib's frontal face detector
- **Face recognition** using 128-dimension face embeddings (based on dlib)
- **Multiple enrollment methods**:
  - Add faces from image files
  - Capture faces directly from webcam
- **Real-time recognition** with confidence scores
- **Adjustable sensitivity** to control strictness of face matching
- **Persistent storage** of face data and trained models
- **User-friendly interface** with simple command-line menus

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- scikit-learn
- dlib
- face_recognition

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/charles1246/Face_Recognition_Python.git
   cd Face_Recognition_Python
   ```

2. Install the required dependencies:
   ```
   pip install numpy opencv-python scikit-learn dlib face_recognition
   ```
   
   Note: Installing `dlib` may require additional system dependencies. On Ubuntu/Debian:
   ```
   sudo apt-get install build-essential cmake
   sudo apt-get install libopenblas-dev liblapack-dev
   sudo apt-get install libx11-dev libgtk-3-dev
   ```

## Usage

Run the application with:
```
python face_recognition_app.py
```

## How It Works

1. **Face Detection**: The system uses dlib's HOG-based face detector to locate faces in images or video frames.

2. **Face Encoding**: Detected faces are processed to generate 128-dimensional embeddings that capture unique facial features.

3. **Face Recognition**: During recognition, new face embeddings are compared against the stored embeddings to find the closest match.

4. **Confidence Scoring**: Distance calculations between face embeddings determine confidence scores.