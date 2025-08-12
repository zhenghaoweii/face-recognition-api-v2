from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Union
import hashlib
import os
import io
import cv2
import base64
import requests
import numpy as np
from pathlib import Path
from PIL import Image

from deepface import DeepFace

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


def load_image_from_base64(uri: str) -> np.ndarray:
    """
    Load image from base64 string.
    Args:
        uri: a base64 string.
    Returns:
        numpy array: the loaded image.
    """
    encoded_data_parts = uri.split(",")

    if len(encoded_data_parts) < 2:
        raise ValueError("format error in base64 encoded string")

    encoded_data = encoded_data_parts[1]
    decoded_bytes = base64.b64decode(encoded_data)

    # similar to find functionality, we are just considering these extensions
    # content type is safer option than file extension
    with Image.open(io.BytesIO(decoded_bytes)) as img:
        file_type = img.format.lower()
        if file_type not in {"jpeg", "png"}:
            raise ValueError(f"Input image can be jpg or png, but it is {file_type}")

    nparr = np.frombuffer(decoded_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_bgr


def load_image_from_file(file_storage) -> np.ndarray:
    """
    Load image from uploaded file.
    Args:
        file_storage: Flask FileStorage object
    Returns:
        numpy array: the loaded image.
    """
    if not file_storage or file_storage.filename == '':
        raise ValueError("No file provided")
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    file_extension = file_storage.filename.rsplit('.', 1)[1].lower() if '.' in file_storage.filename else ''
    
    if file_extension not in allowed_extensions:
        raise ValueError(f"File type '{file_extension}' not supported. Use: {', '.join(allowed_extensions)}")
    
    # Read file bytes
    file_bytes = file_storage.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("Could not decode image file")
    
    return img_bgr


def get_image_input(request) -> tuple[np.ndarray, str]:
    """
    Get image input from various sources with fallback.
    Args:
        request: Flask request object
    Returns:
        tuple: (image_array, source_description)
    """
    print(f"DEBUG: Content-Type: {request.content_type}")
    print(f"DEBUG: Has files: {bool(request.files)}")
    print(f"DEBUG: Files keys: {list(request.files.keys()) if request.files else []}")
    print(f"DEBUG: Is JSON: {request.is_json}")
    print(f"DEBUG: Has form: {bool(request.form)}")
    print(f"DEBUG: Form keys: {list(request.form.keys()) if request.form else []}")
    
    # Priority 1: Check for multipart form data (file upload)
    if request.files and 'image' in request.files:
        try:
            print("DEBUG: Attempting to load from uploaded file")
            img_array = load_image_from_file(request.files['image'])
            return img_array, f"uploaded file: {request.files['image'].filename}"
        except Exception as e:
            print(f"Failed to load from uploaded file: {e}")
    
    # Priority 2: Check for JSON with base64 image
    if request.is_json:
        data = request.get_json()
        img_data = data.get('img1') or data.get('image')
        print(f"DEBUG: JSON data keys: {list(data.keys()) if data else []}")
        
        if img_data and isinstance(img_data, str) and img_data.startswith('data:image'):
            try:
                print("DEBUG: Attempting to load from JSON base64")
                img_array = load_image_from_base64(img_data)
                return img_array, "base64 encoded image from JSON"
            except Exception as e:
                print(f"Failed to load from JSON base64: {e}")
    
    # Priority 3: Check for form data with base64
    if request.form and ('img1' in request.form or 'image' in request.form):
        img_data = request.form.get('img1') or request.form.get('image')
        print(f"DEBUG: Form data found, img_data length: {len(img_data) if img_data else 0}")
        print(f"DEBUG: Form data starts with data:image: {img_data.startswith('data:image') if img_data else False}")
        
        if img_data and isinstance(img_data, str) and img_data.startswith('data:image'):
            try:
                print("DEBUG: Attempting to load from form base64")
                img_array = load_image_from_base64(img_data)
                return img_array, "base64 encoded image from form data"
            except Exception as e:
                print(f"Failed to load from form base64: {e}")
    
    # No valid input found
    raise ValueError("No valid image input found. Please provide: file upload, base64 image, or ensure default image exists.")

@app.route("/verify", methods=["POST"])
def verify():
    """
    Face recognition endpoint that accepts multiple input formats:
    1. Multipart form data with file upload (field name: 'image')
    2. JSON with base64 encoded image (field: 'img1' or 'image') 
    3. Form data with base64 encoded image (field: 'img1' or 'image')
    
    Optional parameters:
    - anti_spoofing: Enable anti-spoofing detection (default: True)
    - enforce_detection: Require face detection (default: False)
    """
    try:
        # Get image input from various sources with fallback
        img_input, source_description = get_image_input(request)
        print(f"Using image from: {source_description}")
        
        # Get optional parameters
        anti_spoofing = True  # Default to True for security
        enforce_detection = False
        
        # Check for parameters in JSON data
        if request.is_json:
            data = request.get_json()
            anti_spoofing = data.get('anti_spoofing', True)
        
        # Check for parameters in form data
        elif request.form:
            anti_spoofing = request.form.get('anti_spoofing', 'true').lower() == 'true'
            enforce_detection = request.form.get('enforce_detection', 'false').lower() == 'true'
        
        print(f"Using anti_spoofing: {anti_spoofing}, enforce_detection: {enforce_detection}")
        
        # Use the provided image path from the request
        find_results = DeepFace.find(
            img_path=img_input, 
            model_name="Facenet", 
            db_path="./user/database", 
            anti_spoofing=anti_spoofing,
            enforce_detection=enforce_detection
        )

        print(find_results);

        print(f"Found {len(find_results)} face(s) in source image")
        
        # DeepFace.find returns a list of pandas DataFrames
        # Each DataFrame corresponds to a detected face in the source image
        response_data = []
        
        for i, df in enumerate(find_results):
            face_data = {
                "face_index": i,
                "matches_found": len(df),
                "matches": []
            }
            
            if not df.empty:
                # Convert DataFrame to list of dictionaries for JSON serialization
                for _, row in df.iterrows():
                    match = {
                        "identity": row.get('identity', ''),
                        "distance": float(row.get('distance', 0)),
                        "confidence": float(row.get('confidence', 0)),
                        "threshold": float(row.get('threshold', 0)),
                        "verified": row.get('distance', float('inf')) <= row.get('threshold', 0),
                        "target_face_area": {
                            "x": int(row.get('target_x', 0)),
                            "y": int(row.get('target_y', 0)),
                            "w": int(row.get('target_w', 0)),
                            "h": int(row.get('target_h', 0))
                        },
                        "source_face_area": {
                            "x": int(row.get('source_x', 0)),
                            "y": int(row.get('source_y', 0)),
                            "w": int(row.get('source_w', 0)),
                            "h": int(row.get('source_h', 0))
                        }
                    }
                    face_data["matches"].append(match)
            
            response_data.append(face_data)
        
        return jsonify({
            "success": True,
            "image_source": source_description,
            "anti_spoofing_enabled": anti_spoofing,
            "enforce_detection": enforce_detection,
            "total_faces_detected": len(find_results),
            "results": response_data,
        })
        
    except ValueError as ve:
        error_message = str(ve)
        # Check if it's a spoofing detection error
        if "spoof detected" in error_message.lower():
            return jsonify({
                "success": False,
                "spoof_detected": True,
                "message": "Spoof detected",
                "details": "The image appears to be fake or artificially generated. Please use a real, live image.",
                "anti_spoofing_enabled": anti_spoofing if 'anti_spoofing' in locals() else True
            }), 400
        else:
            return jsonify({
                "success": False,
                "error": f"Input validation error: {error_message}"
            }), 400
    except Exception as e:
        error_message = str(e)
        # Check if it's a spoofing detection error from DeepFace
        if "spoof detected" in error_message.lower():
            return jsonify({
                "success": False,
                "spoof_detected": True,
                "message": "Spoof detected",
                "details": "The image appears to be fake or artificially generated. Please use a real, live image.",
                "anti_spoofing_enabled": anti_spoofing if 'anti_spoofing' in locals() else True
            }), 400
        else:
            return jsonify({
                "success": False,
                "error": f"Processing error: {error_message}"
            }), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Face recognition API is running",
        "supported_formats": [
            "multipart/form-data with 'image' field",
            "JSON with base64 'img1' or 'image' field", 
            "form-data with base64 'img1' or 'image' field",
            "fallback to default image"
        ]
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)