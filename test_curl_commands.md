# API Testing Commands

## 1. Health Check
```bash
curl -X GET http://localhost:5000/health
```

## 2. Multipart File Upload
```bash
curl -X POST http://localhost:5000/verify \
  -F "image=@./haowei-3.jpg"
```

## 3. JSON with Base64
First, encode your image to base64:
```bash
base64_img=$(base64 -i ./haowei-3.jpg)
curl -X POST http://localhost:5000/verify \
  -H "Content-Type: application/json" \
  -d "{\"img1\": \"data:image/jpeg;base64,$base64_img\"}"
```

## 4. Form Data with Base64
```bash
base64_img=$(base64 -i ./haowei-3.jpg)
curl -X POST http://localhost:5000/verify \
  -d "img1=data:image/jpeg;base64,$base64_img"
```

## 5. Alternative Form Data Field Name
```bash
base64_img=$(base64 -i ./haowei-3.jpg)
curl -X POST http://localhost:5000/verify \
  -d "image=data:image/jpeg;base64,$base64_img"
```

## 6. Test with Python Script
```bash
python test_api_methods.py
```

## Expected Response Format
```json
{
  "success": true,
  "image_source": "uploaded file: haowei-3.jpg",
  "total_faces_detected": 1,
  "results": [
    {
      "face_index": 0,
      "matches_found": 3,
      "matches": [
        {
          "identity": "./user/database/Haowei/haowei-1.jpg",
          "distance": 0.2345,
          "confidence": 87.65,
          "threshold": 0.6836,
          "verified": true,
          "target_face_area": {"x": 45, "y": 67, "w": 120, "h": 120},
          "source_face_area": {"x": 32, "y": 54, "w": 115, "h": 115}
        }
      ]
    }
  ]
}
```

## Debug Output
When running the API, you'll see debug output like:
```
DEBUG: Content-Type: multipart/form-data; boundary=...
DEBUG: Has files: True
DEBUG: Files keys: ['image']
DEBUG: Is JSON: False
DEBUG: Has form: False
DEBUG: Form keys: []
DEBUG: Attempting to load from uploaded file
Using image from: uploaded file: haowei-3.jpg
```
