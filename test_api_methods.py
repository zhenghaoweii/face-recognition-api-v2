#!/usr/bin/env python3
"""
Test script to verify all API input methods work correctly
"""

import requests
import base64
import os
from pathlib import Path

API_URL = "http://localhost:5000"

def test_health_endpoint():
    """Test the health endpoint"""
    print("=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_multipart_file_upload():
    """Test multipart form-data file upload"""
    print("\n=== Testing Multipart File Upload ===")
    
    image_path = "./haowei-3.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': ('haowei-3.jpg', f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/verify", files=files)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Image source: {result.get('image_source', 'unknown')}")
            print(f"Faces detected: {result.get('total_faces_detected', 0)}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Multipart upload failed: {e}")
        return False

def test_json_base64():
    """Test JSON with base64 encoded image"""
    print("\n=== Testing JSON Base64 ===")
    
    image_path = "./haowei-3.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False
    
    try:
        # Convert image to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            base64_string = f"data:image/jpeg;base64,{image_data}"
        
        # Send as JSON
        json_data = {"img1": base64_string}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f"{API_URL}/verify", json=json_data, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Image source: {result.get('image_source', 'unknown')}")
            print(f"Faces detected: {result.get('total_faces_detected', 0)}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ JSON base64 failed: {e}")
        return False

def test_form_data_base64():
    """Test form-data with base64 encoded image"""
    print("\n=== Testing Form-Data Base64 ===")
    
    image_path = "./haowei-3.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False
    
    try:
        # Convert image to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            base64_string = f"data:image/jpeg;base64,{image_data}"
        
        # Send as form data
        form_data = {"img1": base64_string}
        response = requests.post(f"{API_URL}/verify", data=form_data)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Image source: {result.get('image_source', 'unknown')}")
            print(f"Faces detected: {result.get('total_faces_detected', 0)}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Form-data base64 failed: {e}")
        return False

def test_no_input():
    """Test what happens when no input is provided"""
    print("\n=== Testing No Input (Should Fail) ===")
    
    try:
        response = requests.post(f"{API_URL}/verify", json={})
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 400:
            result = response.json()
            print(f"✅ Correctly rejected: {result.get('error', 'unknown error')}")
            return True
        else:
            print(f"❌ Unexpected response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ No input test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting API Tests...")
    print("Make sure the API server is running on http://localhost:5000")
    print()
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Multipart File Upload", test_multipart_file_upload),
        ("JSON Base64", test_json_base64),
        ("Form-Data Base64", test_form_data_base64),
        ("No Input", test_no_input)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS:")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the API implementation.")

if __name__ == "__main__":
    main()
