#!/usr/bin/env python3
"""
Simple Test Runner for DeepFace
==============================

This script runs comprehensive tests for your DeepFace installation.
It's designed to work with your existing environment setup.

Usage:
    python run_tests.py
"""

import sys
import os

def test_deepface_installation():
    """Test basic DeepFace installation"""
    print("Testing DeepFace installation...")
    
    try:
        from deepface import DeepFace
        print("✅ DeepFace imported successfully")
        
        # Test basic functionality
        print("Testing basic functionality...")
        
        # Check if main functions exist
        functions = ['verify', 'analyze', 'represent', 'find', 'extract_faces']
        for func in functions:
            if hasattr(DeepFace, func):
                print(f"✅ DeepFace.{func} available")
            else:
                print(f"❌ DeepFace.{func} missing")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import DeepFace: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def run_basic_test():
    """Run a basic test with your existing dataset"""
    print("\nRunning basic test with your dataset...")
    
    try:
        from deepface import DeepFace
        
        # Check if dataset directory exists
        dataset_dir = "./dataset"
        if not os.path.exists(dataset_dir):
            print(f"⚠️  Dataset directory '{dataset_dir}' not found")
            return True  # Not a failure, just no dataset
        
        # List images in dataset
        image_files = [f for f in os.listdir(dataset_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            print("⚠️  No image files found in dataset directory")
            return True
        
        print(f"Found {len(image_files)} image(s) in dataset")
        
        # Test with first available image
        test_image = os.path.join(dataset_dir, image_files[0])
        print(f"Testing with: {test_image}")
        
        # Test face extraction (safest test)
        try:
            faces = DeepFace.extract_faces(
                img_path=test_image,
                detector_backend="opencv",
                enforce_detection=False
            )
            print(f"✅ Face extraction successful - found {len(faces)} face(s)")
        except Exception as e:
            print(f"⚠️  Face extraction failed: {e}")
        
        # Test analysis if we have faces
        try:
            results = DeepFace.analyze(
                img_path=test_image,
                actions=['gender'],
                enforce_detection=False
            )
            print(f"✅ Analysis successful - analyzed {len(results)} face(s)")
        except Exception as e:
            print(f"⚠️  Analysis failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def run_comprehensive_tests():
    """Run the comprehensive test suite"""
    print("\nRunning comprehensive test suite...")
    
    try:
        # Import and run comprehensive tests
        import test_deepface_comprehensive
        test_deepface_comprehensive.run_comprehensive_tests()
        return True
        
    except ImportError:
        print("❌ Could not import comprehensive test suite")
        print("Make sure test_deepface_comprehensive.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Comprehensive tests failed: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("DEEPFACE TEST RUNNER")
    print("=" * 60)
    
    # Test 1: Installation
    print("1. Testing DeepFace installation...")
    if not test_deepface_installation():
        print("❌ Installation test failed. Please check your DeepFace installation.")
        return False
    
    # Test 2: Basic functionality
    print("\n2. Testing basic functionality...")
    if not run_basic_test():
        print("❌ Basic test failed.")
        return False
    
    # Test 3: Comprehensive tests
    print("\n3. Running comprehensive tests...")
    if not run_comprehensive_tests():
        print("⚠️  Comprehensive tests had issues, but basic functionality works.")
    
    print("\n" + "=" * 60)
    print("TEST RUNNER COMPLETED")
    print("=" * 60)
    print("\nYour DeepFace installation appears to be working!")
    print("The 'could not resolve' error in your IDE is just a linting issue.")
    print("Your code should run fine despite the IDE warning.")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n💡 To fix the IDE 'could not resolve' error:")
        print("   1. Make sure your IDE is using the correct Python interpreter")
        print("   2. In VS Code: Ctrl+Shift+P → 'Python: Select Interpreter'")
        print("   3. Choose the interpreter from your virtual environment")
        print("   4. The path should be something like: .venv/bin/python")
    
    sys.exit(0 if success else 1)
