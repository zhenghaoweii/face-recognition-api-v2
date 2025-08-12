#!/usr/bin/env python3
"""
Example script showing how to use DeepFace.find and extract data from the results
"""

from deepface import DeepFace
import pandas as pd
import os

def demonstrate_find_data():
    """
    Demonstrate how to get and use data from DeepFace.find()
    """
    print("=== DeepFace.find() Data Extraction Example ===\n")
    
    # Check if we have the required files
    source_image = "./haowei-3.jpg"
    database_path = "./user/database"
    
    if not os.path.exists(source_image):
        print(f"❌ Source image not found: {source_image}")
        return
    
    if not os.path.exists(database_path):
        print(f"❌ Database path not found: {database_path}")
        return
    
    try:
        print(f"🔍 Searching for faces in '{source_image}'")
        print(f"📁 Database path: '{database_path}'")
        print()
        
        # DeepFace.find returns a List[pd.DataFrame]
        # Each DataFrame corresponds to one detected face in the source image
        find_results = DeepFace.find(
            img_path=source_image,
            db_path=database_path,
            enforce_detection=False,
            silent=False
        )
        
        print(f"✅ Found {len(find_results)} face(s) in the source image\n")
        
        # Process each detected face
        for face_index, df in enumerate(find_results):
            print(f"--- Face #{face_index + 1} ---")
            print(f"DataFrame shape: {df.shape}")
            print(f"Number of matches: {len(df)}")
            
            if df.empty:
                print("❌ No matches found for this face\n")
                continue
            
            print("\n📊 DataFrame columns available:")
            print(f"   {list(df.columns)}\n")
            
            # Show the top matches
            print("🎯 Top matches:")
            for i, (_, row) in enumerate(df.iterrows()):
                if i >= 3:  # Show only top 3 matches
                    break
                
                identity = row.get('identity', 'Unknown')
                distance = row.get('distance', float('inf'))
                confidence = row.get('confidence', 0)
                threshold = row.get('threshold', 0)
                verified = distance <= threshold
                
                print(f"   Match {i+1}:")
                print(f"     Identity: {identity}")
                print(f"     Distance: {distance:.4f}")
                print(f"     Confidence: {confidence:.2f}%")
                print(f"     Threshold: {threshold:.4f}")
                print(f"     Verified: {'✅' if verified else '❌'}")
                
                # Face coordinates
                target_coords = f"({row.get('target_x', 0)}, {row.get('target_y', 0)}, {row.get('target_w', 0)}, {row.get('target_h', 0)})"
                source_coords = f"({row.get('source_x', 0)}, {row.get('source_y', 0)}, {row.get('source_w', 0)}, {row.get('source_h', 0)})"
                print(f"     Target face area (x,y,w,h): {target_coords}")
                print(f"     Source face area (x,y,w,h): {source_coords}")
                print()
            
            # Show how to get the best match
            if len(df) > 0:
                best_match = df.iloc[0]  # Results are sorted by distance (best first)
                print(f"🏆 Best match: {best_match['identity']} (distance: {best_match['distance']:.4f})")
            
            print()
        
        # Example: How to convert to JSON-serializable format
        print("🔄 Converting to JSON-serializable format:")
        json_data = []
        for i, df in enumerate(find_results):
            face_data = {
                "face_index": i,
                "matches_found": len(df),
                "matches": df.to_dict('records') if not df.empty else []
            }
            json_data.append(face_data)
        
        print(f"   Converted {len(json_data)} face result(s) to JSON format")
        
        # Example: How to filter by confidence threshold
        print("\n🎯 Filtering matches with confidence > 50%:")
        for i, df in enumerate(find_results):
            if not df.empty:
                high_confidence_matches = df[df['confidence'] > 50]
                print(f"   Face {i+1}: {len(high_confidence_matches)} high-confidence matches")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def show_dataframe_structure():
    """
    Show what a typical DeepFace.find DataFrame looks like
    """
    print("\n=== Typical DataFrame Structure ===")
    print("""
A typical find result DataFrame contains these columns:
    
    - identity: Path to the matched image in database
    - target_x, target_y, target_w, target_h: Face coordinates in database image
    - source_x, source_y, source_w, source_h: Face coordinates in source image  
    - threshold: Distance threshold for this model/metric combination
    - distance: Calculated distance between faces (lower = more similar)
    - confidence: Confidence percentage (0-100, higher = more confident)
    
Example row:
    identity: './user/database/Haowei/haowei-1.jpg'
    distance: 0.2345
    confidence: 87.65
    threshold: 0.6836
    target_x: 45, target_y: 67, target_w: 120, target_h: 120
    source_x: 32, source_y: 54, source_w: 115, source_h: 115
    """)

if __name__ == "__main__":
    demonstrate_find_data()
    show_dataframe_structure()
