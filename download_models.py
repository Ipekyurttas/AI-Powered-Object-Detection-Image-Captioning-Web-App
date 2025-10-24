import os
import requests
import time

def download_file_with_progress(url, filename):
    """File download karta hai with proper progress"""
    try:
        print(f"📥 Downloading {filename} from {url}")
        
        # Streaming download
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                downloaded += len(data)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"   Progress: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB)", end='\r')
        
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"\n✅ Downloaded: {filename} ({file_size:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def main():
    """YOLO model files download karta hai"""
    print("🚀 Downloading YOLO Model Files...")
    os.makedirs('models', exist_ok=True)
    
    # YOLO model files
    model_files = [
        {
            'url': 'https://pjreddie.com/media/files/yolov3.weights',
            'filename': 'models/yolov3.weights',
            'expected_size': 237  # MB
        },
        {
            'url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg', 
            'filename': 'models/yolov3.cfg',
            'expected_size': 0.008  # MB
        },
        {
            'url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
            'filename': 'models/coco.names', 
            'expected_size': 0.001  # MB
        }
    ]
    
    all_success = True
    for file_info in model_files:
        filename = file_info['filename']
        
        # Check if file already exists with proper size
        if os.path.exists(filename):
            actual_size = os.path.getsize(filename) / (1024 * 1024)
            expected_size = file_info['expected_size']
            
            # Allow 10% size variation
            if abs(actual_size - expected_size) <= (expected_size * 0.1):
                print(f"✅ Already exists: {os.path.basename(filename)} ({actual_size:.1f} MB)")
                continue
            else:
                print(f"⚠️  File exists but wrong size: {actual_size:.1f} MB (expected: {expected_size:.1f} MB)")
                os.remove(filename)
        
        # Download file
        if not download_file_with_progress(file_info['url'], filename):
            all_success = False
    
    if all_success:
        print("\n🎯 All model files downloaded successfully!")
        print("\n📁 Files in 'models' folder:")
        for file_info in model_files:
            if os.path.exists(file_info['filename']):
                size = os.path.getsize(file_info['filename']) / (1024 * 1024)
                print(f"   - {os.path.basename(file_info['filename'])} ({size:.1f} MB)")
    else:
        print("\n❌ Some downloads failed!")

if __name__ == "__main__":
    main()