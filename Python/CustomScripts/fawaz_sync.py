import os
import shutil
import re

def extract_timestamp(filename):
    """Extract timestamp from filename."""
    match = re.search(r'(\d+\.\d+)', filename)
    return float(match.group(1)) if match else None

def find_closest_match(target_time, candidates, max_time_diff):
    """
    Find the closest timestamp match within max_time_diff.
    Returns (filename, timestamp) or (None, None) if no match found.
    """
    closest_file = None
    closest_time = None
    min_diff = float('inf')
    
    for filename, timestamp in candidates:
        time_diff = abs(timestamp - target_time)
        if time_diff < min_diff and time_diff <= max_time_diff:
            min_diff = time_diff
            closest_file = filename
            closest_time = timestamp
    
    return closest_file, closest_time

def sync_sensor_data(pcl_dir, yolo_dir, synced_dir, max_time_diff):
    """
    Synchronize PCL and YOLO data based on timestamps.
    
    Args:
        pcl_dir (str): Directory containing PCL files
        yolo_dir (str): Directory containing YOLO images
        synced_dir (str): Output directory for synchronized pairs
        max_time_diff (float): Maximum allowed time difference in seconds
    """
    # Create sync directories
    sync_pcl_dir = os.path.join(synced_dir, 'pcl')
    sync_yolo_dir = os.path.join(synced_dir, 'yolo')
    os.makedirs(sync_pcl_dir, exist_ok=True)
    os.makedirs(sync_yolo_dir, exist_ok=True)

    # Get all files and their timestamps
    pcl_files = [(f, extract_timestamp(f)) for f in os.listdir(pcl_dir) if f.endswith('.pcd')]
    yolo_files = [(f, extract_timestamp(f)) for f in os.listdir(yolo_dir) if f.endswith('.png')]

    # Remove any files where timestamp couldn't be extracted
    pcl_files = [(f, t) for f, t in pcl_files if t is not None]
    yolo_files = [(f, t) for f, t in yolo_files if t is not None]

    # Sort by timestamp
    pcl_files.sort(key=lambda x: x[1])
    yolo_files.sort(key=lambda x: x[1])

    # Match files and track used timestamps
    matched_pairs = []
    used_yolo = set()
    
    for pcl_file, pcl_time in pcl_files:
        # Filter out already used YOLO files
        available_yolo = [(f, t) for f, t in yolo_files if f not in used_yolo]
        
        # Find closest YOLO file within threshold
        yolo_file, yolo_time = find_closest_match(pcl_time, available_yolo, max_time_diff)
        
        if yolo_file:
            time_diff = abs(pcl_time - yolo_time)
            matched_pairs.append((pcl_file, yolo_file, time_diff))
            used_yolo.add(yolo_file)
            print(f"Matched: PCL {pcl_time:.3f} with YOLO {yolo_time:.3f} (diff: {time_diff*1000:.1f}ms)")

    # Copy matched files to sync directory
    for pcl_file, yolo_file, time_diff in matched_pairs:
        # Copy PCL file
        src_pcl = os.path.join(pcl_dir, pcl_file)
        dst_pcl = os.path.join(sync_pcl_dir, pcl_file)
        shutil.copy2(src_pcl, dst_pcl)

        # Copy YOLO file
        src_yolo = os.path.join(yolo_dir, yolo_file)
        dst_yolo = os.path.join(sync_yolo_dir, yolo_file)
        shutil.copy2(src_yolo, dst_yolo)

    return len(matched_pairs)

if __name__ == "__main__":
    # Paths
    base_dir = r"E:\AILiveSim_1_9_7\SensorData"
    pcl_dir = os.path.join(base_dir, "pcl")
    yolo_dir = os.path.join(base_dir, "yolo")
    synced_dir = os.path.join(base_dir, "synched")

    # Maximum allowed time difference (in seconds)
    max_time_diff = 0.1  # 100ms

    # Run synchronization
    num_matched = sync_sensor_data(pcl_dir, yolo_dir, synced_dir, max_time_diff)
    print(f"\nSuccessfully synchronized {num_matched} pairs of files")
    print(f"Synchronized files saved to: {synced_dir}")