import open3d as o3d
import numpy as np
import time
import os

def visualize_sequence():
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("LiDAR Point Cloud Sequence", width=1400, height=1000)
    print("create window")
    # Get render options
    opt = vis.get_render_option()
    opt.background_color = (255, 255, 255)
    opt.point_size = 10
    
    # Base path
    base_path = r"E:\AILiveSim_1_9_7\SensorData\pcl"

    # Add first point cloud to initialize
    first_pcd = o3d.io.read_point_cloud(os.path.join(base_path, "pcl0.pcd"))
    vis.add_geometry(first_pcd)
    
    print("\nControls:")
    print("- Left mouse: Rotate")
    print("- Right mouse: Zoom")
    print("- Middle mouse: Pan")
    print("- 'esc': Exit")
    
    try:
        for i in range(1000):  # 0 to 410 inclusive
            file_path = os.path.join(base_path, f"pcl{i}.pcd")
            
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping...")
                continue
                
            print(f"\rFrame {i}/1000", end="", flush=True)
            
            # Read and process point cloud
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_colors():
                pcd.paint_uniform_color([0.7, 0.7, 0.7])
            
            # Update points and colors
            first_pcd.points = pcd.points
            first_pcd.colors = pcd.colors
            
            # Update visualization
            vis.update_geometry(first_pcd)
            vis.poll_events()
            vis.update_renderer()
            
            # Wait 0.1 seconds
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    finally:
        vis.destroy_window()

if __name__ == "__main__":
    try:
        visualize_sequence()
    except Exception as e:
        print(f"\nError: {str(e)}")
