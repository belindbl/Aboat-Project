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
    
    # Get view control
    view_control = vis.get_view_control()
    
    print("\nControls:")
    print("- Left mouse: Rotate")
    print("- Right mouse: Zoom")
    print("- Middle mouse: Pan")
    print("- 'esc': Exit")
    
    try:
        for i in range(1000):  # 0 to 1000 inclusive
            file_path = os.path.join(base_path, f"pcl{i}.pcd")
            
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping...")
                break
                
            print(f"\rFrame {i}/1000", end="", flush=True)
            
            # Read point cloud
            pcd = o3d.io.read_point_cloud(file_path)
            
            # Filter points by distance
            distances = np.linalg.norm(np.asarray(pcd.points), axis=1)  # Calculate distances from origin
            filtered_indices = distances < 200000  # Keep points with distance < 200000
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[filtered_indices])
            if pcd.has_colors():
                pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[filtered_indices])
            
            # Handle uncoloured point clouds
            if not pcd.has_colors():
                pcd.paint_uniform_color([0.7, 0.7, 0.7])
            
            # Update points and colours
            first_pcd.points = pcd.points
            first_pcd.colors = pcd.colors
            
            # Adjust viewpoint to fit all points
            view_control.set_lookat(first_pcd.get_center())
            view_control.set_zoom(0.8)  # Adjust zoom level to fit better
            view_control.set_front([0, 0, -1])  # Ensure a front view
            view_control.set_up([0, -1, 0])  # Set up vector for orientation
            
            # Update visualisation
            vis.update_geometry(first_pcd)
            vis.poll_events()
            vis.update_renderer()
            
            # Wait 0.1 seconds
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nVisualisation stopped by user")
    finally:
        vis.destroy_window()

if __name__ == "__main__":
    try:
        visualize_sequence()
    except Exception as e:
        print(f"\nError: {str(e)}")
