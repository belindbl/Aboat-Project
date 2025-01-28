import open3d as o3d
import sys
import os
import numpy as np

def load_and_visualize_pcd(arg):
    # Construct the full path
    path = os.path.join(r"E:\AILiveSim_1_9_7\SensorData\pcl", arg)

    # Load the PCD file
    try:
        pcd = o3d.io.read_point_cloud(path)
    except Exception as e:
        print(f"Error reading PCD file: {e}")
        return

    # Check if the point cloud was loaded successfully
    if pcd.is_empty():
        print(f"Failed to load point cloud data from {path}")
        return

    # Find the densest region (most points in a small bounding box)
    points = np.asarray(pcd.points)
    if points.size == 0:
        print("Point cloud has no points.")
        return
    
    # Calculate point density by dividing the space into a grid
    voxel_size = 0.2  # Adjust this for finer or coarser density evaluation
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    density = np.asarray(pcd_down.compute_point_cloud_distance(pcd))

    # Find the most dense voxel
    densest_idx = np.argmin(density)
    densest_point = pcd_down.points[densest_idx]
    
    # Print the densest point for debugging
    print(f"Densest point: {densest_point}")

    # Create a Visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=640, window_name="PCD Visualization")
    
    # Set the background colour to black
    opt = vis.get_render_option()
    opt.background_color = [255, 255, 255]  # RGB values for black

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    # Adjust the camera to focus on the densest point
    ctr = vis.get_view_control()
    ctr.set_lookat(densest_point)
    ctr.set_zoom(0.5)  # Adjust zoom to get closer
    ctr.set_front([0, 0, -1])  # Set the front direction
    ctr.set_up([0, -1, 0])  # Set the upward direction

    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Check if a file name is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python visualize_pcd.py <file_name>")
        sys.exit(1)

    file_name = sys.argv[1]
    
    # Call the function to load and visualize the point cloud
    load_and_visualize_pcd(file_name)
