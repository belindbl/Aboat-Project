import open3d as o3d
import sys
import os

# Example: arg = pcl0.pcd
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

    # Create a Visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=640, window_name="PCD Visualization")
    
    # Set the background colour
    opt = vis.get_render_option()
    opt.background_color = [255, 255, 255]  # White

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

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
