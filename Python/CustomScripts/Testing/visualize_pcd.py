import open3d as o3d
import sys

def load_and_visualize_pcd(pcd_file):
    # Load the PCD file
    try:
        pcd = o3d.io.read_point_cloud(pcd_file)
    except Exception as e:
        print(f"Error reading PCD file: {e}")
        return

    # Check if the point cloud was loaded successfully
    if pcd.is_empty():
        print(f"Failed to load point cloud data from {pcd_file}")
        return

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], width = 800, height = 640, window_name="PCD Visualization")

if __name__ == "__main__":
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python visualize_pcd.py <path_to_pcd_file>")
        sys.exit(1)

    pcd_file = sys.argv[1]
    
    # Call the function to load and visualize the point cloud
    load_and_visualize_pcd(pcd_file)
