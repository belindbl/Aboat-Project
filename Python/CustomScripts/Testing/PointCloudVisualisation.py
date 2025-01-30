# Author: Benjamin Lindblom

import open3d as o3d
import os

# Create and configure an Open3D Visualizer
def setup_visualizer(window_width=1200, window_height=900, window_name="PCD Visualization"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_width, height=window_height, window_name=window_name)

    # Set render options
    opt = vis.get_render_option()
    opt.background_color = [1, 1, 1]  # White background
    opt.point_size = 3.5
    return vis

# Load point cloud
def load_point_cloud(base_path):
    pcd_input = input("Enter the PCD file name: ")
    pcd_file = os.path.join(base_path, f"{pcd_input}.pcd")
    pcd = o3d.io.read_point_cloud(pcd_file)
    return pcd

# Create axis lines to show the coordinate system, specifically the X axis is used as a reference point 
# against the captured images within the same time frame
def create_axis_lines():
    axis_points = [
        [0, 0, 0],    # Origin
        [10000, 0, 0],  # X-axis
        [0, 100, 0],    # Y-axis
        [0, 0, 100],    # Z-axis
    ]
    
    axis_lines = [
        [0, 1],  # X-axis
        [0, 2],  # Y-axis
        [0, 3],  # Z-axis
    ]
    
    axis_colours = [
        [1, 0, 0],  # Red for X-axis
        [0, 1, 0],  # Green for Y-axis
        [0, 0, 1],  # Blue for Z-axis
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(axis_points)
    line_set.lines = o3d.utility.Vector2iVector(axis_lines)
    line_set.colors = o3d.utility.Vector3dVector(axis_colours)
    
    return line_set

# Set up the camera viewpoint and orientation
def configure_camera(vis):
    ctr = vis.get_view_control()
    ctr.set_lookat([200, 0, 300])
    ctr.set_zoom(0.02)
    ctr.set_front([-1, 0, 1])
    ctr.set_up([1, 0, 1])

def main():
    base_path = r"SensorData\pcl"

    # 1) Prompt for input first
    pcd_input = input("Enter the PCD file name: ")
    pcd_file = os.path.join(base_path, f"{pcd_input}.pcd")
    pcd = o3d.io.read_point_cloud(pcd_file)

    # 2) Now set up and show the visualizer
    vis = setup_visualizer()
    vis.add_geometry(pcd)

    line_set = create_axis_lines()
    vis.add_geometry(line_set)

    configure_camera(vis)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
