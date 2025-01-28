import open3d as o3d
import numpy as np
import sys
import os

#DEPRECATED

def load_and_transform_pcd(pcd_file):
    # Load the PCD file
    try:
        pcd = o3d.io.read_point_cloud(pcd_file)
    except Exception as e:
        print(f"Error reading PCD file: {e}")
        return None

    # Check if the point cloud was loaded successfully
    if pcd.is_empty():
        print(f"Failed to load point cloud data from {pcd_file}")
        return None

    # Print basic information about the point cloud
    print(f"Point cloud loaded with {len(pcd.points)} points")

    # Check for invalid points (NaN or Inf)
    points = np.asarray(pcd.points)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        print("Warning: Point cloud contains NaN or Inf values. Cleaning up...")
        # Remove NaN/Inf points
        valid_points = points[~np.isnan(points).any(axis=1)]
        valid_points = valid_points[~np.isinf(valid_points).any(axis=1)]
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        print(f"Cleaned point cloud has {len(pcd.points)} valid points.")

    # Define a Z-axis flip matrix (flips the Z-axis)
    transformation_matrix = np.array([[1, 0, 0, 0],  # X stays the same
                                      [0, 1, 0, 0],  # Y stays the same
                                      [0, 0, -1, 0], # Flip Z
                                      [0, 0, 0, 1]]) # Homogeneous coordinate

    # Apply the transformation
    pcd.transform(transformation_matrix)

    return pcd


def update_pcd_header(pcd_file, transformed_pcd):
    # Manually update the PCD header (if necessary)
    header = ""
    header += "# .PCD v7 - Point Cloud Data file format\n"
    header += "VERSION .7\n"
    header += "FIELDS x y z\n"
    header += "SIZE 4 4 4\n"
    header += "TYPE F F F\n"
    header += "COUNT 1 1 1\n"
    header += f"POINTS {len(transformed_pcd.points)}\n"
    
    # Example of updating the VIEWPOINT field:
    # If you have a known viewpoint you want to use (for example, after transformation)
    viewpoint = "0 0 0 1 0 0 0"  # Update this if your viewpoint changes
    header += f"VIEWPOINT {viewpoint}\n"

    # Return the updated header
    return header


def save_transformed_pcd(pcd, input_file):
    # Ensure the input file path is correct
    input_dir = os.path.dirname(input_file)
    print(f"Input file directory: {input_dir}")

    # Generate the output file name by appending "_transform" before the extension
    base_name, ext = os.path.splitext(os.path.basename(input_file))
    output_file = os.path.join(input_dir, f"{base_name}_transform{ext}")

    print(f"Output file path: {output_file}")

    # Ensure the directory exists for the output file
    if not os.path.exists(input_dir):
        print(f"Creating directory: {input_dir}")
        os.makedirs(input_dir)
    
    # Save the transformed point cloud to a new PCD file
    try:
        header = update_pcd_header(input_file, pcd)
        
        # Write the updated header to the PCD file
        with open(output_file, "w") as file:
            file.write(header)

            # Save the point cloud data in binary or ASCII format
            print(f"Saving point cloud data to {output_file}...")
            o3d.io.write_point_cloud(output_file, pcd)
        
        print(f"Transformed point cloud saved to {output_file}")

    except Exception as e:
        print(f"Error saving point cloud: {e}")


if __name__ == "__main__":
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python visualize_pcd.py <path_to_pcd_file>")
        sys.exit(1)

    input_file = sys.argv[1]  # Path to the input PCD file
    
    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Load, transform, and save the point cloud
    pcd = load_and_transform_pcd(input_file)
    if pcd:
        save_transformed_pcd(pcd, input_file)
