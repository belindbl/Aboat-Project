import threading
import struct
import math
import numpy as np
import open3d as o3d
import ALSLib.TCPClient
import ALSLib.ALSClient 
import ALSLib.ALSHelperLidarLibrary as ALSLidar
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path


# WORKING
HOST = '127.0.0.1'

def myMessageHandler(rawMessage):
    str = rawMessage.decode('utf-8')
    cmdList = str.split(" ")
    if cmdList[0].startswith("EndCondition"):
        TestContext.client.request_destroy_situation()
        TestContext.lock.acquire()
        TestContext.testEnded = True
        print("setting TestEnded")
        TestContext.lock.release()

class TestContext:
    lock = threading.Lock()
    testEnded = False
    simulation_control_port = 9000
    client = ALSLib.ALSClient.Client((HOST, simulation_control_port), myMessageHandler)

def SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array):
    posY = -posY
    pclFileContent = '# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb\n' \
                     'SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\n' \
                     'VIEWPOINT %f %f %f %f %f %f %f\nPOINTS %d\nDATA ascii\n' % \
                     (int(numPoints), posX, posY, posZ, quatW, quatX, quatY, quatZ, int(numPoints))
    for p in point_array:
        intensity = 1000 if math.isinf(p[3]) else int(p[3])
        pclFileContent += '%.5f %.5f %.5f %d\n' % (p[0], p[1], p[2], intensity)
    return pclFileContent



def dynamic_plot_ply(vis, pcd, points):
    """
    Dynamically updates points in the Open3D visualisation.
    :param vis: Open3D Visualizer object
    :param pcd: Open3D PointCloud object
    :param points: numpy array of shape (N, 4) representing the point cloud
    """
    if points.size == 0:
        print("No points to display!")
        return

    # Assign only x, y, z to the point cloud (ignore intensity or RGB for now)
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Use only first three columns

    # Optionally assign colors if intensity is available (normalize intensity for RGB)
    if points.shape[1] >= 4:
        intensity = points[:, 3]
        normalized_intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
        colors = np.tile(normalized_intensity[:, None], (1, 3))  # Convert intensity to RGB
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Update the visualisation
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

if __name__ == "__main__":
    TestContext.client.connect()
    TestContext.client.request_load_scenario('Default_Scenario')

    ALSLidar.create_sensor_data_folders()

    sensorprofile_path = "EgoVehicleSettings\\DemoAllSensors.SensorProfile.ini"
    sensor_path_in_file = "Sensors.Sensors.[1]"
    overrides = "{file_path};{sensor_path}.StreamToNetwork;True".format(file_path=sensorprofile_path, sensor_path=sensor_path_in_file)
    TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)

    lazer_proximity_port = 8881
    client = ALSLib.TCPClient.TCPClient(HOST, lazer_proximity_port, 5)
    client.connect(5)

    # Initialise Open3D visualiser
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Dynamic 3D Point Cloud Visualization", width=800, height=640)
    vis.get_render_option().background_color = [255, 255, 255]  # Black background
    vis.get_render_option().point_size = 2.0

    # Initialise point cloud object
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    imageNum = 0
    try:
        while True:
            data = client.read()
            sizeofFloat = 4
            index = 11
            posX, posY, posZ, quatW, quatX, quatY, quatZ, numPoints, timeStart, timeEnd, \
                numberOfBeams = struct.unpack('<fffffffffff', data[0:index * sizeofFloat])
            print(f'PCL Position: ({posX:.2f}, {posY:.2f}, {posZ:.2f})')

            # Process the point cloud data
            pointCloudData = data[index * sizeofFloat:]
            point_array = np.frombuffer(pointCloudData, dtype=np.dtype("float32"))
            point_array = np.reshape(point_array, (-1, 4))

            print("Point array shape:", point_array.shape)
            print("Sample points:", point_array[:5])

            # Update visualisation dynamically
            dynamic_plot_ply(vis, pcd, point_array)

            # Save the point cloud to a file
            pclFileContent = SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array)
            filename = get_sensordata_path(f'/pcl/pcl{imageNum}.pcd')

            try:
                with open(filename, mode='w') as fileObject:
                    fileObject.write(pclFileContent)
                print(f"Point cloud saved to {filename}")
                imageNum += 1
            except Exception as e:
                print(f"Failed to save point cloud to file: {e}")
    except KeyboardInterrupt:
        print("Terminating...")
    finally:
        vis.destroy_window()
