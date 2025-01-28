import threading
import struct
import math
import numpy as np
import os
import cv2
import ALSLib.TCPClient
import ALSLib.ALSClient
import ALSLib.ALSHelperLidarLibrary as ALSLidar
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path
import time
import json
import ALSLib.ALSHelperImageLibrary as ALSImg

# Camera settings: AboatCamera.ini
# LIDAR settings: VL16.xml

# Global variables
HOST = '127.0.0.1'
ABOAT = "EgoVehicleSettings\AboatSensorTest.SensorProfile.ini"
TESTING_PROFILE = "E:\AILiveSim_1_9_7\ConfigFiles\Scenario\Testing.ini"

MAX_LIDAR_PCDS = 100
IMAGE_NUM = 100
lidar_pcd_count = 0
stop_threads_event = threading.Event()  # Event to signal threads to stop

# Shared condition variable for synchronization
condition = threading.Condition()

# Synchronization flags
camera_ready = False  # To indicate that Camera is ready to capture next image

def myMessageHandler(rawMessage):
    str = rawMessage.decode('utf-8')
    # print ('\n---> recieved Raw message: '+str)
    
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
    pclFileContent = '# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb\n'\
    'SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT %f %f %f %f %f %f %f\n'\
    'POINTS %d\nDATA ascii\n' % (int(numPoints), posX, posY, posZ, quatW, quatX, quatY, quatZ, int(numPoints))
    
    for p in point_array:
        intensity = 1000
        if not math.isinf(p[3]) and p[3] != 0:
            intensity = int(p[3])
        pclFileContent += '%.5f %.5f %.5f %d\n' % (p[0], -p[1], p[2], intensity) 
    return pclFileContent

# LiDAR thread to handle point cloud data collection and saving
def lidar_thread():
    global lidar_pcd_count, camera_ready
    client = ALSLib.TCPClient.TCPClient(HOST, 8881, 5)  # Assuming LiDAR is on port 8881
    client.connect(5)

    while lidar_pcd_count < MAX_LIDAR_PCDS and not stop_threads_event.is_set():  # Stop after 300 PCD files
        # Read LiDAR data
        data = client.read()
        sizeofFloat = 4
        index = 11
        posX, posY, posZ, quatW, quatX, quatY, quatZ, numPoints, timeStart, timeEnd, numberOfBeams = struct.unpack('<fffffffffff', data[0:index * sizeofFloat])

        # LiDAR point cloud data
        pointCloudData = data[index * sizeofFloat:]
        point_array = np.frombuffer(pointCloudData, dtype=np.dtype("float32"))
        point_array = np.reshape(point_array, (-1, 4))

        # Save the point cloud to file
        pclFileContent = SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array)

        filename = get_sensordata_path(f"/pcl/pcl{lidar_pcd_count}.pcd")
        try:
            with open(filename, mode='w') as fileObject:
                fileObject.write(pclFileContent)
            print(f"Point cloud {lidar_pcd_count} saved to {filename}")
            lidar_pcd_count += 1
        except Exception as e:
            print(f"Failed to save point cloud to file: {e}")

        # Signal Camera thread to proceed after LiDAR frame is processed
        with condition:
            camera_ready = True  # LiDAR frame ready, Camera can capture next image
            condition.notify()  # Notify Camera thread to capture
            condition.wait()    # Wait for Camera thread to finish before processing next LiDAR frame

        time.sleep(0.05)

    stop_threads_event.set()  # Signal the camera thread to stop once LiDAR thread completes


# Camera thread to handle image display and saving
def camera_thread():

    # Camera image save path
    CAMERA_SAVE_PATH = r"E:\AILiveSim_1_9_7\SensorData\imgs"
    if not os.path.exists(CAMERA_SAVE_PATH):
        os.makedirs(CAMERA_SAVE_PATH)

    global camera_ready
    # Get the sensor list
    time.sleep(0.1)
    sensorlist = TestContext.client.get_sensor_list()
    parsed_json = json.loads(sensorlist)

    # Find the host and port of the camera
    for x in parsed_json['sensors']:
        if x['path'] == 'Sensors.[0]':
            sensor = x
            break
    camera_port = sensor['sensor_port']
    host = sensor['sensor_ip']

    print("Connecting sensor socket to " + host + " " + str(camera_port))
    client = ALSLib.TCPClient.TCPClient(host, camera_port, 5)
    client.connect(5)

    image_num = 0
    while image_num < IMAGE_NUM:
        with condition:
            while not camera_ready:  # Wait for LiDAR thread to signal camera readiness
                condition.wait()

            # Read and save image
            data = client.read()
            index = 0
            image_num += 1
            img, index, width, height = ALSLib.ALSHelperFunctionLibrary.ReadImage_Stream(data, index)

            ALSImg.JustDisplay(img)
            image_filename = os.path.join(CAMERA_SAVE_PATH, f"image_{image_num}.jpg")
            cv2.imwrite(image_filename, img)  # Save the image

            print(f"Image {image_num} saved to {image_filename}")

            # Signal LiDAR thread to proceed to the next LiDAR frame
            camera_ready = False  # Camera has finished capturing
            condition.notify()  # Notify LiDAR thread to process next LiDAR frame


# Main execution flow
if __name__ == "__main__":
    TestContext.client.connect()
    TestContext.client.request_load_scenario("Testing")

    ALSLidar.create_sensor_data_folders()

    sensorprofile_path = ABOAT
    sensor_path_in_file = "Sensors.Sensors.[1]"
    cam_path_in_file = "Sensors.Sensors.[0]"

    # Start LiDAR thread
    lidar_thread_instance = threading.Thread(target=lidar_thread)
    lidar_thread_instance.start()

    # Start camera thread
    camera_thread_instance = threading.Thread(target=camera_thread)
    camera_thread_instance.start()

    # Wait for the LiDAR thread to finish
    lidar_thread_instance.join()

    # Once the LiDAR thread finishes, the camera thread will also be stopped
    camera_thread_instance.join()
    print("All threads finished.")
