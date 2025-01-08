import threading
import time
import json
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import ALSLib.TCPClient
import ALSLib.ALSClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg
import ALSLib.ALSHelperLidarLibrary as ALSLidar
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path

HOST = '127.0.0.1'
ALSweights = "E:/Team1Docs/yolov5/weights_ailivesim/best.pt"
TARGET_FPS = 10  # Desired frame rate for both threads
FRAME_TIME = 1 / TARGET_FPS

# Shared lock for synchronising access to shared resources
lock = threading.Lock()

# YOLOv5 model initialisation
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=ALSweights)
model.conf = 0.25
model.iou = 0.45

# Setup Matplotlib figure for LiDAR visualisation
fig = plt.figure()

# Message handler
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

def process_frame(image):
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    results = model(image)
    processed_image = results.render()[0]
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2RGBA)
    return processed_image

def plot_ply(points):
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.pause(0.07)
    plt.clf()

def camera_thread(camera_host, camera_port):
    client = ALSLib.TCPClient.TCPClient(camera_host, camera_port, 5)
    client.connect(5)

    while True:
        frame_start_time = time.time()

        # Read data from the camera
        data = client.read()
        index = 0
        img, index, width, height = ALSFunc.ReadImage_Stream(data, index)

        processed_img = process_frame(img)

        # Display the processed image
        ALSImg.JustDisplay(processed_img)

        # Frame rate control
        frame_time = time.time() - frame_start_time
        if frame_time < FRAME_TIME:
            time.sleep(FRAME_TIME - frame_time)

def SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array, timeEnd):

    posY = -posY
    pclFileContent = '# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb time\n'\
    'SIZE 4 4 4 4 4\nTYPE F F F U F\nCOUNT 1 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT %f %f %f %f %f %f %f\n'\
    'POINTS %d\nDATA ascii\n' % (int(numPoints), posX, posY, posZ, quatW, quatX, quatY, quatZ, int(numPoints))
    
    for p in point_array:
        intensity = 1000
        if not math.isinf(p[3]):
            intensity = int(p[3])
        pclFileContent += '%.5f %.5f %.5f %d %.5f\n' % (p[0], p[1], p[2], intensity, timeEnd) 
    return pclFileContent

def lidar_thread(lidar_host, lidar_port):
    client = ALSLib.TCPClient.TCPClient(lidar_host, lidar_port, 5)
    client.connect(5)

    image_num = 0
    while True:
        data = client.read()
        sizeof_float = 4
        index = 11
        posX, posY, posZ, quatW, quatX, quatY, quatZ, numPoints, timeStart, timeEnd, \
            numberOfBeams = struct.unpack('<fffffffffff', data[0:index * sizeof_float])

        # Parse the point cloud
        point_cloud_data = data[index * sizeof_float:]
        point_array = np.frombuffer(point_cloud_data, dtype=np.dtype("float32"))
        point_array = np.reshape(point_array, (-1, 4))

        # Visualise the point cloud
        plot_ply(point_array)

        # Save the point cloud to file
        pcl_file_content = SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array, timeEnd)
        filename = get_sensordata_path(f'/pcl/pcl{image_num}.pcd')

        with lock:  # Ensure file writing is thread-safe
            try:
                with open(filename, mode='w') as file_object:
                    file_object.write(pcl_file_content)
                print(f"Point cloud saved to {filename}")
                image_num += 1
            except Exception as e:
                print(f"Failed to save point cloud to file: {e}")

if __name__ == "__main__":
    # Initialize simulation and retrieve sensor info
    TestContext.client.connect()
    TestContext.client.request_load_scenario('Default_Scenario')
    time.sleep(1)
    ALSLidar.create_sensor_data_folders()

    sensorprofile_path = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini"
    sensor_path_in_file = "Sensors.Sensors.[1]"

    sensor_list = TestContext.client.get_sensor_list()
    parsed_json = json.loads(sensor_list)

    # Get camera and LiDAR information
    camera = next((x for x in parsed_json['sensors'] if x['path'] == 'Sensors.[0]'), None)
    lidar = next((x for x in parsed_json['sensors'] if x['path'] == 'Sensors.[1]'), None)

    if not camera or not lidar:
        raise ValueError("Failed to retrieve sensor configuration.")

    camera_host, camera_port = camera['sensor_ip'], camera['sensor_port']
    lidar_host, lidar_port = lidar['sensor_ip'], lidar['sensor_port']

    # Start threads for camera and LiDAR processing
    camera_thread_instance = threading.Thread(target=camera_thread, args=(camera_host, camera_port))
    lidar_thread_instance = threading.Thread(target=lidar_thread, args=(lidar_host, lidar_port))

    camera_thread_instance.start()
    lidar_thread_instance.start()

    # Wait for threads to complete (optional)
    camera_thread_instance.join()
    lidar_thread_instance.join()
