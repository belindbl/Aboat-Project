import threading
import struct, cv2
import numpy as np
import matplotlib.pyplot as plt
import ALSLib.TCPClient
import ALSLib.ALSClient 
import ALSLib.ALSHelperImageLibrary as ALSImg
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path
import os
from datetime import datetime

# Work in progress, supposed to collect image data as opposed to pcds
# Like LiveSocketLidar.py does
HOST = '127.0.0.1'
VL16 = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini"
OS1_64 = "EgoVehicleSettings\CustomLidar.SensorProfile.ini"

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

fig = plt.figure()

def save_image(image_data, imageNum):
    try:
        # Image dimensions as specified: 480x720 with 3 channels (RGB)
        height, width, channels = 480, 720, 3

        # Decode the image data using ALSHelperImageLibrary
        img = ALSImg.DecodeImageData(image_data, width, height, channels)

        # Get the sensor data path
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = get_sensordata_path(f'/images/image_{current_time}_{imageNum}.png')

        # Save the image using OpenCV
        cv2.imwrite(filename, img)
        print(f"Image saved to {filename}")

        # Optionally display the image using ALSHelperImageLibrary
        ALSImg.JustDisplay(img)  # Display image using the ALSImg library
        
    except Exception as e:
        print(f"Failed to process image: {e}")

if __name__ == "__main__":
    TestContext.client.connect()
    TestContext.client.request_load_scenario('Default_Scenario')

    #ALSLidar.create_sensor_data_folders()

    sensorprofile_path = VL16
    sensor_path_in_file = "Sensors.Sensors.[1]"
    cam_path_in_file = "Sensors.Sensors.[0]"
    overrides = "{file_path};{sensor_path}.StreamToNetwork;True".format(file_path=sensorprofile_path, sensor_path=cam_path_in_file)
    TestContext.client.request_load_situation_layer_with_overrides('DemoSensorsTest', overrides)

    lazer_proximity_port = 8881  # Keeping this for LIDAR, if needed for later use
    cam_port = 8880  # Camera port
    client = ALSLib.TCPClient.TCPClient(HOST, cam_port, 5)
    client.connect(5)

    imageNum = 0
    while True:
        data = client.read()
        try:
            # The image data is sent as raw bytes, which is directly passed as input
            image_data = data  # Raw data received from the camera

            save_image(image_data, imageNum)
            imageNum += 1  # Increment image number for the next file

        except Exception as e:
            print(f"Failed to process image: {e}")
