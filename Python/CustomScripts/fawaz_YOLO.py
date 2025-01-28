import threading, time, json
import ALSLib.TCPClient, ALSLib.ALSClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg
import torch
import cv2
import time
import os

#WORKING 2025-01-15 13:50
# Set your desired save directory for images
#fawaz - uncommented the next 3 lines until os.makedirs
save_directory = "E:\AILiveSim_1_9_7\SensorData\yolo"
#save_directory = "E:/Team1Docs/captured_images_raw"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

HOST = '127.0.0.1'
ALSweights = "E:/Team1Docs/yolov5/weights_ailivesim/best.pt"
MixedWeights = "E:/Team1Docs/yolov5/weights_mixed/best.pt"

# Initialize YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MixedWeights)
model.conf = 0.25  # Confidence threshold
model.iou = 0.45   # NMS IOU threshold

# Message handler
def myMessageHandler(rawMessage):
    str_msg = rawMessage.decode('utf-8')
    cmdList = str_msg.split(" ")
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
    """
    Apply YOLOv5 detection on a single frame and return an image with bounding boxes.
    """
    # Convert input image to RGB if needed
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    results = model(image)
    processed_image = results.render()[0]  # Image with bounding boxes drawn (RGB)
    
    # Convert to RGBA for consistency with StackImages
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2RGBA)
    return processed_image

def process_pcloud(pcd):
    """
    (Not Implemented)
    """

if __name__ == "__main__":
    TARGET_FPS = 10  # Set your desired frame rate
    FRAME_TIME = 1/TARGET_FPS  # Time per frame in seconds
    
    # Initialize simulation
    TestContext.client.connect()
    start_time = time.time() #fawaz - time start
    TestContext.client.request_load_scenario('Default_Scenario')
    overrides = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini;Sensors.Sensors.[0].StreamToNetwork;True"
    TestContext.client.request_load_situation_layer_with_overrides('DemoSensorsTest', overrides)
    
    # Get the sensor list
    time.sleep(1)
    sensorlist = TestContext.client.get_sensor_list()
    parsed_json = json.loads(sensorlist)
    
    # Find the host and port of the camera
    for x in parsed_json['sensors']:
        if x['path'] == 'Sensors.[0]':
            sensor_cam1 = x
            break
    camera_port = sensor_cam1['sensor_port']
    host = sensor_cam1['sensor_ip']

    print("Connecting sensor socket to " + host + " " + str(camera_port))
    client = ALSLib.TCPClient.TCPClient(host, camera_port, 5 ) # Last arg is timeout in seconds
    client.connect(5)

    imageNum = 0
    while(imageNum < 1000):
        frame_start_time = time.time()  # Start timing the frame
        current_time = time.time() - start_time  #fawaz- Get elapsed time since start
        
        # Read data from stream
        data = client.read()

        index = 0
        img, index, width, height = ALSFunc.ReadImage_Stream(data, index)
        
        # Save the original image without bounding boxes
        # Convert to BGR before saving (assuming img is RGBA)
        if img.shape[2] == 4: # RGBA to BGR
            original_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            # If it's already BGR or RGB, convert accordingly
            # Assume it's RGBA if 4 channels, otherwise it's probably RGB
            original_bgr = img

        #save_path = os.path.join(save_directory, f"processed_image_{imageNum:04d}.png")
        #cv2.imwrite(save_path, original_bgr)

        # Process the image to show bounding boxes (if you still want to display them)
        processed_img = process_frame(img)
        #fawaz - save the image here after processing it to show bounding boxes

        # Save the processed image (with YOLO boxes)
        if processed_img.shape[2] == 4:  # If RGBA
            save_img = cv2.cvtColor(processed_img, cv2.COLOR_RGBA2BGR)
        else:
            save_img = processed_img

        #fawaz - commented the old save_path and made a new one in the next 2 lines    
        timestamp = f"{current_time:.3f}"
        save_path = os.path.join(save_directory, f"{timestamp}.png")
        #save_path = os.path.join(save_directory, f"yolo_output_{imageNum:04d}.png")
        cv2.imwrite(save_path, save_img)

        #end of saving here
        ALSImg.JustDisplay(processed_img)
        imageNum += 1
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Frame rate control
        frame_time = time.time() - frame_start_time
        if frame_time < FRAME_TIME:
            time.sleep(FRAME_TIME - frame_time)
        
        # Optional: Print actual FPS
        actual_fps = 1.0 / (time.time() - frame_start_time)
        print(f"FPS: {actual_fps:.2f}", end='\r')

    cv2.destroyAllWindows()
