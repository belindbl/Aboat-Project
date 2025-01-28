import numpy as numpy
import cv2, time, threading, struct
import ALSLib.ALSClient, ALSLib.TCPClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg


########################
# In this demo we load a situation that has a vehicle and a camera, then 
# we connect to the camera stream and display it in another window using OpenCV
########################

HOST="127.0.0.1"
PORT=8890

def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')
	#print ('\n---> recieved Raw message: '+str)
	
	cmdList = str.split(" ")
	if cmdList[0].startswith("EndCondition") :
		TestContext.client.request("DestroySituation")

		TestContext.lock.acquire()
		TestContext.testEnded = True
		print("setting TestEnded")
		TestContext.lock.release()

class TestContext:
	lock = threading.Lock()
	testEnded = False
	echo_port = 9000
	client = ALSLib.ALSClient.Client((HOST, echo_port),myMessageHandler)


if __name__ == "__main__":
	TestContext.client.connect()
	TestContext.client.request_load_situation("CameraCalibrationSituation_StereoCam")
	client = ALSLib.TCPClient.TCPClient(HOST, PORT, 5 )
	client.connect(5)

	while(True):
		data = client.read()
		index = 0
		group_sensor_amount, index = ALSFunc.ReadUint32(data, index)
		print("amount of sensors in group :", str(group_sensor_amount))

		recieved_images = []
		for i in range(group_sensor_amount):
			sensor_type, index 		= ALSFunc.ReadString(data, index)
			sensor_path, index 		= ALSFunc.ReadString(data, index)
			print("sensor ", i,": Type ", sensor_type, "  path:", sensor_path)

			image, index, image_width, image_height	= ALSFunc.ReadImage_Group(data, index)
			recieved_images.append(image)

			extra_string, index 	= ALSFunc.ReadString(data, index)
			print("extra sting: ", extra_string)
			
		stack = ALSImg.StackImages(recieved_images)
		ALSImg.JustDisplay(stack)