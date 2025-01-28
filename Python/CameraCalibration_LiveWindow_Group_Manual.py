import numpy as numpy
import cv2, time, threading, struct
import ALSLib.ALSClient, ALSLib.TCPClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg


########################
# In this demo we load a situation that has a vehicle and a camera, then 
# we connect to the camera stream and display it in another window using OpenCV
########################


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
	client = ALSLib.ALSClient.Client(("127.0.0.1", echo_port),myMessageHandler)


def findCheckerBoard(img):
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
	cv2.imwrite( "test.bmp", gray)
	ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
	if ret == True:
		corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
		cv2.drawChessboardCorners(img, (11,8), corners2, ret)

if __name__ == "__main__":
	TestContext.client.connect()
	override = 'EgoVehicleSettings/CameraCalibration_GT.SensorProfile.ini;GeneralSensorGroups.GroupData.[0].StreamToNetwork;true'
	TestContext.client.request_load_situation_with_overrides('CameraCalibrationSituation_Manual', override)

	HOST="127.0.0.1"
	PORT=8890
	print( "Connecting to  " + HOST + " " + str(PORT) )
	client = ALSLib.TCPClient.TCPClient(HOST, PORT, 5 )
	client.connect(5)

	while(True):
		data = client.read()
		index = 0
		group_sensor_amount, index = ALSFunc.ReadUint32(data, index)
		print("group sensor amount:", str(group_sensor_amount))

		recieved_images = []
		for i in range(group_sensor_amount):
			sensor_type, index 		= ALSFunc.ReadString(data, index)
			sensor_path, index 		= ALSFunc.ReadString(data, index)
			print("sensor ", i,": Type ", sensor_type, "  path:", sensor_path)

			image, index, image_width, image_height	= ALSFunc.ReadImage_Group(data, index)

			if i == 0: 
				findCheckerBoard(image)
			recieved_images.append(image)

			extra_string, index 	= ALSFunc.ReadString(data, index)
			print("extra sting: ", extra_string)
			
		
		stack = ALSImg.StackImages(recieved_images)
		ALSImg.JustDisplay(stack) 