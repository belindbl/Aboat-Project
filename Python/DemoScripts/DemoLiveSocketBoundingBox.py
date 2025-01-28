import cv2
import numpy as np
import threading, json
import ALSLib.ALSClient, ALSLib.TCPClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg

###############################################################################
# This demo shows how to collect the Bounding Box data
###############################################################################

def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')
	cmdList = str.split(" ")
	if cmdList[0].startswith("EndCondition") :
		TestContext.client.request_destroy_situation()

		TestContext.lock.acquire()
		TestContext.testEnded = True
		print("setting TestEnded")
		TestContext.lock.release()

class TestContext:
	lock = threading.Lock()
	testEnded = False
	echo_port = 9000
	localhost = 'localhost'
	client = ALSLib.ALSClient.Client((localhost, echo_port),myMessageHandler)

def ParsePoint2d(jsonArray):
	parased_points = []
	for p in jsonArray:
		parased_points.append((round(float(p["X"])), round(float(p["Y"]))))
	points_array = np.array(parased_points)
	return points_array


if __name__ == "__main__":
	TestContext.client.connect()
	TestContext.client.request_load_situation('TestBoundingBox')

	image_port 		= 8880
	client = ALSLib.TCPClient.TCPClient("localhost", image_port, 5 )
	client.connect(5)

	while(True):
		data 	= client.read()
		index 	= 0
		img, index, width, height = ALSFunc.ReadImage_Stream(data, index)
		img = cv2.UMat(img)
		images_to_display = []
		if index < len(data) :
			extra_string, index = ALSFunc.ReadString(data,index)
			parsed_string 		= json.loads(extra_string)
			time 				= float(parsed_string['T'])
			if 'TrackedObj' in parsed_string.keys():
				tracked_objects =  parsed_string['TrackedObj']
				for a_obj in tracked_objects:
					print(a_obj)
					if None is not a_obj.get('BB2D'):
						BoundingBox			= a_obj['BB2D']
						pt1 = (int(BoundingBox[0]['X']), int(BoundingBox[0]['Y']))
						pt2 = (int(BoundingBox[1]['X']), int(BoundingBox[1]['Y']))
						img = cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
			images_to_display.append(img)

		print(":::::::::::")
		added_image = ALSImg.StackImages(images_to_display)
		ALSImg.JustDisplay(added_image)
