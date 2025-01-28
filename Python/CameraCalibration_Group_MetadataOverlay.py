import cv2
import numpy as np
import math, threading, struct, json
import ALSLib.ALSClient, ALSLib.TCPClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg

def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')
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
	TestContext.client.execute_load_situation("CameraCalibrationSituation_PointTracking")
	sensor_group 		= 8880
	client = ALSLib.TCPClient.TCPClient("localhost", sensor_group, 5 )
	client.connect(5)

	point_colors = [(0, 0, 255),(0, 255, 255),(255, 0, 0)]
	last_brown = None
	while(True):
		data = client.read()
		index = 0
		group_sensor_amount, index = ALSFunc.ReadUint32(data, index)
		recieved_images = []

		for i in range(group_sensor_amount):
			sensor_type, index 		= ALSFunc.ReadString(data, index)
			sensor_path, index 		= ALSFunc.ReadString(data, index)
			print("sensor ", i,": Type ", sensor_type, "  path:", sensor_path)

			image, index, image_width, image_height	= ALSFunc.ReadImage_Group(data, index)
			extra_string, index 	= ALSFunc.ReadString(data, index)

			
			
			print("extra sting: ", extra_string)
			parsed_string 		 	= json.loads(extra_string)
			brown 			 		= ALSImg.BrownParams().Load2(parsed_string['BrownModel'], image_width, image_height)
			points 			 		= ParsePoint2d(parsed_string['TrackedObj'][0]['Points2D'])
			distorted_points 		= brown.DoUndistortPixel(points)
			for point in distorted_points:
				image = cv2.circle(image, (int(point[0]), int(point[1])), radius=5, color=point_colors[i], thickness=-1)
			
			recieved_images.append(image)

			# OPTIONAL:
			# here we are going to redistort the image using open CV and overlay to the original
			# image distorted in AILiveSim 
			if i == 0:
				last_brown 			= brown
			if i == 1: # this is the non distorted image
				distorted_image 	= last_brown.DoDistortImage(image)
				distorted_points 	= last_brown.DoUndistortPixel(points)
				for point in distorted_points:
					distorted_image = cv2.circle(distorted_image, (int(point[0]), int(point[1])), radius=2, color=point_colors[i+1], thickness=-1)
				recieved_images.append(distorted_image)


		print(":::::::::::")
		stack = ALSImg.StackImages(recieved_images)
		ALSImg.JustDisplay(stack)

