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
	TestContext.client.request_load_situation("CameraCalibrationSituation_CollectGT")

	#get 2 cameras, one distorted and one not
	non_distorted_port 	= 8881
	distorted_port 		= 8880
	client = ALSLib.TCPClient.TCPClient("localhost", non_distorted_port, 5 )
	client.connect(5)
	client_disto = ALSLib.TCPClient.TCPClient("localhost", distorted_port, 5 )
	client_disto.connect(5)

	while(True):
		data 	= client.read()
		index 	= 0
		img, index, width, height = ALSFunc.ReadImage_Stream(data, index)
		images_to_display = []
		if index < len(data) :
			extra_string, index = ALSFunc.ReadString(data,index)
			parsed_string 		= json.loads(extra_string)
			time_nodisto 		= float(parsed_string['T'])
			for point in parsed_string['TrackedObj'][0]['Points2D']:
				img = cv2.circle(img, (int(point['X']), int(point['Y'])), radius=3, color=(0, 0, 255), thickness=-1)
			images_to_display.append(img)

		data_disto 	= client_disto.read()
		index_disto = 0
		img_disto, index_disto, width, height = ALSFunc.ReadImage_Stream(data_disto, index_disto)
		if index_disto < len(data_disto) :
			extra_string, index_disto = ALSFunc.ReadString(data_disto,index_disto)
			parsed_string 			  = json.loads(extra_string)
			time_disto 				  = float(parsed_string['T'])
			print(parsed_string)

			brown = ALSImg.BrownParams().Load2(parsed_string['BrownModel'], width, height)
			
			points 			 = ParsePoint2d(parsed_string['TrackedObj'][0]['Points2D'])
			distorted_points = brown.DoUndistortPixel(points)
			for point in distorted_points:
				img_disto 	 = cv2.circle(img_disto, (int(point[0]), int(point[1])), radius=3, color=(0, 255, 255), thickness=-1)

			images_to_display.append(img_disto)		
		
		added_image = cv2.addWeighted(img,0.3,img_disto,0.5,0)

		###############
		# This just makes sure we are in sync (after the first frame)
		###############
		while time_disto < time_nodisto:
			data_disto 	= client_disto.read()
			index_disto = 0
			img_disto, index_disto, width, height = ALSFunc.ReadImage_Stream(data_disto, index_disto)
			if index_disto < len(data_disto) :
				extra_string, index = ALSFunc.ReadString(data_disto,index)
				parsed_string 		= json.loads(extra_string)
				time_disto 			= float(parsed_string['T'])
		while time_nodisto < time_disto:
			data = client.read()
			index = 0
			img, index, width, height = ALSFunc.ReadImage_Stream(data, index)
			if index < len(data) :
				extra_string, index = ALSFunc.ReadString(data,index)
				parsed_string = json.loads(extra_string)
				time_nodisto = float(parsed_string['T'])

		print(":::::::::::")
		added_image = ALSImg.StackImages(images_to_display)
		ALSImg.JustDisplay(added_image)

