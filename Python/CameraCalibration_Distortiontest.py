import cv2
import numpy as np
import math, threading, struct, json
import ALSLib.ALSClient, ALSLib.TCPClient
from PIL import Image


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

class BrownParams:
	k1 = 0
	k2 = 0
	k3 = 0
	p1 = 0
	p2 = 0
	def __init__(self, k1, k2, p1, p2, k3 = 0.0):
		self.k1, self.k2, self.k3, self.p1, self.p2 = k1, k2, k3, p1, p2


def DoDistortImage(src_img, image_h_fov, brown_params):
	#defining the coefficients for the distortion
	distCoeff = np.zeros((4,1),np.float64)
	distCoeff[0,0] = brown_params.k1
	distCoeff[1,0] = brown_params.k2
	distCoeff[2,0] = brown_params.p1
	distCoeff[3,0] = brown_params.p2
	distCoeff[4,0] = brown_params.k3

	width  = src_img.shape[1]
	height = src_img.shape[0]
	p = height/width
	f = (float(width)/2.0) / math.tan(math.radians(image_h_fov/2.0)) #Calculating the focal length from the FoV

	# Assume unit matrix for camera
	cam = np.eye(3,dtype=np.float32)
	cam[0,2] = width/2.0  # define center x
	cam[1,2] = height/2.0 # define center y
	cam[0,0] = f       # define focal length x
	cam[1,1] = f * p   #(vFov / fov)  # define focal length y scaled to vertical fov

	# Creating the distorted image 
	dst = cv2.undistort(src,cam,distCoeff)
	return dst

def ReadUint32(buffer, index):
	return (int(struct.unpack('<L', buffer[index:index+4])[0]), index + 4)


def DecodeImageData(image_data, image_width, image_height, image_channels):
	array = np.frombuffer(image_data, dtype=np.dtype("uint8"))
	array = np.reshape(array, (image_height, image_width, image_channels))
	return array

def JustDisplay(image,  showImage= True):
	if showImage:
		try:
			cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			cv2.imshow('img', image)
			cv2.waitKey(500)
		except Exception as e:
			print("Unable to show image: "+str(e))


def BrownianDistortion(coord):
	# return coord
	k1 = -0.05
	k2 = -0.05
	k3 = 0.0
	p1 = -0.0
	p2 = -0.0


	normCoord = ((coord[0] / 720.0),(coord[1] / 480.0))
	normCoord = (normCoord[0]*2.0, normCoord[1]*2.0,)
	normCoord = (normCoord[0] -1.0, normCoord[1] -1.0)
	# normCoord = (normCoord[0] -0.5, normCoord[1] -0.5)

	r2 = np.linalg.norm(normCoord)
	r=math.dist(normCoord, (0.0,0.0))
	x = float(normCoord[0])
	y = float(normCoord[1])

	# //f=1+k1*r^2 + k2*r^4 + k3*r^6
	f = 1.0 + (k1 * (r*r)) + (k2*(r*r*r*r)) + (k3*(r*r*r*r*r*r))
	
	ud = normCoord[0] * f
	vd = normCoord[1] * f
	
	# //fx = 2p1 * x * y+p2*(r^2+2*x^2)
	fx =  (2.0 * p1 * x *y) + (p2 * ((r*r) + 2.0* (x*x)))
	# //fy=p1*(r^2+2*y^2)+2*p2*x*y
	fy = p1 * ((r*r) + (2.0 * (y*y))) + (2.0 * p2 * x * y)

	ud += fx
	vd += fy

	ud = (ud * 0.5) + 0.5
	vd = (vd * 0.5) + 0.5

	# ud += 0.5
	# vd += 0.5

	return (ud * 720, vd * 480)

def Undistort2(idealX, idealY):
	k1 = 0.1
	k2 = 0.1
	k3 = 0.0
	p1 = 0.05
	p2 = 0.05
	height, width, HorizontalFoV = 480, 720, 90.0
	p = height/width
	f = (float(width)/2.0) / math.tan(math.radians(HorizontalFoV/2.0)) #Calculating the focal length from the FoV
	ux, uy=width*0.5, height*0.5
	fx, fy  = f, f*p
	x, y = (idealX/width)*2-1, (idealY/height)*2-1
		# normCoord = (normCoord[0] -0.5, normCoord[1] -0.5)

	r2 = x*x + y*y
	xCorrected = x * (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
	yCorrected = y * (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
	xCorrected = xCorrected + (2. * p1 * x * y + p2 * (r2 + 2. * x * x))
	yCorrected = yCorrected + (p1 * (r2 + 2. * y * y) + 2. * p2 * x * y)
	xCorrected = xCorrected * fx + ux
	yCorrected = yCorrected * fy + uy
	return (xCorrected, yCorrected)


def Undistort3(idealX, idealY):
	k1,k2,k3,p1,p2 = 0.1, 0.1, 0.0, 0.05, 0.05
	height, width, HorizontalFoV = 480, 720, 90.0
	p = height/width
	f = (float(width)/2.0) / math.tan(math.radians(HorizontalFoV/2.0)) #Calculating the focal length from the FoV
	ux, uy=width*0.5, height*0.5
	fx, fy  = f, f*p
	x, y = (idealX/width)*2-1, (idealY/height)*2-1
	#x, y = idealX, idealY

	cam = np.eye(3,dtype=np.float32)
	cam[0,2] = width/2.0  # define center x
	cam[1,2] = height/2.0 # define center y
	cam[0,0] = f       # define focal length x
	cam[1,1] = f * p   #(vFov / fov)  # define focal length y scaled to vertical fov

	result = cv2.undistortPoints((x,y),cam, (k1,k2,p1,p2))
	print(".........", result[0][0] )
	final = (-result[0][0][0]*width,-result[0][0][1]*height) 
	return final

if __name__ == "__main__":
	TestContext.client.connect()
	TestContext.client.request("LoadSituation CameraCalibrationSituation_CollectGT")

	#get camera data (undistorted)
	non_distorted_port 	= 8881
	distorted_port 		= 8880
	client = ALSLib.TCPClient.TCPClient("localhost", distorted_port, 5)
	client2 = ALSLib.TCPClient.TCPClient("localhost", non_distorted_port, 5)

	client.connect(5)
	client2.connect(5)

	while(True):
		data = client.read()
		index = 0

		image_height, 	index	= ReadUint32(data, index)
		image_width	, 	index	= ReadUint32(data, index)
		image_channels, index 	= ReadUint32(data, index)
		image_size = image_height * image_width * image_channels
		image_data = data[index:index + image_size]
		index += image_size
		img = DecodeImageData(image_data, image_width, image_height, image_channels)

		data2 = client2.read()
		index2 = index

		
		if index < len(data) :
			extra_string_len, 	index	= ReadUint32(data, index)
			extra_string_len2, 	index2	= ReadUint32(data2, index2)

			extra_string = data[index : ]
			extra_string2 = data2[index2 : ]
			extra_string = extra_string.decode('utf-8')
			extra_string2 = extra_string2.decode('utf-8')
			parsed_string = json.loads(extra_string)
			parsed_string2 = json.loads(extra_string2)

			for point in parsed_string['Points2D']:
				img = cv2.circle(img, (int(point['X']), int(point['Y'])), radius=5, color=(0, 0, 255), thickness=-1)

			for point2 in parsed_string2['Points2D']:
				distorted = BrownianDistortion((10,10))
				distorted2 = BrownianDistortion((710,470))
				distorted3 = BrownianDistortion((10,470))
				distorted4 = BrownianDistortion((710,10))
				distorted5 = BrownianDistortion((720/2,480/2))
				# distorted6 = BrownianDistortion((int(point2['X']), int(point2['Y'])))
				# distorted6 = Undistort2(int(point2['X']), int(point2['Y']))
				distorted6 = Undistort3(int(point2['X']), int(point2['Y']))
				print( distorted6)
				# print(distorted)
				img2 = cv2.circle(img, (int(distorted[0]), int(distorted[1])), radius=5, color=(0, 255, 255), thickness=-1)
				img2 = cv2.circle(img2, (int(distorted2[0]), int(distorted2[1])), radius=5, color=(0, 255, 255), thickness=-1)
				img2 = cv2.circle(img2, (int(distorted3[0]), int(distorted3[1])), radius=5, color=(0, 255, 255), thickness=-1)
				img2 = cv2.circle(img2, (int(distorted4[0]), int(distorted4[1])), radius=5, color=(0, 255, 255), thickness=-1)
				img2 = cv2.circle(img2, (int(distorted5[0]), int(distorted5[1])), radius=5, color=(0, 255, 255), thickness=-1)
				img2 = cv2.circle(img2, (int(distorted6[0]), int(distorted6[1])), radius=5, color=(255, 0, 255), thickness=-1)


		JustDisplay(img)

