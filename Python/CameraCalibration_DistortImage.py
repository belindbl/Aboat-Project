import cv2
import numpy as np
import math
import ALSLib.ALSClient
import threading
import glob
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path

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


TestContext.client.connect()
TestContext.client.request_load_situation("CameraCalibrationSituation")

#Defining the paths used to get the sensor data
car = "Camera"
camera = "Sensors.[0]"

FoVString = TestContext.client.request_get_camera_fov(car, camera)

VerticalFovS, HorizontalFoVS = FoVString.split(";")
print("Camera vertical Fov:{V}, horizontal Fov: {H}".format(V=VerticalFovS, H=HorizontalFoVS))
HorizontalFoV = float(HorizontalFoVS)
# *********************** DISTORT IMAGE**********************
src   = cv2.imread(get_sensordata_path('/images/img.png'))
width  = src.shape[1]
height = src.shape[0]

#defining the coefficients for the distortion
distCoeff = np.zeros((4,1),np.float64)
k1 = 0.1
k2 = 0.1
p1 = 0.05
p2 = 0.05
distCoeff[0,0] = k1
distCoeff[1,0] = k2
distCoeff[2,0] = p1
distCoeff[3,0] = p2
p = height/width
f = (float(width)/2.0) / math.tan(math.radians(HorizontalFoV/2.0)) #Calculating the focal length from the FoV

# Assume unit matrix for camera
cam = np.eye(3,dtype=np.float32)
cam[0,2] = width/2.0  # define center x
cam[1,2] = height/2.0 # define center y
cam[0,0] = f       # define focal length x
cam[1,1] = f * p   #(vFov / fov)  # define focal length y scaled to vertical fov

# Creating the distorted image 
dst = cv2.undistort(src,cam,distCoeff)
cv2.imshow('dst',dst)
cv2.imwrite(get_sensordata_path('/images/imgDistorted.png'), dst)
cv2.waitKey(500)
cv2.destroyAllWindows()
# *****************************************************************