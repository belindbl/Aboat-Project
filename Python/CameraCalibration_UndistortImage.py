import cv2
import numpy as np
import math
import ALSLib.ALSClient
import threading
import glob
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path

# def myMessageHandler(rawMessage):
# 	str = rawMessage.decode('utf-8')
# 	cmdList = str.split(" ")
# 	if cmdList[0].startswith("EndCondition") :
# 		TestContext.client.request("DestroySituation")

# 		TestContext.lock.acquire()
# 		TestContext.testEnded = True
# 		print("setting TestEnded")
# 		TestContext.lock.release()

# class TestContext:
# 	lock = threading.Lock()
# 	testEnded = False
# 	echo_port = 9000
# 	localhost = 'localhost'
# 	client = ALSLib.ALSClient.Client((localhost, echo_port),myMessageHandler)


# TestContext.client.connect()
# TestContext.client.request("LoadSituation CameraCalibrationSituation")

# #Defining the paths used to get the sensor data
# car = "Vehicle"
# camera = "Sensors.[0]"

# FoVString = TestContext.client.request("GetCameraFoV {Car} {Camera}".format(Car = car, Camera = camera))
# VerticalFovS, HorizontalFoVS = FoVString.split(";")
# print("Camera vertical Fov:{V}, horizontal Fov: {H}".format(V=VerticalFovS, H=HorizontalFoVS))
# HorizontalFoV = float(HorizontalFoVS)
# # *********************** DISTORT IMAGE**********************
# src   = cv2.imread(get_sensordata_path('/images/img.png')
# width  = src.shape[1]
# height = src.shape[0]

# #defining the coefficients for the distortion
# distCoeff = np.zeros((4,1),np.float64)
# k1 = 0.1
# k2 = 0.1
# p1 = 0.05
# p2 = 0.05
# distCoeff[0,0] = k1
# distCoeff[1,0] = k2
# distCoeff[2,0] = p1
# distCoeff[3,0] = p2
# p = height/width
# f = (float(width)/2.0) / math.tan(math.radians(HorizontalFoV/2.0)) #Calculating the focal length from the FoV

# # Assume unit matrix for camera
# cam = np.eye(3,dtype=np.float32)
# cam[0,2] = width/2.0  # define center x
# cam[1,2] = height/2.0 # define center y
# cam[0,0] = f       # define focal length x
# cam[1,1] = f * p   #(vFov / fov)  # define focal length y scaled to vertical fov

# # Creating the distorted image 
# dst = cv2.undistort(src,cam,distCoeff)
# cv2.imshow('dst',dst)
# cv2.imwrite(get_sensordata_path('/images/imgDistorted.png', dst)
# cv2.waitKey(500)
# cv2.destroyAllWindows()
# *****************************************************************


# *****************************************************************

# *********Calibrating the camera and undistorting an image********
# Setting up initial values
chessboardSize = (11,8)
frameSize=(720,480)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid [0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

images = glob.glob(get_sensordata_path('/images/Calibration/*.png'))
# images=[]
# imageAmount=37
# imagePath = 'C:\\Unreal_Projects\\AILiveSim\\SensorData\\images\\img{Number}.png'
# i = 3
# while i < imageAmount+1:
#     images.append(imagePath.format(Number = i))
#     i +=1

objPoints = []
imgPoints = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
#Calibrate based on checkerboard images
for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.waitKey(500)
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    if ret == True:
        objPoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners)
        # Show the corners found on the image 
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

#Printing out calibrated values
print("Camera calibrated", ret)
print("\nCameraMatrix:\n", cameraMatrix)
print("\nDistortionParameters:\n", dist)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)


src  = cv2.imread(get_sensordata_path('/images/img1.png'))
h, w = src.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

#Undistorting the image
dst = cv2.undistort(src, cameraMatrix, dist, None, newCameraMatrix)

#Cropping the undistorted image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(get_sensordata_path('/images/imgUndistorted.png'), dst)


cv2.imshow('dst', dst)
cv2.waitKey(1000)

#Calculating reprojection error
mean_error = 0

for i in range(len(objPoints)):
    imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgPoints[i], imgPoints2, cv2.NORM_L2)/len(imgPoints2)
    mean_error += error

print("\n Total error: {}".format(mean_error/len(objPoints)))
print("-"*20)