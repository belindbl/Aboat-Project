from io import BytesIO
import ALSClient 
import time
import threading
import logging


###############################################################################
# the reload all camera is handy when you are changing dynamically some 
# parameters of the cameras
###############################################################################

def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')
	print ('\n---> recieved Raw message: '+str)

class TestContext:
	lock = threading.Lock()
	testEnded = False
	echo_port = 9000
	localhost = 'localhost'
	client = ALSClient.Client((localhost, echo_port),myMessageHandler)

def ReloadCameras():
	TestContext.client.connect()
	TestContext.client.request_reload_all_cameras()

ReloadCameras()