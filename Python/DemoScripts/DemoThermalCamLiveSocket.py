#pylint: disable=R, W0401, W0614, W0703
import struct
import logging as log
import numpy as numpy
import cv2
import threading
import ALSLib.ALSClient 
import ALSLib.TCPClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc

###############################################################################
# This script loads the Baltic scene and a boat that has a thermal camera.
# After a while the sea temperature will be set to Hot
# Make sure that the camera has "Stream to Network" enabled
# 
# Note that not all levels support thermal camera.
###############################################################################


BUFF = 1024
HOST = '127.0.0.1'
PORT = 8880
log.basicConfig(format='%(levelname)s:%(message)s',level=log.DEBUG)


def JustDisplay(image, showImage= True):
    if showImage:
        try:
            import numpy as np
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('img', image)

        except Exception as e:
            print("Unable to show image: "+str(e))

def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')
	print ('\n---> recieved Raw message: '+str)
	
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



if __name__ == "__main__":

    #Establish connection with simulation controller
    TestContext.client.connect()

    TestContext.client.request_load_scenario('BalticNoon')
    TestContext.client.request_load_situation('BalticThermal')


    #Establish connection with thermal camera socket
    log.debug( "Connecting to  " + HOST + " " + str(PORT) )
    client = ALSLib.TCPClient.TCPClient(HOST, PORT, 5 )
    client.connect(5)
    log.debug( "connected %i", client.connected() )
    log.debug('waiting for message')

    # Overrides for switching sea temperature
    overrides = ""
    overrides += "SeaStates\\BalticDefault.SeaState.ini;Color.OceanTemperatureName;Hot\n"
    overrides = ALSFunc.ConvertStringToParameterFormat(overrides)

    frames = 0
    while(True):
        if frames > 60:
            TestContext.client.request_load_sea_state_with_overrides("BalticDefault", overrides)
        
        data = client.read()

        index = 0
        getval = lambda index: struct.unpack('<L', data[index*4:index*4+4])[0]

        image_height = getval(index)
        image_width = getval(index+1)
        image_channels = getval(index+2)
        image_data = data[3*4:]
        array = numpy.frombuffer(image_data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (image_height, image_width, image_channels))
        
        JustDisplay(array)
        cv2.waitKey(5) & 0xFF # allows OpenCV to update the current image
        frames += 1
