import numpy as np
import PIL.Image
from io import BytesIO
import ALSClient 
import time
import threading
import logging

###############################################################################
# This simple script calls the help, and destroys the situation
# ###############################################################################

def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')
	print(str)

class TestContext:
	lock = threading.Lock()
	testEnded = False
	echo_port = 9000
	localhost = 'localhost'
	client = ALSClient.Client((localhost, echo_port),myMessageHandler)

def Cleanup():
	TestContext.client.connect()
	TestContext.client.request_destroy_situation()

Cleanup()
