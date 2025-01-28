import threading
import logging
import ALSLib.ALSClient

###############################################################################
# This script loads the Baltic scene and a Situation 
###############################################################################


_L = logging.getLogger(__name__)
_L.setLevel(0)

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


def DriveableDemo():
	TestContext.client.connect()

	TestContext.client.request_load_scenario('BalticNoon')
	TestContext.client.request_load_situation('BalticFerryFollow')
	

	
print("Starting the demo")
DriveableDemo()



print("--- end of script ---")
