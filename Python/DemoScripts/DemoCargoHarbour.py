import threading
import logging
import ALSLib.ALSClient

###############################################################################
# This script loads the OpenOcean level with Harbour scene and a Situation 
###############################################################################


_L = logging.getLogger(__name__)
_L.setLevel(0)

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


def LaunchDemo():
	TestContext.client.connect()

	TestContext.client.request_load_scenario('OpenOceanNaked')
	TestContext.client.request_load_sub_scene('Harbour', True)
	TestContext.client.request_load_sub_scene('Containers', True)
	TestContext.client.request_load_situation('OpenOceanCargoBoat')
	

	
print("Starting the demo")
LaunchDemo()



print("--- end of script ---")
