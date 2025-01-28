import time
import threading
import logging
import ALSLib.ALSClient 

###############################################################################
# This script load several environments both as subscene layers 
#
# This script does it 10 times in a row
###############################################################################

_L = logging.getLogger(__name__)
_L.setLevel(0)

def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')
	#print ('\n---> recieved Raw message: '+str)
	
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


def EnvironmentDemoScenario():
	TestContext.client.connect()
	print("loading full scene")
	TestContext.client.request_load_scenario('OpenOceanFullScene')

	time.sleep(10)
	print("Removing background")
	TestContext.client.request_load_sub_scene('Landscape', False)
	time.sleep(10)
	print("Switching middleground")
	TestContext.client.request_load_sub_scene('Coast_SkyScrapers', False)
	TestContext.client.request_load_sub_scene('Coast_SmallHouses', True)
	time.sleep(10)
	print("Removing Harbour")
	TestContext.client.request_load_sub_scene('Harbour', False)
	TestContext.client.request_load_sub_scene('Containers', False)
	time.sleep(7)
	print("Removing Small houses")	
	TestContext.client.request_load_sub_scene('Coast_SmallHouses', False)

	time.sleep(1)
	result = TestContext.client.request_toggle_pause()
	result = TestContext.client.request_destroy_situation()
	result = TestContext.client.request_toggle_pause()




print("Environment Demo ")
EnvironmentDemoScenario()



print("--- end of script ---")
