import time
import threading
import logging
import ALSLib.ALSClient 

###############################################################################
# This script loads the same scene twice: 
# First as default
# second using the override file that can make significant changes to any
# part of the ini files
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

	print("First load the scene with default values...")
	TestContext.client.request_load_scenario('OpenOceanEgo')

	time.sleep(2)
	print("Then we reload the same scenario, but this time with overrides...")
	TestContext.client.request_load_scenario_with_overrides('OpenOceanEgo', '/Overrides/Overrides.csv')

	
print("Starting the demo")
LaunchDemo()



print("--- end of script ---")
