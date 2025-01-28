import time
import threading
import logging
import ALSLib.ALSClient 

###############################################################################
# This script loads the Baltic scene and a Situation that has a boat
# 
# Also there are several triggers in the scene
# When a trigger fires up, the associated even get sent to the message handler
# The message is configurable inside the trigger (and inside the config file)
# 
# In this demo we only print the trigger, but you can do other reactions, like
# call "destroy_situation" or anything else..
###############################################################################


_L = logging.getLogger(__name__)
_L.setLevel(0)


def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')

	cmdList = str.split(" ")
	if cmdList[0].startswith("EndCondition") :
		
		print("Trigger received: ", cmdList[1])
		#TestContext.client.request_destroy_situation()

		#TestContext.lock.acquire()
		#TestContext.testEnded = True
		#print("setting TestEnded")
		#TestContext.lock.release()

	

class TestContext:
	lock = threading.Lock()
	testEnded = False
	echo_port = 9000
	localhost = 'localhost'
	client = ALSLib.ALSClient.Client((localhost, echo_port),myMessageHandler)


def EnvironmentDemoScenario():
	TestContext.client.connect()
	print("loading full scene")
	TestContext.client.request_load_scenario('OpenOceanNaked')

	time.sleep(1)
	print("Drive towards the boat to hit the triggers")
	TestContext.client.request_load_situation('OpenOceanTestTriggers')

	time.sleep(30)

	TestContext.client.request_toggle_pause()
	TestContext.client.request_destroy_situation()
	TestContext.client.request_toggle_pause()




print("Environment Demo ")
EnvironmentDemoScenario()
print("--- end of script ---")
