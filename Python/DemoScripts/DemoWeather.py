import time
import threading
import logging
import ALSLib.ALSClient

_L = logging.getLogger(__name__)
_L.setLevel(0)

###############################################################################
# This shows how the API can be use to changes some individual weather 
# parameters
# use TestContext.client.request('help'), TestContext.client.request_help()
# or the documentation for more details
# about what can be done with weathers
###############################################################################


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


def EnvironmentDemoScenario():
	TestContext.client.connect()
	print("loading full scene")

	SleepTime = 5	
	time.sleep(SleepTime)
	print("EveningClear")
	TestContext.client.request_load_weather('EveningClear')
	time.sleep(SleepTime)
	print("MorningClear")
	TestContext.client.request_load_weather('MorningClear')
	time.sleep(SleepTime)
	print("NightClear")
	TestContext.client.request_load_weather('NightClear')
	time.sleep(SleepTime)
	print("FoggyCloudy")	
	TestContext.client.request_load_weather('FoggyCloudy')
	time.sleep(SleepTime)
	print("Default")
	TestContext.client.request_load_weather('Default')
	time.sleep(1)
	TestContext.client.request_toggle_pause()
	TestContext.client.request_destroy_situation()
	TestContext.client.request_toggle_pause()




print("Environment Demo ")
EnvironmentDemoScenario()



print("--- end of script ---")
