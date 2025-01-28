import threading
import logging
import random
import time
from ALSLib.ALSHelperFunctionLibrary import ConvertStringToParameterFormat
import ALSLib.ALSClient 

_L = logging.getLogger(__name__)
_L.setLevel(0)

###############################################################################
# This script demonstrates how to use the object specific API commands
###############################################################################

def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')
	#print ('\n---> recieved Raw message: '+str)
	if not TestContext.testEnded:
		cmdList = str.split(" ")
		if len(cmdList) > 1:
			if cmdList[1].startswith("RouteFinished") :
				TestContext.route_finished = True
			elif cmdList[0].startswith("EndCondition") :
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
	route_finished = False

def GetLocationFromString(LocationString):
	values = LocationString.split(" ")
	for v in values:
		exec(v)
	location = [locals()['X'],locals()['Y'],locals()['Z']]
	return location


def DemoScenario():
	TestContext.client.connect()
	TestContext.client.request_load_scenario('OpenOceanFullScene', 999)
	TestContext.client.request_load_situation_layer("TestObjectAPICommands")

	TextPath = "InfoText"

	TestContext.client.execute_set_object_property(TextPath, "TextToDisplay", ConvertStringToParameterFormat("Changing Text Color"))
	for i in range(10):
		r = random.random()
		g = random.random()
		b = random.random()
		a = 1.0
		colorstring = "(R={Red:.6f},G={Green:.6f},B={Blue:.6f},A={Alpha:.6f})".format(Red=r, Green=g, Blue=b, Alpha=a)
		TestContext.client.request_set_object_property(TextPath, "TextColor", colorstring)
		time.sleep(1)


	ObjectPath1 = "Buoy1"
	ObjectPath2 = "Buoy2"

	TestContext.client.execute_set_object_property(TextPath, "TextToDisplay", ConvertStringToParameterFormat("Teleporting buoys"))

	# Disable physics for buoys so they don't fall after teleporting
	TestContext.client.request_set_physics_active("Buoy1", False)
	TestContext.client.request_set_physics_active("Buoy2", False)


	Buoy1Location = TestContext.client.request_get_object_location("Buoy1")
	Buoy2Location = TestContext.client.request_get_object_location("Buoy2")
	B1Location = GetLocationFromString(Buoy1Location)
	B2Location = GetLocationFromString(Buoy2Location)

	Filename = "TestObjectAPICommands"

	# # Teleport buoys into the air
	TestContext.client.request_teleport_object(ObjectPath1, B1Location[0], B1Location[1], B1Location[2] + 1000, Filename)
	TestContext.client.request_teleport_object(ObjectPath2, B2Location[0], B2Location[1], B2Location[2] + 1000, Filename)

	TestContext.client.execute_set_object_property(TextPath, "TextToDisplay", ConvertStringToParameterFormat("Changing buoy colors"))

	time.sleep(3)
	# Hide buoys from view
	TestContext.client.request_set_object_visibility("Buoy1", False)
	TestContext.client.request_set_object_visibility("Buoy2", False)


	time.sleep(2)
	TestContext.client.execute_set_object_property(TextPath, "TextToDisplay", ConvertStringToParameterFormat("Dropping buoys"))
	# Dropping the buoys
	TestContext.client.request_set_object_visibility("Buoy1", True)
	TestContext.client.request_set_object_visibility("Buoy2", True)
	TestContext.client.request_set_physics_active("Buoy1", True)
	TestContext.client.request_set_physics_active("Buoy2", True)

	TestContext.client.execute_set_object_property(TextPath, "TextToDisplay", ConvertStringToParameterFormat("Waiting to finish first route"))
	while not TestContext.route_finished:
		time.sleep(.5)
	BoatPath = "ObjectList.ObjectName.[3]"
	Route2Location = GetLocationFromString(TestContext.client.request_get_object_location("Route2"))
	TestContext.client.request_teleport_object(BoatPath, Route2Location[0], Route2Location[1], Route2Location[2] + 1000, Filename)
	
	TestContext.client.execute_set_object_property(TextPath, "TextToDisplay", ConvertStringToParameterFormat("Waiting to finish second route"))
	# Wait for the second route to be finished
	while not TestContext.testEnded:
		time.sleep(.5)

	time.sleep(5)
	# Clean up after the test is done
	TestContext.client.request_destroy_situation()



print(" Demo ")
DemoScenario()
print("--- end of script ---")
