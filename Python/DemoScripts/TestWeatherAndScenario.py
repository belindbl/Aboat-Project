import time
import threading
import logging
from datetime import datetime
import ALSLib.ALSClient
import ALSLib.TCPClient
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path


###############################################################################
# here we demonstrate some of the weather API
###############################################################################

_L = logging.getLogger(__name__)
_L.setLevel(0)

def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')
	
	cmdList = str.split(" ")
	if cmdList[0].startswith("EndCondition") :
		TestContext.client.request("DestroySituation")

		TestContext.lock.acquire()
		TestContext.testEnded = True
		print("setting TestEnded")
		TestContext.lock.release()
	# elif cmdList[0].startswith("[SkySettings]") or cmdList[0].startswith("[SeasonDefaultSettings]"):
	#     with open(r"../SensorData/file.weather.ini", "w+") as file:
	#         file.write(rawMessage.decode("utf8"))
	#     print(rawMessage)
	elif(cmdList[0].startswith("<?xml")):
		print(rawMessage)

class TestContext:
	lock = threading.Lock()
	testEnded = False
	echo_port = 9000
	localhost = 'localhost'
	client = ALSLib.ALSClient.Client((localhost, echo_port),myMessageHandler)
	PORT = 7700
	HOST = '127.0.0.1'

def TestScenario():
	TestContext.client.connect()

	TestContext.client.request('GetScenario')
	
	# weatherMessage = TestContext.client.request('GetWeatherStatus')
	# print(weatherMessage)
	# with open(r"../SensorData/file1.weather.ini", "w+", newline="") as file:
	#    file.write(weatherMessage)

	seastateMessage = TestContext.client.request('GetSeaState')
	print(seastateMessage)
	with open(get_sensordata_path("/file1.seastate.ini"), "w+", newline="") as file:
		file.write(seastateMessage)

	time.sleep(5)

	result = TestContext.client.response

	print(result)
	TestContext.client.request_set_rain_intensity(0.5)
	time.sleep(2)
	TestContext.client.request_set_rain_intensity(1)
	time.sleep(10)
	TestContext.client.request_set_rain_type("snow")
	time.sleep(50)
	TestContext.client.request_set_rain_type("normal")
	time.sleep(50)
	# TestContext.client.request_set_rain_type("hail")
	# time.sleep(5)
	# TestContext.client.request_set_rain_type("leaves")
	# time.sleep(5)
	TestContext.client.request_set_rain_type("norain")
	time.sleep(2)
	TestContext.client.request_set_rain_intensity(0.0)
	time.sleep(2)

	TestContext.client.request_set_ground_wetness_type("wet")
	time.sleep(2)
	TestContext.client.request_set_ground_wetness_type("snowy")
	time.sleep(2)
	TestContext.client.request_set_ground_wetness_type("dry")

	dt = datetime(2021, 4, 1, 8, 0, 0, 0)
	t = 8
	for i in range(12):
		print(t)
		dt = datetime(2021, 4, 1, t, 0, 0, 0)
		TestContext.client.request_set_time_of_day(dt.isoformat())
		time.sleep(0.5)
		t += 1

	TestContext.client.request_set_cloud_style("Clear")
	time.sleep(3)
	TestContext.client.request_set_cloud_style("Cloudy")
	time.sleep(3)
	TestContext.client.request_set_cloud_style("Overcast")
	time.sleep(3)
	TestContext.client.request_set_cloud_style("StormClouds")
	time.sleep(3)
	TestContext.client.request_set_cloud_style("Cloudy")

	# TestContext.client.request('RandomizeWeather')
	
TestScenario()
print("--- end of script ---")
