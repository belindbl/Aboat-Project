import threading, re
import ALSLib.TCPClient, ALSLib.ALSClient 
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path

###############################################################################
# This demo shows how to collect the GNSS data
#
# We use a premade SensorProfile where all the sensors are pre-configured, but 
# disabled so we use overrides to only enable the sensor we want in each case
#
# All ports and IDs are defined this way in the particular Sensorprofile
# and can be changed in other use cases
#
###############################################################################

HOST = '127.0.0.1'


savepath = get_sensordata_path('/GPS')

class GNSS_datapoint():
	def __init__(self, time, latitude, longitude, altitude, health):
		self.time = time
		self.latitude = latitude
		self.longitude = longitude
		self.altitude = altitude
		self.health = health

	@staticmethod
	def msg_to_datapoint(msg):
		pattern = r"T:(-?\d*.\d*);lat:(-?\d*.\d*);lon:(-?\d*.\d*);alt:(-?\d*.\d*);hea:(-?\d*)"
		compiled = re.compile(pattern)
		match = compiled.search(msg)
		if (match.lastindex == 5):
			return GNSS_datapoint(float(match[1]), float(match[2]),
			float(match[3]), float(match[4]), int(match[5]))
		else:
			print('Bad datapoint')
			return GNSS_datapoint(0,0,0,0,0)
		
	def to_array(self):
		array = [self.time, self.latitude, self.longitude, self.altitude, self.health]
		return array


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
	simulation_control_port = 9000
	client = ALSLib.ALSClient.Client((HOST, simulation_control_port),myMessageHandler)

if __name__ == "__main__":
	TestContext.client.connect()
	TestContext.client.request_load_scenario('Default_Scenario')

	sensorprofile_path = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini"
	sensor_path_in_file = "Sensors.Sensors.[11]"
	overrides = "{file_path};{sensor_path}.StreamToNetwork;True".format(file_path=sensorprofile_path, sensor_path=sensor_path_in_file)
	TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)

	gps_port = 8891
	client = ALSLib.TCPClient.TCPClient(HOST, gps_port, 5 )
	client.connect(5)

	gpsFileContent = "latitude;longitude;altitude;health\n"
	while True:
		data = client.read()

		datapoint = GNSS_datapoint.msg_to_datapoint(data.decode('utf8'))
		array = datapoint.to_array()

		gpsFileContent =  "{lat}; {lon}; {alt}; {health}\n".format(lat = array[1], lon = array[2],
		alt = array[3], health = array[4])
		print(gpsFileContent)
