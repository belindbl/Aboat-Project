import threading
import ALSLib.TCPClient, ALSLib.ALSClient


###############################################################################
# This demo shows how to collect the OccupancyGrid data
#
# We use a premade SensorProfile where all the sensors are pre-configured, but
# disabled so we use overrides to only enable the sensor we want in each case
###############################################################################

HOST = '127.0.0.1'

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
	client = ALSLib.ALSClient.Client((HOST, simulation_control_port), myMessageHandler)

if __name__ == "__main__":
	TestContext.client.connect()
	TestContext.client.request_load_scenario('Default_Scenario')

	sensorprofile_path = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini"
	sensor_path_in_file = "Sensors.Sensors.[5]"
	overrides = "{file_path};{sensor_path}.StreamToNetwork;True".format(file_path=sensorprofile_path, sensor_path=sensor_path_in_file)
	TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)

	occupancy_grid_port = 8885
	client = ALSLib.TCPClient.TCPClient(HOST, occupancy_grid_port, 5 )
	client.connect(5)
	while(True):
		data = client.read()

		datastring = data.decode('ascii')
		splitdata = datastring.split(";")
		time = float(splitdata[0])
		x = splitdata[1]
		y = splitdata[2]
		z = splitdata[3]
		roll = splitdata[4]
		pitch = splitdata[5]
		yaw = splitdata[6]
		states = splitdata[7]

		array = [state == '1' for state in states]
		print('time:{} x:{} y:{} z:{} roll:{} pitch:{} yaw:{} len:{}\ndata:{}'.format(time, x, y, z, roll, pitch, yaw, len(array), array))



