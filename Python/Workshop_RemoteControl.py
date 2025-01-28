import csv, shutil,json, os, time, threading
import numpy as np
import ALSLib.ALSClient, ALSLib.TCPClient
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path

ALS_SERVER_IP_ADDRESS = '127.0.0.1'

###############################################################################
class SensorDataThread( threading.Thread):
	def __init__(self, ThreadID, client, function):
		threading.Thread.__init__(self)
		self.threadID = ThreadID
		self.client = client
		self.function = function
	def run(self):
		print ("Starting ", self.threadID,"\n")
		self.function(self.client)
		print ("Exiting ", self.threadID,"\n")

###############################################################################
class ThreadedSensorRecieve:
	def __init__(self, sensorDefinitions):
		self.sensorThreads = []
		[self.addSensor(sensor[0], sensor[1], sensor[2])
		 	for sensor in sensorDefinitions]
		self.startAllThreads()
	def addSensor(self, socketAddress, socketPort, recieveFunction):
		socket = ALSLib.TCPClient.TCPClient(socketAddress, socketPort, 15 )
		socket.connect(10)
		thread = SensorDataThread(len(self.sensorThreads), socket, recieveFunction)
		self.sensorThreads.append([socket, thread])
	def startAllThreads(self):
		[thread[1].start() for thread in self.sensorThreads]
	def waitAllThreads(self):
		[thread[1].join() for thread in self.sensorThreads ]

###############################################################################
class VehicleStatus:
	def __init__(self):
		self.SimulationTime 		= 0.0
		self.StatusString 			= ''
		self.CurrentDestination 	= [0.0, 0.0, 0.0]
		self.CurrentPosition		= [0.0, 0.0, 0.0]
		self.num_waypoints_reached 	= 0
		self.num_waypoints_sensed 	= 0
		self.current_target_waypoint= -1
	def Copy(self, other):
		self.SimulationTime 		= other.SimulationTime
		self.StatusString 			= other.StatusString
		self.CurrentDestination 	= other.CurrentDestination
		self.CurrentPosition 		= other.CurrentPosition
		self.num_waypoints_reached 	= other.num_waypoints_reached
		self.num_waypoints_sensed 	= other.num_waypoints_sensed
		self.current_target_waypoint= other.current_target_waypoint

###############################################################################
def messageHandlerBoat(rawMessage):
	str = rawMessage.decode('utf-8')
	commandList = str.split(" ")
	if commandList[0].startswith('EndCondition'):
		if( commandList[1].startswith("TimeEnded")):
			SimulationContext.lock.acquire()
			SimulationContext.should_end_simulation = True
			print("Timeout recieved .. declaring end of test")
			SimulationContext.lock.release()
		if commandList[1].startswith("Waypoint"):
			print(commandList[2], "reached", commandList[4] + ":", commandList[1])
			if SimulationContext.last_waypoint != commandList[1]:
				SimulationContext.last_waypoint = commandList[1]
				SimulationContext.vehicleStatus.num_waypoints_reached += 1
				SimulationContext.client.execute("SetObjectProperty StatusText TextToDisplay Reached_{0}_waypoints".format(SimulationContext.vehicleStatus.num_waypoints_reached))
				SimulationContext.vehicleStatus.current_target_waypoint = (SimulationContext.vehicleStatus.current_target_waypoint+1)%SimulationContext.vehicleStatus.num_waypoints_sensed
				chosenPos = SimulationContext.waypoint_list[SimulationContext.vehicleStatus.current_target_waypoint]
				print("New Destination chosen: Waypoint",SimulationContext.vehicleStatus.current_target_waypoint, ", pos: ", chosenPos)

	if commandList[0].startswith('Status'):
		SimulationContext.lock.acquire()
		SimulationContext.vehicleStatus.SimulationTime = float(commandList[1])
		SimulationContext.vehicleStatus.StatusString = commandList[2]
		SimulationContext.lock.release()

###############################################################################
class SimulationContext:
	lock = threading.Lock()
	should_end_simulation = False
	ALS_simulation_control_port = 9000
	client = ALSLib.ALSClient.Client((ALS_SERVER_IP_ADDRESS, ALS_simulation_control_port),messageHandlerBoat)
	vehicleStatus = VehicleStatus()
	waypoint_list = []
	last_waypoint = ''

###############################################################################
def SteerTowardsGoal(position, forward, right, destination):
	d = np.asarray(destination)
	p = np.asarray(position)	
	targetDir = d-p
	distance = np.linalg.norm(targetDir)
	if distance != 0.0:
		targetDir = targetDir/distance
	else :
		targetDir = np.asarray([1.0,0.0,0.0])

	Throttle = 0
	Steering = np.dot(targetDir,right)	
	dot = np.dot(forward, targetDir)
	
	if  distance > 100 :
		if dot > 0:
			Throttle = 0.5
		else:
			Throttle = 0
			Steering = -1.0
	else:
		Throttle = distance / 100

	return (Throttle, Steering)

###############################################################################
def recieveObjectList(FilteredObject_Socket):
	def decodeJson(jsonstring):
		msg = json.loads(jsonstring)
		return msg

	while(True):
		try:
			data = FilteredObject_Socket.read()
		except :
			SimulationContext.lock.acquire()
			SimulationContext.should_end_simulation = True
			print("Connection closed: declaring end of test")
			SimulationContext.lock.release()
			break

		message 		= data.decode('utf-8')
		decoded 		= decodeJson(message)
		SimulationContext.vehicleStatus.num_waypoints_sensed = len(decoded["MSG"])
		#NOTE: we add the element in the order they arrive, but we could instead sort them by distance
		# when we add them.
		for num, element in enumerate(decoded["MSG"]):
			waypoint_pos = element["Pos"]
			if not waypoint_pos in SimulationContext.waypoint_list:
				SimulationContext.waypoint_list.append(waypoint_pos)
		
		#when starting up the first time we didn't pick a target yet
		if SimulationContext.vehicleStatus.current_target_waypoint < 0:
			SimulationContext.vehicleStatus.current_target_waypoint = 0
			chosenPos = SimulationContext.waypoint_list[SimulationContext.vehicleStatus.current_target_waypoint]
			print("First destination: ", chosenPos)

		chosenPos = SimulationContext.waypoint_list[SimulationContext.vehicleStatus.current_target_waypoint]
		SimulationContext.vehicleStatus.CurrentDestination = chosenPos
		SimulationContext.client.request('SpawnDebugSphere {pos1} {pos2} {pos3} 2.5 2.5 2.5 {color} 1'.format(pos1=chosenPos[0], pos2=chosenPos[1], pos3=chosenPos[2]+250, color="bluesphere"))

		if SimulationContext.should_end_simulation:
			break

###############################################################################
def LaunchWorkshop():
	SimulationContext.client.connect()

	print("Loading Turku, a boat and waypoints")
	SimulationContext.client.request('LoadScenario Workshop_BoatControl')
	SimulationContext.client.wait_for_task_complete()
	
	print("Connecting to the control socket")
	remoteControlSocket = ALSLib.TCPClient.TCPClient(ALS_SERVER_IP_ADDRESS, 7700, 5 )
	remoteControlSocket.connect(5)

	print("Starting the sensors collection Threads")
	sensorDefinintion = [
		[ALS_SERVER_IP_ADDRESS, 8880, recieveObjectList]
	]
	sensorThread = ThreadedSensorRecieve(sensorDefinintion)
	lastStatus = SimulationContext.vehicleStatus.SimulationTime

	datapath = get_sensordata_path('/Workshop/')
	if( os.path.exists(datapath)):
		shutil.rmtree(datapath, ignore_errors=True)
		time.sleep(1)
	os.mkdir(datapath)
	PositionLog=[['time', 'posX', 'posY', 'posZ', 'forwardX', 'forwardY', 'forwardZ']]
	with open(os.path.join(datapath, 'Positions.csv'), 'a') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(PositionLog)

	while True:
		if 	SimulationContext.should_end_simulation == True:
			print( 'end of test recieved')
			break
		time.sleep(0.05)

		if lastStatus != SimulationContext.vehicleStatus.SimulationTime:
			lastStatus 	= SimulationContext.vehicleStatus.SimulationTime
			parsedJson 	= json.loads(SimulationContext.vehicleStatus.StatusString)
			position 	= parsedJson['Position']
			forward 	= parsedJson['Forward']
			right 		= parsedJson['Right']
			SimulationContext.vehicleStatus.CurrentPosition = position

			currentEntry = [ SimulationContext.vehicleStatus.SimulationTime, position[0], position[1], position[2], forward[0],forward[1], forward[2]]
			PositionLog.append(currentEntry)
			with open(get_sensordata_path('/Workshop/Positions.csv'), 'a') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow(currentEntry)
			
		(throttle, steering) = SteerTowardsGoal(position, forward, right, SimulationContext.vehicleStatus.CurrentDestination)
		command_string = 'SetControl t:%f s:%f' %(throttle, steering)
		remoteControlSocket.write(command_string.encode('utf-8'))

	SimulationContext.client.execute('DestroySituation')
	sensorThread.waitAllThreads()

print("Start the demo ")

LaunchWorkshop()

print("--- end of script ---")
