#Import SensorThreadManager from library module.
from ALSLib.ALSHelperDataCollection import SensorThreadManager
import ALSLib
from ALSLib import ALSClient


#Instantiate SensorThreadManager.
mgr = SensorThreadManager()

def auto_message_handler(x:str):
	pass

class Context:
	client = ALSLib.ALSClient.Client((HOST, 9000), auto_message_handler)

client = ALSClient.Client((HOST, 9000), messageHandler)
Context.client.connect(10)

#Here, the argument to the function must be an object of type ALSLib.ALSClient.Client.
#This is the simulation control socket.
mgr.auto_setup(client) 