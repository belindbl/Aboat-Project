#It is possible to collect data without reloading the whole scenario,
#when changes are situational. Weather conditions, sea state and situations
#can be loaded, when sensor thread manager has finished collecting the previous batch of samples,
#and receiving has paused.

import os.path as ospath
import threading, ALSLib.ALSClient
from ALSLib.ALSHelperDataCollection import SensorThreadManager, SensorOptions, load_override_smart, ManagerState
from ALSLib.ALS_TestDefinitionGenerator import generate_tests
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path, get_config_path, find_unique_directory_name

HOST = "127.0.0.1"

def auto_message_handler(x:str):
	pass

class Context:
	client = ALSLib.ALSClient.Client((HOST, 9000), auto_message_handler)

#=============================================================================================

if __name__ == "__main__":
	unique_dir_name = find_unique_directory_name("TEST-STR")

	#Sensor options can be used to change the settings for collecting the samples.
	options = SensorOptions()
	options.out_dir = unique_dir_name

	#When the clean dir is set to false, the directory will not be cleaned
	#from previous samples.
	options.clear_dir = False

	#Burn in samples amount tells the manager to discard number of samples before
	#starting actual collection. This might be necessary if scenario is complex, and simulation
	#needs time to fully be stable.
	options.burn_in_samples = 1
	
	#Batch limit controls the amount of samples to collect.
	
	options.batch_limit = 3
	
	#When save metadata is true, the samples will contain also a metadata in a json file.
	#This is the default value of the library.
	options.save_metadata = True

	#Add one or more scenarios that will be loaded in the loop below that have an ego situation
	scenarios = ["DemoDatasetOverrideScenario"]

	#Add the name and path of the overrides text file
	overridepath = get_config_path("/SituationOverrides/DemoDatasetSmartOverride.txt")

	assert ospath.isdir(get_sensordata_path())

	#Generate the overrides from the txt-file
	overrides = generate_tests(overridepath)
	overrides = [override.split(" ",1)[1] for override in overrides]

	#Create sensor thread manager, which will create receiving data and saving threads to collect the data to the disk.
	mgr = SensorThreadManager()

	print(overrides[0])

	Context.client.connect(30)
	mgr.log_enable(True)
	#Purpose for this is to represent four different time using Weather overrides.
	for i, scenario in enumerate(scenarios):
		Context.client.request_load_scenario(scenario)
		mgr.auto_setup(Context.client, options, "{sensor_type}{sensor_number}__{frame_id}")
		for j, override in enumerate(overrides):
			print(f"Loading override {j + 1}/{len(overrides)}")
			load_override_smart(override, Context.client)
			mgr.collect_samples_until_paused(delta_time=.05)
		mgr.stop()
	Context.client.disconnect()
