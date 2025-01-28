#Data can be collected by loading a whole scenario with overrides.
#This allows us to reset the simulation state.

import time
import ALSLib.ALSClient
from ALSLib.ALSHelperDataCollection import SensorThreadManager, SensorOptions
from ALSLib.ALS_TestDefinitionGenerator import generate_tests
from ALSLib.ALSHelperFunctionLibrary import get_config_path

HOST = "127.0.0.1"

def auto_message_handler(x:str):
	pass

class Context:
	client = ALSLib.ALSClient.Client((HOST, 9000), auto_message_handler)

#=============================================================================================

if __name__ == "__main__":
	#Add one or more scenarios that will be loaded in the loop below
	scenarios = ["DemoDatasetOverrideScenario"]
	
	#Add the name and path of the overrides text file
	overridepath = get_config_path("/SituationOverrides/DemoDatasetOverride.txt")

	Context.client.connect(30)

	mgr = SensorThreadManager()
	options = SensorOptions()
	# If clean dir is set to True, helper data collection script will clean the directory.
	options.clear_dir = True
	# Control the amount of samples to collect.
	options.batch_limit = 3
	# Control the amount of samples to burn, before actual collection.
	options.burn_in_samples = 1
	
	#1st for-loop for iterating through the given scenarios
	for i, scenario in enumerate(scenarios):
		print(f"Loading scenario {scenario} ({i+1}/{len(scenarios)})")

		#Generate the overrides from the txt-file
		overrides = generate_tests(overridepath)
		#2nd for-loop for iterating through the given overrides
		for j, override in enumerate(overrides):
			print(f"Running test {j+1}/{len(overrides)}")
			#Override selected for this example changes the vehicle in the image by cycling through vehicle type indices.
			#Load the given scenario using the given override files.
			Context.client.request_load_scenario_with_overrides(scenario, override.removeprefix(scenario+" "))

			#In this example we are using a Situation "DemoDataCollection" with an Ego vehicle that
			#is collecting a real image and a pixel segmentation ground truth image
			# You can add one or more situation layers on top of the scenario as well
			#
			#Context.client.request_load_situation_layer('DemoDatasetOverride_Obstacles')
			options.out_dir = f"Test_{scenario}_{j}"
			# Setup the sensors for each scenario override.
			mgr.auto_setup(Context.client, options)
			# Keep collecting data until manager has finished collecting
			mgr.collect_samples_until_paused()
			#Stop the manager explicitly to setup sensors for next set of samples.
			mgr.stop()
