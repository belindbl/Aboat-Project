import time
import ALSLib.ALSClient
from ALSLib.ALSHelperDataCollection import SensorThreadManager, SensorOptions, ManagerState, load_override_smart, CameraData, LidarData
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
	scenarios = ["Default_Scenario"]
	
	#Add the name and path of the overrides text file
	overridepath = get_config_path("/SituationOverrides/DemoDatasetSmartOverride.txt")

	Context.client.connect(30)

	mgr = SensorThreadManager()
	lidar_options = SensorOptions()
	lidar_options.clear_dir = False
	lidar_options.burn_in_samples = 1
	lidar_options.batch_limit = 3
	#Increase the allocation amount, so that there is enough space for the data.
	lidar_options.alloc_num = 100

	camera_options = SensorOptions()
	camera_options.clear_dir = False
	camera_options.burn_in_samples = 1
	camera_options.batch_limit = 15
	camera_options.save_metadata = True
	#Increase the allocation amount, so that there is enough space for the data.
	camera_options.alloc_num = 100
	
	#1st for-loop for iterating through the given scenarios
	for i, scenario in enumerate(scenarios):
		print(f"Loading scenario {scenario} ({i+1}/{len(scenarios)})")
		Context.client.request_load_scenario(scenario)
		Context.client.request_load_situation_layer("DemoDataCollectionAdvanced", 20)
		camera_options.out_dir = f"CameraData_{scenario}"
		lidar_options.out_dir = f"LidarData_{scenario}"

		#Generate the overrides from the txt-file
		overrides = generate_tests(overridepath)
		overrides = [override.split(" ",1)[1] for override in overrides]
		#2nd for-loop for iterating through the given overrides

		for j, override in enumerate(overrides):
			print(f"Running test {j+1}/{len(overrides)}")
			camera_options.additional_metadata = {"override_id": j}
			lidar_options.out_dir = f"LidarData_{scenario}/override_{j}"
			load_override_smart(override, Context.client)

			#Check the manager state and add the desired sensor groups.
			if mgr.state == ManagerState.STOPPED:
				#Add the sensor groups for the given situation
				mgr.add(HOST, 8882, CameraData, "{sensor_type}{sensor_number}_{frame_id}", camera_options)
				mgr.add(HOST, 8883, LidarData, "{sensor_type}{sensor_number}_{frame_id}", lidar_options)
				mgr.log_enable(True)
				mgr.start()

			#When we are in paused state, load new  options for the camera
			#to update additional metadata and lidar directory
			elif mgr.state == ManagerState.PAUSED:
				mgr.set_options(camera_options, port=8882)
				mgr.set_options(lidar_options, port=8883)
				mgr.unpause()

			#When we are running the manager, poll is paused state and stop if finished commands.
			while mgr.state == ManagerState.RUNNING:
				time.sleep(0.05)
				mgr.is_paused()
				mgr.stop_if_finished()
				
		#Stop the manager explicitly to setup sensors for next set of samples.
		mgr.stop()
