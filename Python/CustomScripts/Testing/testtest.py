import ALSLib.ALSClient
from ALSLib.ALSHelperDataCollection import SensorThreadManager, SensorOptions, CameraData, LidarData, load_override_smart, ManagerState
from ALSLib.ALS_TestDefinitionGenerator import generate_tests   
from ALSLib.ALSHelperFunctionLibrary import get_config_path
import time

HOST = "127.0.0.1"
VL16 = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini"

def auto_message_handler(x:str):
	pass

class Context:
	client = ALSLib.ALSClient.Client((HOST, 9000), auto_message_handler)

#=============================================================================================

if __name__ == "__main__":

	Context.client.connect(30)

	mgr = SensorThreadManager()
	
    # Camera config
	camera_options = SensorOptions()
	camera_options.batch_limit = 3
	# Control the amount of samples to burn, before actual collection.
	camera_options.burn_in_samples = 1
	camera_options.save_metadata=True
	camera_options.alloc_num=100

    #LiDAR config
	lidar_options = SensorOptions()
	# Control the amount of samples to collect.
	lidar_options.batch_limit = 3
	# Control the amount of samples to burn, before actual collection.
	lidar_options.burn_in_samples = 1
	lidar_options.alloc_num=100
	# Output path
	camera_options.out_dir = r"E:\AILiveSim_1_9_7\SensorData\CollTest\img"
	lidar_options.out_dir = r"E:\AILiveSim_1_9_7\SensorData\CollTest\pcl"
	
    # Config file path
	sensorprofile_path = VL16
	sensor_path_in_file = "Sensors.Sensors.[1]"
	cam_path_in_file = "Sensors.Sensors.[0]"
	overrides = "{file_path};{sensor_path}.StreamToNetwork;True<br>".format(file_path=sensorprofile_path, sensor_path=sensor_path_in_file)
	overrides += "{file_path};{sensor_path}.StreamToNetwork;True".format(file_path=sensorprofile_path, sensor_path=cam_path_in_file)
	scenario = "Default_Scenario"
	
	Context.client.request_load_scenario(scenario)
	
	for j, override in enumerate(overrides):
		print(f"Running test {j+1}/{len(overrides)}")
		#Load the given scenario using the given override files.
		load_override_smart(override, Context.client)
		if mgr.state==ManagerState.STOPPED:
			mgr.add(HOST, 8882, CameraData, "{sensor_type}{sensor_number}_{frame_id}", camera_options)
			mgr.add(HOST, 8883, LidarData, "{sensor_type}{sensor_number}_{frame_id}", lidar_options)
			mgr.log_enable(True)
			mgr.start()
		elif mgr.state==ManagerState.PAUSED:
			mgr.set_options(camera_options, port=8880)
			mgr.set_options(lidar_options, port=8881)
			mgr.unpause()
		while mgr.state==ManagerState.RUNNING:
			time.sleep(0.05)
			mgr.is_paused()
			mgr.stop_if_finished()
		
    #Stop the manager explicitly to setup sensors for next set of samples.
	mgr.stop()




    