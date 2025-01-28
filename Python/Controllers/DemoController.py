import json

class ControllerContext:
	def __init__(self, rpm, rpm_acceleration, direction):
		self.current_rpm = rpm
		self.rpm_acceleration = rpm_acceleration
		self.direction = direction

def create_context(rpm, rpm_acceleration, direction):
	return ControllerContext(rpm, rpm_acceleration, direction)

def sign(number):
	if number < 0:
		return -1
	if number > 0:
		return 1
	return 0

def update_velocity(context, deltatime, status, control_values):
	# Print will be shown in the application log
	print(control_values)

	# Move RPM toward target
	rpm_target = control_values["a0"]
	rpm_difference = rpm_target - context.current_rpm
	rpm_change = min(abs(rpm_difference), context.rpm_acceleration) * sign(rpm_difference)

	# Change direction with initialization parameters
	if (context.direction == "forward"):
		rpm_change *= 1
	if (context.direction == "reverse"):
		rpm_change *= -1

	# Store new RPM and convert it to acceleration
	context.current_rpm += rpm_change * deltatime
	acceleration = context.current_rpm / 3000 * (10000 * deltatime)

	boat_status = json.loads(status)

	# Apply drag to velocity and add the acceleration
	current_velocity = boat_status["VelocityWorldCmS"]
	current_forward = boat_status["Forward"]
	drag = 0.1
	new_velocity = [
				(current_velocity[0] * (1 - drag)) + (current_forward[0] * acceleration), 
				(current_velocity[1] * (1 - drag)) + (current_forward[1] * acceleration),
				(current_velocity[2] * (1 - drag)) + (current_forward[2] * acceleration)
				]

	return "Velocity", new_velocity, [False, False, False], "{{acceleration:{0}}}".format(acceleration)