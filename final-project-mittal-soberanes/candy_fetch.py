import cv2
import random
from math import radians

from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Quaternion, PointStamped
import tf
from std_msgs.msg import Empty

import sys
import traceback
import rospy
from geometry_msgs.msg import Twist
import math
from kobuki_msgs.msg import BumperEvent, CliffEvent, WheelDropEvent
import numpy as np

from sensor_msgs.msg import PointCloud2, LaserScan, Image
from cv_bridge import CvBridge, CvBridgeError

from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient
from kobuki_msgs.msg import Sound

from ar_track_alvar_msgs.msg import AlvarMarkers

import time

# Which square drawing method to use: "open_loop" or "waypoint"
SQUARE_METHOD = 'waypoint'

# constant for speed 
LIN_SPEED = 0.2  # 0.2 m/s
ROT_SPEED = math.radians(45)  # 45 deg/s in radians/s
ROT_K = 3  # Constant for proportional angular velocity control
LIN_K = 0.5  # Constant for proportional linear velocity control

# Parameters for drawing shape
SIDE_LENGTH = 0.7  # meter
TURN_ANGLE = math.radians(90)  # 90 deg in radians

# Center of screen image point
CENTER = 355

# Empty map array (40x50)
# mark each grid as unknown to begin with
# MAP = [[0 for i in range(50)] for j in range(40)]

# These are the only tag IDs that are being considered here
VALID_IDS = range(18)

def dist(pos1, pos2):
    """
    Get cartesian distance between the (x, y) positions
    :param pos1: (x, y) position 1
    :param pos2: (x, y) position 2
    :return: Distance (float)
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def orient(curr_pos, goal_pos):
    """
    Get necessary heading to reach the goal
    :param curr_pos: (x, y) current position of the robot
    :param goal_pos: (x, y) goal position to orient toward
    """
    return math.atan2(
        goal_pos[1] - curr_pos[1],
        goal_pos[0] - curr_pos[0])


def angle_compare(curr_angle, goal_angle):
    """
    Determine the difference between the angles, normalized from -pi to +pi
    :param curr_angle: current angle of the robot, in radians
    :param goal_angle: goal orientation for the robot, in radians
    """
    pi2 = 2 * math.pi
    # Normalize angle difference
    angle_diff = (curr_angle - goal_angle) % pi2
    # Force into range 0-2*pi
    angle_diff = (angle_diff + pi2) % pi2
    # Adjust to range of -pi to pi
    if (angle_diff > math.pi):
        angle_diff -= pi2
    return angle_diff


def sign(val):
    """
    Get the sign of direction to turn if >/< pi
    :param val: Number from 0 to 2*pi
    :return: 1 if val < pi (left turn), else -1 (right turn)
    """
    if val < 0:
        return -1
    elif val > 0:
        return 1
    return 0


class FetchCandy:
	"""
    Allows turtlebot to receive a designated station from which to pick up candy
    and return home, awaiting further instruction

	"""

	def __init__(self):
		# Initialize
		rospy.init_node('FetchCandy', anonymous=False)

		self.first = True
		self.id = None

		# Boolean to track whether bumper was pressed
		self.bump = False
		self.bump_dir = 0

		# initialize camera direction
		self.cam_dir = 3
		self.prev_dir = 1

		# How long to go forward/turn in open loop control
		self.turn_dur = rospy.Duration(TURN_ANGLE/ROT_SPEED)  # seconds
		self.straight_dur = rospy.Duration(SIDE_LENGTH/LIN_SPEED)  # seconds

		# Localization determined from EKF
		# Position is a tuple of the (x, y) position
		self.position = None

		# Variables for midpoints and home
		self.home = None
		self.home_id = None
		self.midpoint = (-3.5, -1)
		self.midpoint_home = (-0.6, -0.6)

		# Boolean for candy collection
		self.collected_candy = False

		# Orientation is the rotation in the floor plane from initial pose
		self.orientation = None

		# Offset to initialize coordinates to (0,0)
		self.x_off = 0
		self.y_off = 0

		# initialize counters
		self.counter_direction = 0

		# Variables to determine robot conditions
		self.state = 'order'
		self.blocked = False
		self.see_robot = False
		self.prev_state = 'order'
		self.destination_reached = False
		self.started_going_home = False
		self.marker_reached = False

		# Keep track of all the markers seen
		# This is a dictionary of tag_id: distance pairs
		# If the marker is lost, the value is set to None, but a marker is never removed
		self.markers = {}

		# create blank image for map (400x500 pixels)
		#self.map = np.zeros((40*10, 50*10, 3), np.uint8)

		# What to do you ctrl + c (call shutdown function written below)
		rospy.on_shutdown(self.shutdown)

		# Publish to topic for controlling robot
		#self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

		# Create a publisher which can "talk" to TurtleBot and tell it to move
		self.cmd_vel = rospy.Publisher('wanderer_velocity_smoother/raw_cmd_vel',Twist, queue_size=10)

		# Subscribe to bumper events topic
		rospy.Subscriber('mobile_base/events/bumper', BumperEvent, self.process_bump_sensing)

		# Subscribe to topic for AR tags
		rospy.Subscriber('/ar_pose_marker', AlvarMarkers, self.process_ar_tags)

		# Subscribe to robot_pose_ekf for odometry/position information
		rospy.Subscriber('/robot_pose_ekf/odom_combined', PoseWithCovarianceStamped, self.process_ekf)

		# Set up the odometry reset publisher (publishing Empty messages here will reset odom)
		reset_odom = rospy.Publisher('/mobile_base/commands/reset_odometry', Empty, queue_size=1)

		# Reset odometry (these messages take about a second to get through)
		timer = rospy.Time.now()
		while rospy.Time.now() - timer < rospy.Duration(1) or self.position is None:
			reset_odom.publish(Empty())

		# 5 HZ
		self.rate = rospy.Rate(5)

		self.bridge = CvBridge()

		# Subscribe to camera depth topic
		rospy.Subscriber('/camera/depth/image', Image, self.process_depth_image, queue_size=1, buff_size=2 ** 24)


	def run(self):
		"""
		Run the robot until Ctrl+C is pressed
		It waits to receive a station the moves to it and comes back
		:return: None
		"""

		while not rospy.is_shutdown():

			# if obstacle is detected
			if self.blocked:
				if self.counter_direction >= 10:
					# STOP
					move_cmd.linear.x = 0

					# turn 90 degrees 
					for i in range(8):
						move_cmd.angular.z = radians(self.prev_dir * 60)
						self.cmd_vel.publish(move_cmd)
						self.rate.sleep()

					# reset counter
					self.counter_direction = 0

				# Sensed obstacle LEFT
				elif self.cam_dir == 0:
					# STOP
					move_cmd.linear.x = 0

					# turn right
					while self.cam_dir == 0:
						move_cmd.angular.z = radians(-30)
						self.cmd_vel.publish(move_cmd)
						self.rate.sleep()

					# set previous turn direction
					self.prev_dir = -1
					self.counter_direction += 1

				# Sensed obstacle in MIDDLE
				elif self.cam_dir == 1:
					# STOP
					move_cmd.linear.x = 0

					# turn based on previous turn
					while self.cam_dir == 1:
						move_cmd.angular.z = radians(self.prev_dir * 30)
						self.cmd_vel.publish(move_cmd)
						self.rate.sleep()

				# Sensed obstacle RIGHT
				else:
					# STOP
					move_cmd.linear.x = 0

					# turn left
					while self.cam_dir == 2:
						move_cmd.angular.z = radians(45)
						self.cmd_vel.publish(move_cmd)
						self.rate.sleep()

					# set previous turn direction
					self.prev_dir = 1
					self.counter_direction += 1

			# if bumpers are triggered
			elif self.bump:
				# STOP
				move_cmd.linear.x = 0
				move_cmd.angular.z = 0

				# back up
				for i in range(5):
					move_cmd.linear.x = -LIN_SPEED
					self.cmd_vel.publish(move_cmd)
				self.rate.sleep()

				# STOP
				move_cmd.linear.x = 0

				# select random angle to turn based on bumper pressed
				if self.bump_dir == BumperEvent.LEFT:
					turn_angle = random.randint(-45,0)
				elif self.bump_dir == BumperEvent.RIGHT:
					turn_angle = random.randint(0,45)
				else:
					turn_angle = random.randint(-45,45)

				# turn
				for i in range(12):
					move_cmd.angular.z = radians(turn_angle)
					self.cmd_vel.publish(move_cmd)
					self.rate.sleep()

				self.rate.sleep()
			# if at HOME, either awaiting an order, or returning home after collecting candy
			elif self.state == 'order' or self.state == 'midpoint orient' or self.state == 'midpoint approach':
				if self.id is None:
					self.id = int(raw_input("Please enter a station number: "))
				# if candy collected, return home
				else:
					if self.collected_candy:
						self.destination = self.midpoint_home
					else:
						self.destination = self.midpoint
					if self.state != 'midpoint approach':
						self.state = 'midpoint orient'
						print 'MIDPOINT ORIENT'
						self.prev_state = self.state
					self.dest_orientation = orient(self.position, self.destination)
					move_cmd = self.waypoint_navigation()
					self.cmd_vel.publish(move_cmd)
					self.rate.sleep()

			# state for returning home
			elif self.state == 'homing orient' or self.state == 'homing approach':
				# Move home (destination should already be set above)
				self.destination = self.home
				self.dest_orientation = orient(self.position, self.home)
				move_cmd = self.waypoint_navigation()
				self.cmd_vel.publish(move_cmd)
				self.rate.sleep()

			# state for if at midpoint
			elif self.state == 'midpoint':
				while self.state == 'midpoint':
					move_cmd = Twist()
					move_cmd.linear.x = 0
					move_cmd.angular.z = radians(15)
					self.cmd_vel.publish(move_cmd)
					self.rate.sleep()

			elif self.state == 'stop':
				self.rate.sleep()

			# else, orient and approach
			else:
				if self.state != 'approach' and self.state != 'approach final' and self.state != 'approach checking' and self.state != 'homing approach':
					self.state = 'orient'
					print 'ORIENT'
					self.prev_state = self.state

				# after seeing a marker for the first time, calculate its global coordinates
				if self.first:
					x_ar = self.markers[self.id][0][0]
					z_ar = self.markers[self.id][0][2]
					d = self.markers[self.id][1]

					print 'x_ar: '
					print x_ar
					print 'z_ar: '
					print z_ar
					alpha = math.asin(x_ar / d)
					print 'alpha: '
					print alpha
					if self.collected_candy:
						beta = self.orientation
					else:
						beta = sign(self.orientation) * angle_compare(abs(self.orientation), 0)
					print 'beta: '
					print beta


					# if sign(beta) == sign(x_ar):
					# 	x_global = self.position[0] + (d * math.cos(alpha + beta)) #- 0.8
					# 	y_global = self.position[1] + (d * math.sin(alpha + beta)) 

					# else:
					# 	x_global = self.position[0] + (d * math.cos(beta - alpha)) #- 0.8
					# 	y_global = self.position[1] + (d * math.sin(beta - alpha)) 

					# if abs(beta) < math.pi/2:
					# 	#if beta > 0:
					# 	x_global = self.position[0] + (d * math.cos(beta - alpha)) #- 0.8
					# 	y_global = self.position[1] - (d * math.sin(beta - alpha)) - 0.5

					# 	# else:
					# 	# 	x_global = self.position[0] - (d * math.cos(math.pi - abs(beta - alpha))) #- 0.8
					# 	# 	y_global = self.position[1] + (d * math.sin(math.pi - abs(beta - alpha))) - abs(beta/3.5)
					# else:

					if self.collected_candy:
						if beta > 0:
							x_global = self.position[0] + (d * math.sin(beta - alpha)) + 3
							y_global = self.position[1] - (d * math.cos(beta - alpha)) - 3

						else:
							x_global = self.position[0] + (d * math.sin(beta - alpha)) + 3
							y_global = self.position[1] - (d * math.cos(beta - alpha)) - 3
					else:
						if beta > 0:
							x_global = self.position[0] + (d * math.cos(abs(beta - alpha)))
							y_global = self.position[1] + (d * math.sin(abs(beta - alpha)))

						else:
							x_global = self.position[0] + (d * math.cos(abs(beta - alpha)))
							y_global = self.position[1] - (d * math.sin(abs(beta - alpha)))

					print 'x global: '
					print x_global
					print 'y global: '
					print y_global
					if self.collected_candy:
						x_global += 3
						y_global -= 3
					elif self.id == 7:
						y_global += 0.29
						x_global -= 0.08
					elif self.id == 3:
						y_global -= 0.244
					elif self.id == 4:
						y_global -= 0.14
					elif self.id == 6:
						x_global -= 0.365
						y_global += 0.23

					#if forwards is actually positive, change x_global to + not -
					#if left is positive, change bottom lines' signs

					# if self.marker_reached:
					# 	# If waypoint reached, go to station
					# 	y_global = y_global + 1

					self.first = False

					if not self.collected_candy:
						self.destination = (x_global, y_global)
					else:
						self.destination = self.home

				#print self.position
				#print self.destination
				self.dest_orientation = orient(self.position, self.destination)

				move_cmd = self.waypoint_navigation()
				self.cmd_vel.publish(move_cmd)
				self.rate.sleep()


	def dist(self, pos):
		"""
		Get a the distance to a position from the origin. This can be any number of dimensions
		:param pos: Tuple of position
		:return: Float distance of position from origin
		"""
		return np.sqrt(sum([i**2 for i in pos]))

	def waypoint_navigation(self):
		"""
		Navigate to each of the waypoints in self.waypoints (in order)
		:return: Twist object of movement to make
		"""
		# Movement command to send to the robot
		move_cmd = Twist()

		#print self.id

		# Get angle difference of robot from destination (-pi to pi)
		#print self.dest_orientation
		#print self.orientation

		angle_diff = angle_compare(self.dest_orientation, self.orientation)
		#print angle_diff

		# Determine turn angle (proportional control with upper bound)
		# This also selects the direction that minimizes required turning
		prop_angle = abs(angle_diff) * ROT_K
		turn_angle = sign(angle_diff) * min(prop_angle, ROT_SPEED)

		destination_dist = dist(self.position, self.destination)

		if self.state == 'move in':
			move_cmd.angular.z = 0
			move_cmd.linear.x = LIN_SPEED
		if self.state == 'orient' or self.state == 'midpoint orient' or self.state == 'homing orient':
			#print 'ORIENT'
			# Orient to the destination (turning only) until with 5 degrees of goal
			move_cmd.angular.z = turn_angle
			#print abs(angle_diff)
			#print math.radians(5)
			if abs(angle_diff) < math.radians(5):
				if self.state == 'midpoint orient':
					print 'ORIENTED MIDPOINT'
					self.state = 'midpoint approach'
					self.prev_state = self.state
					print 'APPROACHING MIDPOINT'
				elif self.state == 'homing orient':
					print 'ORIENTED HOME'
					self.state = 'homing approach'
					self.prev_state = self.state
				else:
					self.state = 'approach'
					self.prev_state = self.state
					print 'APPROACHING STATION'
		elif self.state == 'approach' or self.state == 'midpoint approach' or self.state == 'approach final' or self.state == 'homing approach':
			# Move toward the destination (proportional control for linear and angular)
			# Robot can only move FORWARD (for safety), since distance to destination
			# is always positive, though the robot should turn around to return to the
			# goal if it overshoots
			move_cmd.angular.z = turn_angle
			move_cmd.linear.x = min(LIN_SPEED, destination_dist * LIN_K)
			if destination_dist < 0.4 and self.state == 'midpoint approach':
				print 'REACHED MIDPOINT'
				self.state = 'midpoint'
				self.prev_state = self.state
			elif (destination_dist < 1 and (self.state == 'approach' or self.state == 'homing approach')) or self.state == 'approach checking':
				print 'CHECKING FOR ROBOT'
				self.state = 'approach checking'
				print self.see_robot
				#time.sleep(3)
				if self.see_robot:
					print 'WAITING FOR ROBOT'
				else:
					print 'ARRIVING'
					self.state = 'approach final'
			elif destination_dist < 0.2 and self.state == 'approach final':
				print 'REACHED DESTINATION'
				if self.collected_candy:
					self.collected_candy = False
					self.id = None
					self.first = True
					self.state = 'order'
				else:
					self.collected_candy = True
					self.first = True
				#self.state = 'stop'
				#self.prev_state = self.state
					time.sleep(10)
					print 'GOING HOME'
					self.state = 'order'
			# elif destination_dist < 0.05:
			# 	# Consider destination reached if within 5 cm
			# 	self.marker_reached = True
			# 	# else will return empty/stop move_cmd

		return move_cmd

	def process_ekf(self, data):
		"""
		Process a message from the robot_pose_ekf and save position & orientation to the parameters
		:param data: PoseWithCovarianceStamped from EKF
		"""
		# Extract the relevant covariances (uncertainties).
		# Note that these are uncertainty on the robot VELOCITY, not position
		cov = np.reshape(np.array(data.pose.covariance), (6, 6))
		x_var = cov[0, 0]
		y_var = cov[1, 1]
		rot_var = cov[5, 5]
		# print '({}, {})  {}'.format(x_var, y_var, rot_var)
		pos = data.pose.pose.position

		# Set robot's position coordinates 
		if self.position is None and self.home is None:
			self.x_off = pos.x
			print self.x_off
			self.y_off = pos.y
			print self.y_off
			self.home = (0, 0)
			pos.x = pos.x - self.x_off
			pos.y = pos.y - self.y_off
			print self.home
		
		self.position = (pos.x, pos.y)
		#print self.position

		# print '({}, {})  {}'.format(x_var, y_var, rot_var)
		orientation = data.pose.pose.orientation
		list_orientation = [orientation.x, orientation.y, orientation.z, orientation.w]
		self.orientation = tf.transformations.euler_from_quaternion(list_orientation)[-1]


	def process_ar_tags(self, data):
		"""
		Process the AR tag information.
		:param data: AlvarMarkers message telling you where multiple individual AR tags are
		:return: None
		"""
		# Set the position for all the markers that are in the received message
		# for marker in data.markers:
		#print data.markers
		if len(data.markers) != 0:
			marker = data.markers[0]

			if marker.id in VALID_IDS and len(self.markers) == 0:
				#if self.first:
					#self.id = marker.id
					#self.first = False

				self.home_id = marker.id

				if self.home_id == 2:
					self.midpoint = (-3.5, -0.25)
					self.midpoint_home = (-2, -0.5)
				elif self.home_id == 0:
					self.midpoint = (-1, -2.5)
					self.midpoint_home = (-1, -1.5)

				pos = marker.pose.pose.position
				#print pos

				distance = self.dist((pos.x, pos.y, pos.z))

				coords = (pos.x, pos.y, pos.z)

				package = (coords, distance)
				#print package

				self.markers[marker.id] = package
				#print self.markers[self.id]

			elif (marker.id == self.id and self.state == 'midpoint' and not self.collected_candy) or (marker.id == self.home_id and self.state == 'midpoint' and self.collected_candy):
				if marker.id == self.home_id:
					self.state = 'move in'

				pos = marker.pose.pose.position
				#print pos

				distance = self.dist((pos.x, pos.y, pos.z))

				coords = (pos.x, pos.y, pos.z)

				package = (coords, distance)
				#print package

				self.markers[marker.id] = package
				print self.markers[self.id]

				self.state = 'station seen'
				self.prev_state = self.state

	def process_depth_image(self, data):
		try:
			# Use bridge to convert to CV::Mat type. (i.e., convert image from ROS format to OpenCV format)
			# NOTE: SOME DATA WILL BE 'NaN'
			# and numbers correspond to distance to camera in meters
			# This imports as the default data encoding. For the ASUS Xtion cameras,
			# this is '32FC1' (single precision floating point [32F], single channel [C1])
			cv_image = self.bridge.imgmsg_to_cv2(data)

			# Apply a threshold to your depth image with inRange()
			distance_image = cv2.inRange(cv_image, 0.4, 0.7)
			#map_image = cv2.inRange(cv_image, 0.6, 1.5)
			distance_image = distance_image[0:300, :]

			# gets index of obstacle in image array
			y,x = np.where(distance_image > 0)

			# no obstacle present
			if x.size == 0:
				self.blocked = False
				self.see_robot = False
				self.cam_dir = 3

			# records direction of obstacle (LEFT, RIGHT, MIDDLE)
			elif self.state != 'order' and self.state != 'stop' and self.state != 'midpoint orient' and self.state != 'approach final' and self.state != 'homing orient':
				if self.state == 'approach checking':
					self.see_robot = True
				elif (self.state == 'homing approach' or self.state == 'moving in') and distance_image[y[0]][x[0]] < 2:
					self.state = 'approach checking'
				elif self.state != 'approach checking':
					self.blocked = True 
				if x[0] < 250:
					self.cam_dir = 0
				elif x[0] > 450: 
					self.cam_dir = 2
				else:
					self.cam_dir = 1

			cv2.waitKey(5)
		except CvBridgeError, err:
			rospy.loginfo(err)


	def process_bump_sensing(self, data):
		"""
		If bump data is received, process the data
		data.bumper: LEFT (0), CENTER (1), RIGHT (2)
		data.state: RELEASED (0), PRESSED (1)
		:param data: Raw message data from bump sensor 
		:return: None
		"""

		# records which bumper was pressed 
		if data.state == BumperEvent.PRESSED:
			self.bump = True
			self.bump_dir = data.bumper
		elif data.state == BumperEvent.RELEASED:
			self.bump = False
		rospy.loginfo("Bumper Event")
		rospy.loginfo(data.bumper)

	def shutdown(self):
		"""
		Pre-shutdown routine. Stops the robot before rospy.shutdown 
		:return: None
		"""
		# stop turtlebot

		rospy.loginfo("Stop fetching!")
		self.state = 'stop'
		# publish a zeroed out Twist object
		self.cmd_vel.publish(Twist())
		# sleep before final shutdown
		rospy.sleep(1)


if __name__ == '__main__':
	try:
		robot = FetchCandy()
		robot.run()
	except Exception, err:
		rospy.loginfo("FetchCandy node terminated.")
		ex_type, ex, tb = sys.exc_info()
		traceback.print_tb(tb)
		print err