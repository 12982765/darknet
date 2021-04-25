#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
import cv2
import math
import numpy as np


homography = np.array([[-4.89775e-05, -0.0002150858, -0.1818273],
                       [0.00099274, 1.202336e-06, -0.3280241],
                       [-0.0004281805, -0.007185673, 1]])

# The inverse of the homography matrix can be used to proints on the ground plane back into the image
homography_inverse = np.linalg.inv(homography)

# Images from the Duckiebot are 640 x 480
image_width = 640
image_height = 480

u_min = 0
u_max = image_width
v_min = image_height / 2
v_max = image_height

vel_pub=rospy.Publisher('/cmd_vel', Twist, queue_size=1)
curr_orientation_angle=0.0

def start_node():
    rospy.init_node('lane_controller')
    rospy.loginfo('image_subcriber node started')
    
    rospy.Subscriber("/duckiebot/camera_node/image/compressed", CompressedImage, process_image)
    
    rospy.spin()
    
def process_image(msg):
    try:
       #convert sensor_msgs/CompressedImage to OpenCV Image
       
       # orig = bridge.imgmsg_to_cv2(msg, "bgr8")
        np_arr = np.fromstring(msg.data, np.uint8)
        orig = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        edge_filtered = detect_edges(orig)
        roi_filtered = region_of_interest(edge_filtered)
       #line detection 
        line_segments = detect_line_segments(roi_filtered)
        
        lane_lines = average_slope_intercept(orig, line_segments)
        #lane_lines = combine_lines (orig, line_segments)
       
       #draw detected lines on the origial image 
        lane_lines_image = display_lines(orig,lane_lines)
            
    except Exception as err:
        print err
        
        
    global curr_orientation_angle    
    robot_orientation_new=compute_heading_angle(orig, lane_lines)
    image_heading = display_heading_line( lane_lines_image,robot_orientation_new)
    curr_orientation_angle= curr_orientation_angle*0.0+ robot_orientation_new*1.0
    
    lane_controller( curr_orientation_angle,len(lane_lines))    
    # show results
    show_image(image_heading )
    
def lane_controller(orientation,num_lane):
    
    vel_reducer=(1-abs(orientation)/30.0)      
    if  vel_reducer < 0.0:
        vel_mul=0.0        
    else:
        vel_mul=vel_reducer 
        
    
        
    # proprotional controller values 
    kp_two_lines = -0.01
    kp_single_line= 0.001  
    
    if num_lane==2:
        forward_vel=0.02*vel_mul
        angular_vel= kp_two_lines*float(orientation)
        
    elif num_lane==1:
        forward_vel=0.005 * vel_mul
        angular_vel=kp_single_line*float(orientation)
    else  :
        forward_vel=0.00
        angular_vel=0.00
    
    publishing_vel( forward_vel, angular_vel)
    
    
def publishing_vel( forward_vel, angular_vel):
    vel = Twist()
    vel.angular.x = 0.0
    vel.angular.y = 0.0
    vel.angular.z = angular_vel
    vel.linear.x = forward_vel
    vel.linear.y = 0.0
    vel.linear.z = 0.0
    vel_pub.publish(vel)     
    

    
def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 140, 40])
    upper_blue = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    

    # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges
    

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges
    
    
def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 50  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=5, maxLineGap=10)                                    
    return line_segments
    
    
def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        rospy.loginfo('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                #rospy.loginfo('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            
            if slope < 0:               
               left_fit.append((slope, intercept))
            else:
               right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
    

    return lane_lines
    
    
    
def combine_lines (frame, line_segments):  
    color_1=[255, 0, 0] 
    color_2=[0, 0, 255] 
    thickness=2
    lane_lines = []
    if line_segments is None:
        rospy.loginfo('No line_segment segments detected')
        return lane_lines
    
    # state variables to keep track of most dominant segment
    largestLeftLineSize = 0
    largestRightLineSize = 0
    largestLeftLine = (0,0,0,0)
    largestRightLine = (0,0,0,0)
    left_lane_flag= False
    right_lane_flag= False
    
    for line in line_segments:
        for x1,y1,x2,y2 in line:
            size = math.hypot(x2 - x1, y2 - y1)
            slope = ((y2-y1)/(x2-x1))
            # Filter slope based on incline and
            # find the most dominent segment based on length
            if (slope > 0.0): #right
                if (size > largestRightLineSize):
                    largestRightLine = (x1, y1, x2, y2)
                    right_lane_flag= True                  
               #cv2.line(frame, (x1, y1), (x2, y2), color_1, thickness)
            elif (slope < 0.0): #left
                if (size > largestLeftLineSize):
                    largestLeftLine = (x1, y1, x2, y2)
                    left_lane_flag= True
               #cv2.line(frame, (x1, y1), (x2, y2), color_2, thickness)
                
    if  left_lane_flag: 
        x1,y1,x2,y2 = largestLeftLine         
        fit_left = np.polyfit((x1,y1), (x2,y2), 1) 
        cv2.line(frame, (x1, y1), (x2, y2), color_1, thickness)
        lane_lines.append(make_points(frame, fit_left))   
    
    if  right_lane_flag:
        x1,y1,x2,y2 = largestRightLine    
        fit_right = np.polyfit((x1,y1), (x2,y2), 1) 
        cv2.line(frame, (x1, y1), (x2, y2), color_2, thickness)  
        lane_lines.append(make_points(frame, fit_right))
    
    return lane_lines
    
def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]
    
    
def compute_heading_angle(frame, lane_lines):
    height, width, _ = frame.shape

    if len(lane_lines) == 2: # if two lane lines are detected
        _, _, left_x2, _ = lane_lines[0][0] # extract left x2 from lane_lines array
        _, _, right_x2, _ = lane_lines[1][0] # extract right x2 from lane_lines array
        mid = float(width / 2)
        x_offset = float( (left_x2 + right_x2) / 2 - mid)
        y_offset = float(height / 2)  
 
    elif len(lane_lines) == 1: # if only one line is detected
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = float(x2 - x1)
        y_offset = float(height / 2)

    elif len(lane_lines) == 0: # if no line is detected
        x_offset = 0.0
        y_offset = float(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = float(angle_to_mid_radian * 180.0 / math.pi)  
    heading_angle = angle_to_mid_deg 
    rospy.loginfo("Heading angle %f :",angle_to_mid_deg)

    
    return heading_angle  
    
   
    
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image
    
    
def display_heading_line(frame, heading_angle, line_color=(0, 0, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    
    heading_angle_radian = (heading_angle + 90.0) / 180.0 * math.pi 
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(heading_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image

        
        
def show_image(img):
    cv2.imshow('ROI Filter', img)
    cv2.waitKey(1)

def line_points2slope_intercept(line_points):
    """Converts a line represented by two points to a line represented by slope and intercept

    :param line_points: An array of four values: [u1, v1, u2, v2]
    :return: An array of two values: [slope, intercept]
    """

    u1, v1, u2, v2 = line_points

    fit = np.polyfit((u1, u2), (v1, v2), 1)

    slope = fit[0]
    intercept = fit[1]

    return np.array([slope, intercept])


def line_slope_intercept2points(line_slope_intercept):
    """Converts a line represented by slope and intercept to a line represented by two points within the bottom half of
    the image

    :param line_slope_intercept: An array of two values: [slope, intercept]
    :return: An array of four values: [u1, v1, u2, v2]
    """

    slope, intercept = line_slope_intercept

    # Find where line intercepts bottom of image
    u1 = (v_max - intercept) / slope
    v1 = v_max

    if u1 < u_min:  # Line extends past left of image
        # Find where line intercepts left of image
        u1 = u_min
        v1 = slope * u_min + intercept
    elif u1 > u_max:  # Line extends past right of image
        # Find where line intercepts with right of image
        u1 = u_max
        v1 = slope * u_max + intercept

    # Find where line intercepts with middle of image
    u2 = (v_min - intercept) / slope
    v2 = v_min

    if u2 < u_min:  # Line extends past left of image
        # Find where line intercepts left of image
        u2 = u_min
        v2 = slope * u_min + intercept
    elif u2 > u_max:  # Line extends past right of image
        # Find where line intercepts with right of image
        u2 = u_max
        v2 = slope * u_max + intercept

    return np.array([u1, v1, u2, v2])


def point_image2ground(image_point):
    """Converts a point in the image to a point on the ground plane

    :param image_point: An array of two values: [u, v]
    :return: An array of two values: [x, y]
    """

    image_point = np.append(image_point, 1.)

    ground_point = np.dot(homography, image_point)

    ground_point /= ground_point[2]

    return np.array([ground_point[0], ground_point[1]])


def point_ground2image(ground_point):
    """Converts a point on the ground plane to a point in the image

    :param ground_point: An array of two values: [x, y]
    :return: An array of two values: [u, v]
    """

    ground_point = np.append(ground_point, 1.)

    image_point = np.dot(homography_inverse, ground_point)

    image_point /= image_point[2]

    return np.array([image_point[0], image_point[1]])


def line_image2ground(image_line):
    """Converts a line in the image to a line on the ground plane

    :param image_line: An array of four values: [u1, v1, u2, v2]
    :return: An array of four values: [x1, y1, x2, y2]
    """

    image_point1 = np.array([image_line[0], image_line[1]])
    image_point2 = np.array([image_line[2], image_line[3]])

    ground_point1 = point_image2ground(image_point1)
    ground_point2 = point_image2ground(image_point2)

    return np.array([ground_point1[0], ground_point1[1], ground_point2[0], ground_point2[1]])


def line_ground2image(ground_line):
    """Converts a line on the ground plane to a line in the image

    :param ground_line: An array of four values: [x1, y1, x2, y2]
    :return: An array of four values: [u1, v1, u2, v2]
    """

    ground_point1 = np.array([ground_line[0], ground_line[1]])
    ground_point2 = np.array([ground_line[2], ground_line[3]])

    image_point1 = point_ground2image(ground_point1)
    image_point2 = point_ground2image(ground_point2)

    line_image_points = np.array([image_point1[0], image_point1[1], image_point2[0], image_point2[1]])

    # Convert to slope, intercept and back to force points to be within image
    line_slope_intercept = line_points2slope_intercept(line_image_points)
    line_image_points_from_slope_intercept = line_slope_intercept2points(line_slope_intercept)

    if line_image_points[0] < u_min or line_image_points[0] > u_max or \
            line_image_points[1] < v_min or line_image_points[1] > v_max:
        # Replace first point from first point converted from slope and intercept
        line_image_points[0] = line_image_points_from_slope_intercept[0]
        line_image_points[1] = line_image_points_from_slope_intercept[1]

    if line_image_points[2] < u_min or line_image_points[2] > u_max or \
            line_image_points[3] < v_min or line_image_points[3] > v_max:
        # Replace first point from first point converted from slope and intercept
        line_image_points[2] = line_image_points_from_slope_intercept[2]
        line_image_points[3] = line_image_points_from_slope_intercept[3]

    return line_image_points


def plot_image_lines(image_lines):
    """Plot image coordinate lines onto image

    :param image_lines: An array of arrays of four values: [[u1, v1, u2, v2], [u1, v1, u2, v2]]
    :return:
    """
    # Initialise blank image
    image = np.zeros([image_height, image_width], np.uint8)

    # Draw lines on image
    for image_line in image_lines:
        u1, v1, u2, v2 = image_line
        cv2.line(image, (int(u1), int(v1)), (int(u2), int(v2)), 255, 3, cv2.LINE_AA)

    # Show image
    cv2.imshow('Image', image)
    cv2.waitKey(0)


def plot_ground_lines(ground_lines):
    """Plot ground coordinate lines onto image

    :param ground_lines: An array of arrays of four values: [[x1, y1, x2, y2], [x1, y1, x2, y2]]
    :return:
    """
    # Convert to image lines
    image_lines = []

    for ground_line in ground_lines:
        image_lines.append(line_ground2image(ground_line))

    # Plot
    plot_image_lines(image_lines)


class Particle:
    def __init__(self, y, phi, weight):
        self.y = y
        self.phi = phi
        self.weight = weight

    def move(self, d, theta):
        """Moves the particle

        :param d: distance (metres)
        :param theta: rotation (radians)
        :return:
        """

        self.y += d * np.sin(self.phi)
        self.phi += theta


class ParticleFilter:
    def __init__(self, num_particles, y_min, y_max, phi_min, phi_max):
        """Create particle filter

        :param num_particles: number of particles to create
        :param y_min: minimum y position of particles (metres)
        :param y_max: maximum y position of particles (metres)
        :param phi_min: minimum orientation of particles (radians)
        :param phi_max: maximum orientation of particles (radians)
        """
        self.num_particles = num_particles
        self.y_min = y_min
        self.y_max = y_max
        self.phi_min = phi_min
        self.phi_max = phi_max

        self.particles = []

        # Initialise particles
        self.initialise_particles()

    def initialise_particles(self):
        """Initialise particles with uniform distribution"""
        self.particles = []

        for i in range(0, self.num_particles):
            particle = Particle(y=np.random.uniform(self.y_min, self.y_max),
                                phi=np.random.uniform(self.phi_min, self.phi_max),
                                weight=1. / self.num_particles)

            self.particles.append(particle)

    def step(self, v, omega, t, lane_lines):
        """A complete step of the particle filter

        :param v: velocity (metres/second)
        :param omega: angular velocity (radians/second)
        :param t: time (seconds)
        :param lane_lines: lane_lines: an array of up to two lines on the ground plane, e.g. [[x1, y1, x2, y2], [x1, y1, x2, y2]]
        :return: pose estimate (y position and orientation): [y, phi]
        """

        self.motion_prediction(v, omega, t)
        self.observation_update(lane_lines)
        self.normalise_particles()
        return self.estimate_pose()

    def motion_prediction(self, v, omega, t):
        """Move particles

        :param v: velocity (metres/second)
        :param omega: angular velocity (radians/second)
        :param t: time (seconds)
        :return:
        """

        d = v * t  # distance, metres
        theta = omega * t  # rotation, radians

        for particle in self.particles:
            # Add noise to "d" and "theta"
            d_noisy = np.random.normal(d, 0.01)  # The second argument is standard deviation in metres
            theta_noisy = np.random.normal(theta, np.pi / 16)  # The second argument is standard deviation in radians

            particle.move(d_noisy, theta_noisy)

    def observation_update(self, lane_lines):
        """Update particle weights using observation

        :param lane_lines: an array of up to two lines on the ground plane, e.g. [[x1, y1, x2, y2], [x1, y1, x2, y2]]
        :return:
        """

        # Determine Duckiebot pose from observation
        left_slope, left_intercept = line_points2slope_intercept(lane_lines[0])
        right_slope, right_intercept = line_points2slope_intercept(lane_lines[1])
        
        y_expected = -(left_intercept + right_intercept) / 2.
        left_angle = np.arctan(left_slope)
        right_angle = np.arctan(right_slope)
        phi_expected = -((left_angle + right_angle) / 2.)
        
        # TODO: What if there's only one line? What's the expected pose from the observation?
        # You should know if the line is the left or right lane, and you know the width of the lanes, so you can
        # determine the deviation from the centre
        
        
        
        # Compare the position of each particle with the expected position from the observation to adjust weight
        for particle in self.particles:
            y_diff = np.abs(particle.y - y_expected)
            phi_diff = np.abs(particle.phi - phi_expected)

            # This isn't a good way of determining likelihood, but it seems to be work
            y_likelihood = max(1. - y_diff / 0.5, 0.1)
            phi_likelihood = max(1. - phi_diff / (np.pi / 4.), 0.1)

            particle.weight *= y_likelihood * phi_likelihood

    def normalise_particles(self):
        """Normalise particles"""
        weight_sum = 0.

        for particle in self.particles:
            weight_sum += particle.weight

        for particle in self.particles:
            particle.weight /= weight_sum

    def estimate_pose(self):
        """Estimate pose of the Duckiebot

        :return: pose estimate (y position and orientation): [y, phi]
        """

        y = 0.
        phi = 0.

        # Weighted average
        for particle in self.particles:
            y += particle.y * particle.weight
            phi += particle.phi * particle.weight

        return np.array([y, phi])

    def resample_particles(self):
        """Weighted resampling of particles"""

        old_particles = self.particles

        self.particles = []

        for i in range(0, self.num_particles):
            target = np.random.uniform(0., 1.)
            current = 0.
            p = 0
            while True:
                if target > current and target < current + old_particles[p].weight:
                    # numpy.random.normal is used to add noise
                    particle = Particle(y=np.random.normal(old_particles[p].y, 0.05),
                                        phi=np.random.normal(old_particles[p].phi, np.pi / 8.),
                                        weight=1. / self.num_particles)

                    self.particles.append(particle)

                    break

                current += old_particles[p].weight
                p += 1

                if p > len(old_particles):
                    p = 0

if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
    # Two observations of two lane lines (in image coordinates)
    lane_lines_image1 = [[146, 480, 286, 240], [528, 480, 329, 240]]
    lane_lines_image2 = [[203, 480, 313, 240], [428, 480, 363, 240]]

    # Convert lane lines from image coordinates to ground coordinates
    lane_lines_ground1 = [line_image2ground(lane_lines_image1[0]), line_image2ground(lane_lines_image1[1])]
    lane_lines_ground2 = [line_image2ground(lane_lines_image2[0]), line_image2ground(lane_lines_image2[1])]

    # Create particle filter
    lane_width = 0.3

    particle_filter = ParticleFilter(num_particles=1000, y_min=-lane_width / 2., y_max=lane_width / 2.,
                                     phi_min=-np.pi / 8., phi_max=np.pi / 8.)

    # First step
    print(particle_filter.step(v=0., omega=0., t=0., lane_lines=lane_lines_ground1))

    # Second step
    print(particle_filter.step(v=0.25, omega=0.5, t=0.25, lane_lines=lane_lines_ground2))

    # You should also resample particles after several steps
    # particle_filter.resample_particles()

    # cv2.destroyAllWindows()

    exit(0)


