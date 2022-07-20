"""Autonomous Lane-keeping using Raspberry Pi and OpenCV"""

import cv2 as cv
from cv2 import dilate
import numpy as np
import math 
import sys
import time
#import RPi.GPIO as PWM
from matplotlib import pyplot as plt



prev_frame = 0
new_frame = 0

def convert_to_HSV(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    cv.imshow("HSV", hsv)
    return hsv

def detect_edges(frame):
    #using a bluetape
    lower_blue = np.array([90,50,0], dtype="uint8") #lower limit blue color
    upper_blue = np.array([120,255,255], dtype="uint8") #upper blue limit
    mask = cv.inRange(hsv, lower_blue, upper_blue) #filter everything but the blue

    #detect edges
    mask = cv.erode(mask, None, iterations=1)
    mask = cv.dilate(mask, None, iterations=1)
    mask = cv.erode(mask, None, iterations=1)
    mask = cv.dilate(mask, None, iterations=1)

    edges = cv.Canny(mask, 60,60)
    cv.imshow("edges", edges)

def detect_stop(frame):
    stop = 0
    #detect red if it fits btn the upper and lower edges of the coor spectrum
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1= cv.inRange(hsv, lower_red, upper_red)

    red_mask = mask = mask0 + mask1
    cv.imshow("red_mask", red_mask)

    #if the frame has atleast 100 pixels of red, then there is a STOP sign
    has_stop = np.sum(red_mask)
    if has_stop > 100:
        stop = 1
        print("Stop Sign Detected")
    return stop

def region_of_interest(edges):
    height, width = edges.shape #extraxt the height and the width of the edges frame
    mask = np.zeros_like(edges) # make an empty matriix with the sane dimennsion of the edges frame

    #onyl focus on the lower half of the screen
    #specify the coordinates of the four points (lower/upper right/left)
    polygon = np.array([[
        (0, height*(8/8)),
        (0, height*(5/8)),
        (width, height*(8/8)),
        (width, height*(8/8)),
    ]], np.int32)

    cv.fillPoly(mask, polygon,255) #fills the polygon with the blue color
    cropped_edges = cv.bitwise_and(edges, mask)
    cv.imshow("roi", cropped_edges)

def detect_line_segment(cropped_edges):
    rho = 1
    theta = np.pi/180
    min_threshold = 10
    line_segments = cv.HoughLinesP(cropped_edges, rho, theta, min_threshold,
                                    np.array([]), minLineLength=2, maxLineGap=1)

    return line_segments

def average_slope_intercept(frame, line_segment):
    lane_lines = []

    if line_segments is None:
        print("no line segments")
        return lane_lines

    height, width = frame.shape
    left_fit = []
    right_fit = []
    boundary = 1/3

    left_region_boundary = width * (1-boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1,y1,x2,y2 in line_segment:
            if x1 == x2: #comparison
                print("skipping vertical lines (slope = infinty")
                continue

            fit = np.polyfit((x1,x2),(y1,y2), 1)
            slope = (y2-y1) / (x2-x1)
            intercept = y1 - (slope*x1)

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))

            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

            left_fit_average = np.average(left_fit, axis=0)
            if len(left_fit) > 0:
                lane_lines.append(make_points(frame, left_fit_average))

            right_fit_average = np.average(right_fit, axis=0)
            if len(right_fit) > 0:
                lane_lines.append(make_points(frame, right_fit_average))


        #lane_lines is a 2-D array consiting the coordinates of the right/left lane lines
        #i.e: lane_lines = [[x1,y1,x2,y2], [x1,y1,x2,y2]]
        #where the left array is for the left lane and the right array for the right lane
        #all coordinates are in pixels


            
            return lane_lines
def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height # bottom of the frame
    y2 = int(y1/2) #make points from the middle of the frame down

    if slope == 0:
        slope = 0.1

    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/ slope)

    return [[x1,y1,x2,y2]]

def display_lines(frame, lines, line_color=(0,255,0), line_width = 6):
    line_image = np.zeros_like(frame)

    if lines is None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv.line(line_image, (x1,y1), (x2,y2), line_color, line_width)

    line_image = cv.addWeighted(frame, 0.8, line_image, 1,1)
    return line_image

def get_steering_angle(frame, lane_lines):
    height, width, _ = frame.shape

    if len(lane_lines) == 2: #if the two lane lines are detected
        _,_, leftx2, _ = lane_lines[0][0] #extract left x2 from the lane_lines array
        _,_,rightx2,_ = lane_lines[1][0] #extract right x2 from the lane_lines array

        mid = int(width/2)
        x_offset = (leftx2 + rightx2)/2 - mid
        y_offset = int(height/2)
    elif len(lane_lines) == 1: #only one line is detected
        x1,_,x2,_ = lane_lines[0][0]
        x_offset = (x2 - x1)*.7
        y_offset = int(height/2)

    elif len(lane_lines) == 0: #no line is detected
        x_offset = 0
        y_offset = int(height/2)

    angle_to_mid_radian = math.atan(x_offset/y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0/math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle

def display_heading_line(frame, steering_angle, line_color=(0,0,255), line_width=5):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle/180.0*math.pi
    x1 = int(width/2)
    y1 = height
    x2 = int(x1 - height /2/ math.tan(steering_angle_radian))
    y2 = int(height/2)

    cv.line(heading_image, (x1,y1), (x2,y2), line_color, line_width)

    heading_image = cv.addWeighted(frame, 0.8,heading_image,1,1)
    cv.imshow("heading", heading_image)
    return heading_image


video = cv.VideoCapture(0)

#initialize states
state = [0,0]
#initialize params
params = [.25,.2,0]
speedPWM = 7.5
PWM.start("p9_14", 7.5, 50, 0)

#initialize the variables
i=0
check = 1 
dont_check = 0
stop_num = 0

#create a list for plotting
speedPWM_list = []
steerPWM_list = []
error_list = []
der_resp_list = []
prop_resp_list = []

#the loop
while True:
    #the motor should only start after some 50 frames have been recorded
    if i > 50:
        speedPWM = 7.915 #start the motors
        PWM.start("p9_14", 7.5, 50, 0)
    
    ret, frame  = video.read()
    #set the resolution to 100x60 for each frame
    frame  = cv.resize(frame, (100,60), fx=0,fy=0, interpolation=cv.INTER_CUBIC)
    #convert the frames hv
    hsv = convert_to_HSV(frame)
    edges = detect_edges(hsv)

    #if check = 1; check if there is a stop sign
    if check == 1:
        stop = detect_stop(hsv)
        #if the stop sign has been seen recently -> seen within the last 50 frames,
        #don't check for the stop sign
    else:
        #count another 50 frames before checking for the stop sign 
        if dont_check != 50:
            dont_check += 1
        else:
            dont_check = 0
            check = 1 #asign check 1
    if stop == 1:
        #count the number of stop signs seen, stop the car if the stop sign is seen
        stop_num += 1
        speedPWM = 7.5
        PWM.start("p9_14", 7.5, 50, 0)
        check = 0
        #if a seconf stop sign is seen, then stop the car permanently
        if stop_num == 2:
            break
        #at first stop sign, car stops for 2 seconds
        time.sleep(2)
        #later its speeds up
        speedPWM = 7.915 #start the motors
        PWM.start("p9_14", 7.5, 50, 0)
        stop = 0
    
    #functions for getting steering angles from the blueline detection
    roi = region_of_interest(edges)
    line_segments = detect_line_segment(roi)
    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    steering_angle  = (get_steering_angle(frame, lane_lines)) #steering direction for me