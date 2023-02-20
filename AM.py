import numpy as np
import cv2
import cv2.aruco as aruco
from picamera2 import Picamera2, MappedArray, Preview
import time
import signal



# Initialize the camera and grab a reference to the raw camera capture
picam2 = Picamera2()
picam2.start_preview(Preview.DRM)
config = picam2.create_preview_configuration({"size": (640, 480),"format": "BGR888>
picam2.configure(config)

# Define the ArUco dictionary and parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters_create()

# Define the ArUco dictionary and parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters_create()

# Define the intrinsic parameters of the camera
focal_length = 700.0
center = (320.0, 240.0)
camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1>

# Define the distortion coefficients of the camera
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# OSD
colour = (0, 255, 0)
origin = (30, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 1

# interrupts
def signal_handler(sig, frame):
    print("Caught KeyboardInterrupt")
    picam2.stop()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Capture frames from the camera
def aruco_marker(request):
    # get frame
    with MappedArray(request, "main") as m:
        # count time
        start_time = time.time()

        # Convert the image to grayscale
        gray = cv2.cvtColor(m.array, cv2.COLOR_BGR2GRAY)

        # Detect the ArUco markers in the image
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, pa>

        #print("Image shape: ", m.array.shape)
        # If any markers are detected, estimate their pose
        if ids is not None:
            # Estimate the pose of the detected marker(s)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camer>

            # Draw the ArUco markers and axes on the frame
            aruco.drawDetectedMarkers(m.array, corners, ids)
            for i in range(len(ids)):
                aruco.drawAxis(m.array, camera_matrix, dist_coeffs, rvecs[i], tvec>

            # Calculate the angle between the camera and the marker
            rmat, _ = cv2.Rodrigues(rvecs[0])
            normal_vector = np.dot(rmat, np.array([0, 0, 1]))
            camera_vector = np.array([0, 0, -1])
            angle = np.arccos(np.dot(normal_vector, camera_vector) / (np.linalg.no>
            angle_degrees = angle * 180 / np.pi

        # elapsed time
        elapsed_time = time.time() - start_time

        # FPS
        fps = 1 // elapsed_time

        # OSD
        cv2.putText(m.array, str(fps), origin, font, scale, colour, thickness)

# Start
picam2.pre_callback = aruco_marker
picam2.start()

while 1:
    time.sleep(5)
