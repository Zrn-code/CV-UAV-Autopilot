import cv2
import numpy as np
from djitellopy import Tello

# Parameters
num_images = 5
columns = 9
rows = 6
patternSize = (columns, rows)
winSize = (11, 11)
zeroZone = (-1, -1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (columns-1, rows-1, 0)
objp = np.array([[x, y, 0] for y in range(rows) for x in range(columns)], dtype=np.float32)
obj_points = [objp for _ in range(num_images)]  
img_points = []  

# Open the camera
cap = cv2.VideoCapture(0) 
saved_images = 0

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
while True:
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corner = cv2.findChessboardCorners(image, patternSize, None)

    if ret:  
        # Refine the corner positions
        corner = cv2.cornerSubPix(image, corner, winSize, zeroZone, criteria)  
        img_points.append(corner)
        frame = cv2.drawChessboardCorners(frame, patternSize, corner, ret)
        
        filename = f'image_{saved_images}.png'
        cv2.imwrite(filename, frame)
        saved_images += 1 
        
    # Display the frame with drawn corners
    cv2.imshow('frame', frame)
    cv2.waitKey(33) 

    if saved_images >= num_images:
        break

# Perform camera calibration using the object points and image points
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image.shape[::-1], None, None)

f = cv2.FileStorage('Camera_Calibration.xml', cv2.FILE_STORAGE_WRITE)
f.write('intrinsic', cameraMatrix)
f.write('distortion', distCoeffs)

f = cv2.FileStorage("Camera_Calibration.xml", cv2.FILE_STORAGE_READ)
cameraMatrix = f.getNode("intrinsic")
distCoeffs = f.getNode("distortion")

f.release()

drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

cap.release()
cv2.destroyAllWindows()

while True:
    frame = frame_read.frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Frame",frame)
    cv2.waitKey(33)
    intrinsic = cameraMatrix
    distortion = distCoeffs
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    
    if markerIds is not None:

        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)

        for rvec, tvec in zip(rvecs, tvecs):
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec, tvec, 0.1)

            position_text = f"X: {tvec[0][0]:.2f}, Y: {tvec[0][1]:.2f}, Z: {tvec[0][2]:.2f}"
            cv2.putText(frame,"x = "+str(tvec[0,0,0])+", y = "+str(tvec[0,0,1])+", z = "+str(tvec[0,0,2]), (0,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)

    cv2.imshow("Tello ArUco Detection", frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
