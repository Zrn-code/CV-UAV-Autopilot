import cv2
import numpy as np

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
f.release()

cap.release()
cv2.destroyAllWindows()
