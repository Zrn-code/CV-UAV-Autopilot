import cv2
import numpy as np

def Q1():

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

def Q2():
    # Open a connection to the webcam (usually 0 or 1 for built-in webcams)
    cap = cv2.VideoCapture(0)

    # Load the background image
    background = cv2.imread('screen.jpg')

    # Define the coordinates of the four corners of your webcam frame and the corresponding coordinates on the background
    webcam_corners = np.array([[414, 874], [1639, 207], [329, 1406], [1659, 1249]], dtype=np.float32)

    # Create a named window and make it resizable
    cv2.namedWindow('Mapped Webcam Frame', cv2.WINDOW_NORMAL)

    # Resize the window to a smaller size (e.g., 640x480)
    cv2.resizeWindow('Mapped Webcam Frame', 640, 480)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        print("read")
        h, w, _ = frame.shape
        screen_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)

        # Calculate the perspective transform matrix using getPerspectiveTransform
        perspective_matrix = cv2.getPerspectiveTransform(webcam_corners, screen_corners)

        # Iterate through each pixel in the output image and perform bilinear interpolation
        for y in range(background.shape[0]):
            for x in range(background.shape[1]):
                point = np.array([[x, y]], dtype=np.float32)
                transformed_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), perspective_matrix)
                x_transformed, y_transformed = transformed_point[0][0]

                # Check if the transformed coordinates are within the bounds of the webcam frame
                if 0 <= x_transformed < frame.shape[1] - 1 and 0 <= y_transformed < frame.shape[0] - 1:
                    x0, y0 = int(x_transformed), int(y_transformed)
                    x1, y1 = x0 + 1, y0 + 1

                    # Perform bilinear interpolation
                    dx = x_transformed - x0
                    dy = y_transformed - y0
                    background[y, x] = (1 - dx) * (1 - dy) * frame[y0, x0] + dx * (1 - dy) * frame[y0, x1] + (1 - dx) * dy * frame[y1, x0] + dx * dy * frame[y1, x1]

        # Display the output
        cv2.imshow('Mapped Webcam Frame', background)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Lab03 Demo: Q1: Camera Calibration Q2: Webcam Warping") 
    while True:
        choice = input("Enter your choice (Q1/Q2): ")
        if choice == "Q1":
            Q1()
        elif choice == "Q2":
            Q2()
        else:
            print("Invalid choice. Please enter Q1, Q2")
            continue