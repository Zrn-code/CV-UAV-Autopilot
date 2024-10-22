import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
import cv2.aruco as aruco


def main():
    f = cv2.FileStorage("Camera_Calibration.xml", cv2.FILE_STORAGE_READ)
    intrinsic = f.getNode("intrinsic").mat()
    distortion = f.getNode("distortion").mat()

    f.release()

    # 載入預定義的 ArUco 字典
    dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()
    

    while True:
        frame = frame_read.frame        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 轉換為灰階影像 (提高檢測效率)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 檢測 ArUco 標記
        markerCorners, markerIds, rejectedCandidates = aruco.detectMarkers(gray, dictionary, parameters=parameters)

        # 如果檢測到標記，則繪製標記並進行姿態估計
        if markerIds is not None:
            # 繪製檢測到的標記邊框
            frame = aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

            # 姿態估計
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)

            # 繪製每個標記的坐標軸
            for i in range(len(markerIds)):
                # 繪製每個標記的姿態 (坐標軸)
                aruco.drawAxis(frame, intrinsic, distortion, rvecs[i], tvecs[i], 10)
        # 顯示姿態估計的 x, y, z 值
            
            corner = markerCorners[i][0]  # 取得每個標記的第一個角點
            text_position = tuple(corner[0].astype(int))  # 取一個角點作為文字的起始位置
            
            x, y, z = tvecs[i][0]

            # 在影像上繪製文字
            cv2.putText(frame, f"x: {x:.4f}", (text_position[0], text_position[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"y: {y:.4f}", (text_position[0], text_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"z: {z:.4f}", (text_position[0], text_position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 顯示影像
        cv2.imshow('ArUco Marker Detection', frame)

        # 按下 'q' 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    #cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

