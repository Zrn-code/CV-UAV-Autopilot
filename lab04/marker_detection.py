import cv2
import cv2.aruco as aruco
import numpy as np


f = cv2.FileStorage("Camera_Calibration.xml", cv2.FILE_STORAGE_READ)
intrinsic = f.getNode("intrinsic").mat()
distortion = f.getNode("distortion").mat()

f.release()

# 開啟攝影機
cap = cv2.VideoCapture(0)

# 載入預定義的 ArUco 字典
dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

while True:
    # 從攝影機中讀取影像
    ret, frame = cap.read()

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
        pose_text = f"Pose Estimation\nx: {x:.4f}\ny: {y:.4f}\nz: {z:.4f}"
        
        # 在影像上繪製文字
        cv2.putText(frame, f"x: {x:.4f}", (text_position[0], text_position[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"y: {y:.4f}", (text_position[0], text_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"z: {z:.4f}", (text_position[0], text_position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # 顯示影像
    cv2.imshow('ArUco Marker Detection', frame)

    # 按下 'q' 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機並關閉所有視窗
cap.release()
cv2.destroyAllWindows()
