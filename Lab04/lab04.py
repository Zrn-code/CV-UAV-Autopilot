import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

def main():
    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()
    

    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("drone", frame)
        key = cv2.waitKey(33)
    
    #cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

