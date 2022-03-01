import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


camera_width, camera_height = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, camera_width)
cap.set(4, camera_height)

detector = htm.HandDetector(detect_conf=0.6)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]



while True:
    success, img = cap.read()
    detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:

        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1+x2) // 2, (y1+y2) //2

        cv2.circle(img, (x1,y1), 5, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 5, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2,y2), (255, 0, 255), 3)
        cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # Hand Range 50-300 and vol range = (-65, 0)
        vol = np.interp(length, [50,300], [min_vol, max_vol])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if(length<50):
            cv2.circle(img, (cx,cy), 5, (0,255,0), cv2.FILLED)

    cv2.imshow("Img", img)
    cv2.waitKey(1)

