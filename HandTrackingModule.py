import mediapipe as mp
import cv2
import time


class HandDetector():
    def __init__(self, mode = False, max_hands = 2, detect_conf = 0.5, track_conf = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detect_conf = detect_conf
        self.track_conf = track_conf

        self.mp_hands = mp.solutions.hands
        # create the object
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detect_conf, self.track_conf)
        self.mp_draw = mp.solutions.drawing_utils


    def find_hands (self, img, draw = True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # process is method of object hands
        self.results = self.hands.process(img_RGB)

        # To check if something is detected or not
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no = 0, draw = True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,255),cv2.FILLED)
        return lm_list

def main():
    prev_time = 0
    curr_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandTrackingModule()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        if len(lm_list) != 0:
            print(lm_list[1])


        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time =  curr_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

    