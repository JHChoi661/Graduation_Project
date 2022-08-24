
import time
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.85, trackCon=0.85):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        self.detectionCon, self.trackCon) # Same as default parameter
        
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(self.results.multi_hand_landmarks)
        self.ifHands = 0
        if self.results.multi_hand_landmarks:
            self.ifHands = 1
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # print(handLms)
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # print(len(self.results.multi_hand_landmarks))
            for id, lm in enumerate(myHand.landmark):
                # if id == 20:print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, False)
        # if len(lmList) != 0:
        #     print(lmList)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,
        (255,0,255),3)
        cv2.imshow("Image", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        # time.sleep(0.01)
    # print(lmList)
    cap.release()

if __name__ == "__main__":
    # img = cv2.imread('C:\gradProject\still1.jpg')
    # detector = handDetector()
    
    # for i in range(1,8):
    #     img = cv2.imread(os.path.join('C:\gradProject', 'still{}.jpg'.format(i)))
    #     img = detector.findHands(img)
    #     lmList = detector.findPosition(img, False)
    #     cv2.imshow("Image", img)
    #     cv2. waitKey(0)
    main()
    
