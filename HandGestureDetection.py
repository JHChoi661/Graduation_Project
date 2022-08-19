
import cv2
import numpy as np
import os
import math
from time import *
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import tensorflow as tf
import psutil
import time

### init
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                                # use tensorflow-cpu


### detection setup
mp_hands = mp.solutions.hands      # Hand detection model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results):
    if len(results.multi_hand_landmarks) == 1:                             # only one hand detected
        for _, hand_handedness in enumerate(results.multi_handedness):
            handedness_dict = MessageToDict(hand_handedness)
            if handedness_dict['classification'][0]['label'] == 'Left':    # distinguish right or left hand
                rh = np.zeros(21*3)
                lh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[0].landmark)]).flatten() # each landmark has 3 factors(x, y, z)            
                # print(lh)
            else:
                rh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[0].landmark)]).flatten()
                lh = np.zeros(21*3)
                # print(rh)
    elif len(results.multi_hand_landmarks) == 2:
        rh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[0].landmark)]).flatten()
        lh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[1].landmark)]).flatten()

    return np.concatenate([rh, lh])

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

DATA_PATH = 'C:/gradProject/Gesture_DATA'

# Actions to detect
actionsDict = {'Hello':0b000, 'TV':0b001, 'On':0b010, 'Off':0b010}       # use dict to sequence control
# actions = np.array(['Hello', 'TV', 'Channel', 'Volume', 'On', 'Off', 'Next', 'Prev', 'OK', 'Light', 'Brightness'])                          # can add actions
actions = np.array(['Hello', 'TV', 'On', 'Off'])                          # can add actions

# detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.95         # min confidence of prediction


cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0
model = tf.keras.models.load_model('C:/gradProject/Model/action_epoch300.h5')
skipCnt = 0 
stage = -1
# Set mediapipe model 
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():

        ret, frame = cap.read()

        img, r = mediapipe_detection(frame, hands)
        
        if r.multi_hand_landmarks:
            draw_landmarks(img, r)
            keypoints = extract_keypoints(r)
        # print(keypoints)
            sequence.append(keypoints)
            if skipCnt == 5:
                skipCnt = 0
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence,axis=0))[0]
                    # print(res)
                    if res[np.argmax(res)] > threshold:
                        # print(actions[np.argmax(res)])
                        # 순서 로직
                        if actionsDict.get(actions[np.argmax(res)],'-1') == 0b111:
                            stage = 0
                        elif actionsDict.get(actions[np.argmax(res)],'-1') == stage + 1:
                            curActKey = actions[np.argmax(res)]
                            curActValue = actionsDict.get(curActKey,'-1')    
                            print(curActKey)
                            if curActKey == 'On' or curActKey =='Off':
                                stage = 0
                            else:
                                stage += 1

                            time.sleep(0.3)


            skipCnt+=1
        else:
            sequence = []
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,
        (255,0,255),3)

        cv2.imshow('Idle', img)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
#################################################################################
    
