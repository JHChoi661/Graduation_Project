

import numpy as np
import os
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import time

actions = ['Hello','TV', 'On', 'Off', 'Left', 'Right', 'Down', 'Up']

PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'Gesture_DATA')
MERGED_DATA_PATH = os.path.join(DATA_PATH, 'merged_DATA')
MODEL_PATH = os.path.join(PATH, 'Model')