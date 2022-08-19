from math import remainder
import cv2
import numpy as np
import os
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import tensorflow as tf


##############################Detection setup####################################
mp_hands = mp.solutions.hands     # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
#################################################################################



############################Setup Folders for Collection#########################
# Path for exported data, numpy arrays
DATA_PATH = 'C:/gradProject/Gesture_DATA'

# Actions that we try to detect
actions = np.array(['Hello', 'TV', 'On', 'Off'])                 # can add actions
# actions = np.array(['AC', 'Hello', 'TV', 'Channel', 'Volume', 'On', 'Off', 'Next', 'Prev', 'OK', 'Light', 'Brightness'])                 # can add actions
# Thirty videos worth of data
no_sequences = len(list(os.listdir(os.path.join(DATA_PATH,actions[0]))))


# Videos are going to be 30 frames in length
sequence_length = 30

#################################################################################

# ###############Preprocess Data and Create Labels and Features####################
# # one-hot encoding
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import time
label_map = {label:num for num, label in enumerate(actions)}
# t1 = time.time()



sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            # if np.dtype(res[0]) == 'float64':
            #     print(type(res[0]))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
# print(time.time()-t1)

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) # 90% of data will be used as train data
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
print(X.shape)
# #################################################################################



#######################Build and Train LSTM Neural Network########################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GRU, LSTM
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir) # to debug training process
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

model = Sequential()
model.add(LSTM(4, return_sequences=True,  input_shape=(30,126))) # 126 = 21(left hand)*3 + 21(right hand)*3
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(LSTM(8, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#################################################################################
label_map = {label:num for num, label in enumerate(actions)}

def my_generator():
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            yield (window, to_categorical(label_map[action],4).astype(int))

dataset = tf.data.Dataset.from_generator(my_generator,(tf.float32, tf.int8))
dataset = dataset.batch(10)

#################################################################################

model.fit(X_train,y_train, validation_data = (X_val, y_val), epochs=300, callbacks=[es, tb_callback]) # change epochs to prevent overfitting

model.summary()    # for debug
# #################################################################################



##################################Save Weights####################################

model.save('C:/gradProject/Model/action_epoch114_es.h5')

model.load_weights('C:/gradProject/Model/action_epoch114_es.h5')
##############################optimizing model###################################
# convert h5 model into tflite model
h5model = tf.keras.models.load_model('action_test.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(h5model)
tflite_model = converter.convert()

with open('model_ACTEST.tflite','wb') as f:
    f.write(tflite_model)

################Evaluation using Confusion Matrix and Accuracy###################
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score  

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))

print(accuracy_score(ytrue, yhat))
#################################################################################


