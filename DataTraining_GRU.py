from CommonSettings import *
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


##############################Detection setup####################################
mp_hands = mp.solutions.hands     # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

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
PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'Gesture_DATA')
MODEL_PATH = os.path.join(PATH, 'Model')
no_sequences = trueDataCnt

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
    s = np.load(os.path.join(DATA_PATH, 'merged_DATA', "{}Data.npy".format(action)))
    if len(sequences) == 0:
        sequences = s
    else:
        sequences = np.append(sequences, s, axis=0)
    label = np.zeros(augDataCnt) + label_map[action]
    labels = np.append(labels,label)

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) # 80% of data will be used as train data
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
print(X.shape, y.shape)
# #################################################################################



#######################Build and Train LSTM Neural Network########################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GRU, LSTM
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir) # to debug training process
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model = Sequential()
model.add(GRU(16, return_sequences=True, dropout=0.5, input_shape=(30,126))) # 126 = 21(left hand)*3 + 21(right hand)*3
model.add(BatchNormalization())
model.add(GRU(12, return_sequences=False, dropout=0.5,))
model.add(BatchNormalization())
# model.add(Dense(12, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train,y_train, validation_data = (X_val, y_val),batch_size=32, epochs=300, callbacks=[es, tb_callback]) # change epochs to prevent overfitting

model.summary()    # for debug
# #################################################################################

modelName = 'action_0826_1.h5'

##################################Save Weights####################################
model.save(os.path.join(MODEL_PATH ,modelName))

model.load_weights(os.path.join(MODEL_PATH,modelName))
##############################optimizing model###################################
# # convert h5 model into tflite model
# h5model = tf.keras.models.load_model('action_test.h5')

# converter = tf.lite.TFLiteConverter.from_keras_model(h5model)
# tflite_model = converter.convert()

# with open('model_ACTEST.tflite','wb') as f:
#     f.write(tflite_model)

################Evaluation using Confusion Matrix and Accuracy###################
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score  

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))

print(accuracy_score(ytrue, yhat))
#################################################################################


