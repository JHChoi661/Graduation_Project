import numpy as np
import os
import random
import math
import shutil

sequence_length = 30       # frame per one data set
PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'Gesture_DATA')
actions = np.array(['Hello','TV', 'On', 'Off']) # list of actions
no_sequences = 5              # number of true data (not augmented data)

def loadData():

    ### lodaData() : 촬영으로 생성된 실제 데이터 로드

    label_map = {label:num for num, label in enumerate(actions)}
    sequenceDict = {}
    
    for action in actions:
        sequences = []
        for sequence in range(1,no_sequences+1):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                res = np.float32(res)
                # res : 1프레임 데이터
                window.append(res)
                # window : 30프레임 데이터 
            sequences.append(window)
            # labels.append(label_map[action])
        sequenceDict[action] = sequences   

    return sequenceDict

def dataAug_Translation(trueData):
    data = np.copy(trueData)
    ### dataAug_Translation(trueData) : trueData를 랜덤한 x, y축에 대해 평행이동

    # transVal : 평행이동 값, 전체 윈도우 크기의 최대 10% 평행이동 가능
    # transSign : 평행이동 방향, 0:+, 1:-
    # transAxis : 평행이동 축, 0:x, 1:y

    for oneDataSet in data:
        transVal = random.randint(1,100) / 1000.0
        transSign = random.randint(0,1)
        transAxis = random.randint(0,1)
        if transSign == 1: transVal = -transVal   
        for frame_1 in oneDataSet:
            for idx in range(len(frame_1)):
                if frame_1[idx]: # NULL(0) 데이터 처리
                    if transAxis == 0:                     
                        if idx % 3 == 0:
                            frame_1[idx] += transVal
                    else:
                        if idx % 3 == 1:
                            frame_1[idx] += transVal
    return data


def dataAug_flip(trueData):
    data = np.copy(trueData)

    ### dataAug_flip(trueData) : trueData를 x축에 대해 대칭이동
    ### 대칭이동 시 반대 손에 해당하는 데이터 위치로 이동

    for oneDataSet in data:
        for i in range(len(oneDataSet)):
            for idx in range(len(oneDataSet[i])):                      
                if idx % 3 == 0 and oneDataSet[i][idx]!=0:
                    oneDataSet[i][idx] = 1-oneDataSet[i][idx]
            flipedData = np.concatenate([oneDataSet[i][63:], oneDataSet[i][:63]])
            oneDataSet[i] = flipedData
    return data

def dataAug_Rotate_RotateHandData(handData, landmark9_x, landmark9_y, rotateRad):
    
    ### dataAug_Rotate_RotateHandData(handData, landmark9_x, landmark9_y, rotateRad)
    ### 오른손, 왼손 데이터(handData)를 각 손의 landmark9 좌표를 기준으로 rotateRad 만큼 회전
    for idx in range(0,63,3):
        nx = np.round((handData[idx]-landmark9_x)*math.cos(rotateRad)
        - (handData[idx+1]-landmark9_y)*math.sin(rotateRad),9) + landmark9_x
        ny = np.round((handData[idx]-landmark9_x)*math.sin(rotateRad)
        + (handData[idx+1]-landmark9_y)*math.cos(rotateRad),9) + landmark9_y
        handData[idx] = nx
        handData[idx+1] = ny

    return np.float32(handData)

def dataAug_Rotate_Preprocessing(trueData):
    data = np.copy(trueData)
    ### dataAug_Rotate_Preprocessing(data)
    ### data의 각 데이터 셋(oneDataSet)을 오른손 데이터와 왼손 데이터로 분리

    for oneDataSet in data:
        degree = random.randint(-30,30)
        rotateRad = degree * (math.pi / 180.0)
        for i in range(len(oneDataSet)):
            rightHand = oneDataSet[i][:63]
            leftHand  = oneDataSet[i][63:]
            rightHand_lm9_x, rightHand_lm9_y = rightHand[24], rightHand[25] # landmark 9
            leftHand_lm9_x , leftHand_lm9_y  =  leftHand[24],  leftHand[25] # landmark 9
            # print(rightHand_lm9_x, rightHand_lm9_y)

            rotatedData = np.concatenate(
            [dataAug_Rotate_RotateHandData(rightHand, rightHand_lm9_x, rightHand_lm9_y, rotateRad)
            , dataAug_Rotate_RotateHandData(leftHand, leftHand_lm9_x, leftHand_lm9_y, rotateRad)])
            oneDataSet[i] = rotatedData
    return data

def dataAug_windowWarping(trueData):
    data = np.copy(trueData)
    ### dataAug_windowWarping(data) 30frame의 데이터 중 4개의 구간을 임의로 정하여
    ### 2개의 구간은 3frame을 1frame으로 down sample, 2개의 구간은 1frame을 3frame으로 up sample한다.
    ### down sample, up sample 구간은 랜덤하게 결정
    ### down sample : 3frame 각 좌표의 평균 값
    ### up sample : target frame과 그 앞 뒤 좌표와의 평균값을 구하여 3frame으로 확장

    for oneDataSet in range(len(data)):
        # down sample
        # r : down sample 할 target frame의 index 2개

        r = sorted(random.sample(range(1,28),2),reverse=True)
        # r = [28, 27]  
        tempData = np.copy(data[oneDataSet])
        for j in r:            
            
            prevFrameNumber = j-1
            nextFrameNumber = j+1
            avg = np.round((tempData[j] + tempData[prevFrameNumber] + 
                                            tempData[nextFrameNumber]) / 3.0, 9)
            # print(type(avg))
            tempData[j] = avg
            tempData = np.delete(tempData,nextFrameNumber,0)
            tempData = np.delete(tempData,prevFrameNumber,0)
        

        # up sample
        # r1 : down sample 후의 변한 index r의 값 
        # r2 : up sample 할 target frame의 index 2개, r1과의 중복 방지

        r1 = [r[0]-3,r[1]-1]
        r2 = sorted(random.sample(range(1,24),2),reverse=True)
        while True:
            n1, n2 = r2
            if n1 in r1 or n2 in r1:
                r2 = sorted(random.sample(range(1,25),2),reverse=True)
            else: break
        for j in r2:
            prevFrameNumber = j-1
            nextFrameNumber = j+1
            prev_target_avg = np.round((tempData[j]+tempData[prevFrameNumber])/2.0,9)
            next_target_avg = np.round((tempData[j]+tempData[nextFrameNumber])/2.0,9)
            tempData = np.insert(tempData, nextFrameNumber, next_target_avg,0)
            tempData = np.insert(tempData, j, prev_target_avg,0)
        data[oneDataSet] = tempData
    return data

    
def dataAug_Rescale(data):

    ### dataAug_Rescale(data) : 전체 좌표를 랜덤한 배수로 rescale

    # rescaleRate : rescale 배수, 0.5~1.5

    for oneDataSet in data:
        rescaleRate = random.randint(180,240) / 200.0
        # print(rescaleRate)
        for frame_1 in oneDataSet:
            for idx in range(len(frame_1)):
                if frame_1[idx]:
                    if idx % 3 == 0 or idx % 3 == 1:
                        frame_1[idx] *= rescaleRate
                    else: pass
                else: pass
    return data

def normalizeAugmentedData(originalData):
    data = np.copy(originalData)
    normalizeFlag = 0
    for idx, oneDataSet in enumerate(data):
        # print(oneDataSet)
        rHand = np.array([])
        lHand = np.array([])
        for frame in oneDataSet:
            rHand = np.append(rHand,frame[:63])
            lHand = np.append(lHand,frame[63:])
        
        rHand = np.transpose(np.reshape(rHand,(-1,3)))
        lHand = np.transpose(np.reshape(lHand,(-1,3)))

        rHand_x = rHand[0]
        rHand_y = rHand[1]

        lHand_x = lHand[0]
        lHand_y = lHand[1]
        
        rHand_xMax = np.max(rHand_x)
        rHand_xMin = np.min(rHand_x)
        
        rHand_yMax = np.max(rHand_y)
        rHand_yMin = np.min(rHand_y)

        lHand_xMax = np.max(lHand_x)
        lHand_xMin = np.min(lHand_x)
        
        lHand_yMax = np.max(lHand_y)
        lHand_yMin = np.min(lHand_y)

        if rHand_xMax != rHand_xMin:
            if rHand_xMax > 0.999:
                rHand_x = rHand_x-(rHand_xMax - 0.999)
                normalizeFlag = 1
                rHand[0]=rHand_x
            elif rHand_xMin < 0.001:
                rHand_x = rHand_x-(rHand_xMin - 0.001)
                normalizeFlag = 1
                rHand[0]=rHand_x

        if rHand_yMax != rHand_yMin:
            if rHand_yMax > 0.999:
                rHand_y = rHand_y-(rHand_yMax - 0.999)
                normalizeFlag = 1
                rHand[1]=rHand_y
            elif rHand_yMin < 0.001:
                rHand_y = rHand_y-(rHand_yMin - 0.001)
                normalizeFlag = 1
                rHand[1]=rHand_y


        if lHand_xMax != lHand_xMin:
            if lHand_xMax > 0.999:
                lHand_x = lHand_x-(lHand_xMax - 0.999)
                normalizeFlag = 1
                lHand[0]=lHand_x
            elif lHand_xMin < 0.001:
                lHand_x = lHand_x-(lHand_xMin - 0.001)
                normalizeFlag = 1
                lHand[0]=lHand_x
        if lHand_yMax != lHand_yMin:
            if lHand_yMax > 0.999:
                lHand_y = lHand_y-(lHand_yMax - 0.999)
                normalizeFlag = 1
                lHand[1]=lHand_y
            elif lHand_yMin < 0.001:
                lHand_y = lHand_y-(lHand_yMin - 0.001)
                normalizeFlag = 1
                lHand[1]=lHand_y

        if normalizeFlag == 1:
            # print('normalize')            
            rHand = np.reshape(np.transpose(rHand),(-1))
            lHand = np.reshape(np.transpose(lHand),(-1))
            for i in range(0,30):
                data[idx][i] = np.concatenate((rHand[63*i:63*(i+1)],lHand[63*i:63*(i+1)]))
    return data


def saveAugmentedData(data, action):

    ### saveAugmentedData(data)
    dataLen = len(list(os.listdir(os.path.join(DATA_PATH,action))))
    for sequence in range(1,len(data)+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence+dataLen)))
        except:
            pass
        frame_num = 0
        for frame_1 in data[sequence-1]:
            # print(frame_1)
            npy_path = os.path.join(DATA_PATH, action, str(sequence+dataLen), str(frame_num))
            frame_num += 1
            np.save(npy_path,frame_1)


def clearAugmentedData():
    for action in actions:
        dataLen = len(list(os.listdir(os.path.join(DATA_PATH,action))))
        augmentedDataIdx = list(range(no_sequences,dataLen))
        for i in augmentedDataIdx:
            shutil.rmtree(os.path.join(DATA_PATH, action, str(i+1)))


def generateAugmentedData():
    dataDict = loadData()
    for action in actions:
        data = dataDict[action]
        data_copy = np.copy(data)
        for _ in range(400):       # range X no_sequences = number of aug data
            # augType = random.randint(1,3)
            # if augType == 1:
            #     augmentedData = dataAug_Translation(data_copy)
            # elif augType == 2:
            #     augmentedData = dataAug_Rotate_Preprocessing(data_copy)
            # elif augType == 3:
            #     augmentedData = dataAug_windowWarping(data_copy)
            # else: pass                
            # print(augType, np.mean(augmentedData[0][0]))
            augmentedData = dataAug_Translation(data_copy)
            augmentedData = dataAug_Rotate_Preprocessing(augmentedData)
            augmentedData = dataAug_windowWarping(augmentedData)
            flip = random.randint(0,1)
            if flip:
                augmentedData = dataAug_flip(augmentedData)
            augmentedData = dataAug_Rescale(augmentedData)
            # print(type(augmentedData[0][0][0]))
            augmentedData = normalizeAugmentedData(augmentedData)
            saveAugmentedData(augmentedData, action)

if __name__ == "__main__":
    
    clearAugmentedData()
    generateAugmentedData()
    



