from CommonSettings import *
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.getcwd(), 'Gesture_DATA')
sequence_length = 30
mpDraw = mp.solutions.drawing_utils
mpDraw.draw_landmarks
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 0.7, 0.7) # Same as default parameter
actions = np.array(['Hello', 'TV', 'On', 'Off', 'Right', 'Left', 'Up', 'Down'])

hConnenctions = []
for t in mpHands.HAND_CONNECTIONS:
    for v in t:
        hConnenctions.append(v.value)
            
            
plt.figure(figsize=(15,8))
# for action in actions:
for action in ['Up']:
    # for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
    for sequence in range(1,6):
        window = []
        for frame_num in range(1,sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            # print(res)
            # res_compare = np.load(os.path.join(DATA_PATH, action, str(sequence+20), "{}.npy".format(frame_num)))

            
            # print(res_compare)
            rHand = np.transpose(np.reshape(res[:63],(-1,3)))
            lHand = np.transpose(np.reshape(res[63:],(-1,3)))
            

            plt.subplot(1,2,1)
            for lms in [rHand,lHand]:
                if np.count_nonzero(lms)>45:
                    plt.plot(lms[0],lms[1],'bo')
                    for i in range(0,41,2):
                        plt.plot([lms[0][hConnenctions[i]],lms[0][hConnenctions[i+1]]],[lms[1][hConnenctions[i]],lms[1][hConnenctions[i+1]]],'r-')
            plt.title(action+' original')
            plt.text(0,0,'frame #'+str(frame_num))
            plt.axis('square')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.gca().invert_yaxis()
            plt.grid()
            # plt.show()



            rHand_compare = np.transpose(np.reshape(res_compare[:63],(-1,3)))
            lHand_compare = np.transpose(np.reshape(res_compare[63:],(-1,3)))

            plt.subplot(1,2,2)
            for lms in [rHand_compare,lHand_compare]:
                if np.count_nonzero(lms)>45:
                    plt.plot(lms[0],lms[1],'bo')
                    for i in range(0,41,2):
                        plt.plot([lms[0][hConnenctions[i]],lms[0][hConnenctions[i+1]]],[lms[1][hConnenctions[i]],lms[1][hConnenctions[i+1]]],'r-')
            plt.title(action+' augmented')
            plt.text(0,0,'frame #'+str(frame_num))
            plt.axis('square')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.gca().invert_yaxis()
            plt.grid()

            
            plt.show(block=False)
            plt.pause(0.08)
            plt.clf()
            window.append(res)
        # print(window)
