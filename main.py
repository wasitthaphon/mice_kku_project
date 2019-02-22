import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import sys
import time


# Find mice position function
def FindMicePosition(frame):
    micePosition = [[], []]

    cnts, heirarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(cnts, key=cv2.contourArea)

    paddingX = 35
    paddingY = 35
    ratioY = 0.4

    for i in range(2):
        x,y,w,h = cv2.boundingRect(sorted_cnts[len(sorted_cnts)-(i+1)])
        micePosition[i] = {
            "x" : x-paddingX,
            "y" : y+10,
            "w" : w+(2*paddingX),
            "h" : h+paddingY+10
        }

    return micePosition

# Make frame to black white
def MakeFrameToBlackWhite(frame, micePosition):
    thresh = 150
    maxval = 255
    miceImage = [[], []]

    # Image to gray
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Image to black white
    ret, frame = cv2.threshold(frame, thresh, maxval, cv2.THRESH_BINARY)

    for i in range(2):
        miceImage[i] = frame[micePosition[i]['y']:(micePosition[i]['y'] + micePosition[i]['h']),
                            micePosition[i]['x']:(micePosition[i]['x'] + micePosition[i]['w'])]

    return miceImage


# Image smooth
def Convolution(miceImage):
    kernel = np.ones((5,5), np.float32)/30
    mice = [[], []]
    for i in range(len(miceImage)):
        mice[i] = cv2.filter2D(miceImage[i], -1, kernel)

    return mice

def Gaussian(miceImage):
    mice = [[], []]
    for i in range(len(miceImage)):
        mice[i] = cv2.GaussianBlur(miceImage[i], (7, 7), 0)

    return mice


# Make image to 1 0 array
def Normalization(frame):
    for i in range(len(frame)):
        frame[i] = frame[i] / 255
        frame[i] = frame[i].astype(int)

    return frame


# Skeleton image
def Skeletonization(miceImage):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    mice = [[], []]
    for i in range(len(miceImage)):
        done = False
        skel = np.zeros(miceImage[i].shape,np.uint8)
        size = np.size(miceImage[i])
        while (not done):
            eroded = cv2.erode(miceImage[i], element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(miceImage[i], temp)
            skel = cv2.bitwise_or(skel, temp)
            miceImage[i] = eroded

            zeros = size - cv2.countNonZero(miceImage[i])
            if zeros == size:
                done = True
        mice[i] = skel
    return mice

# Count 
def Counting(miceImage):
    mice = []

    for i in range(len(miceImage)):
        mice.append(sum(1 for rows in miceImage[i] for val in rows if val))

    return mice


# Draw keypoints
def DrawKeypoints(image, kps, color=(0, 255, 0)):
    for kp in kps:
        x, y = kp.pt
        cv2.circle(image, (int(x), int(y)), 3, color, -1)
    return image


# SIFT
def SIFT(miceImage):
    sift = cv2.ORB_create()
    mice = [[], []]

    for i in range(len(miceImage)):
        kp = sift.detect(miceImage[i], None)
        kp, des = sift.compute(miceImage[i], kp)
        mice[i] = kp

    return mice

# Plot
class Plot:
    mice = []

    def __init__(self, ax):
        self.ax = ax

    def Plot(self, miceImage):
        axis = [211, 212]
        x = []
        y = []
        plt.clf()
        print('KP : {}, {}'.format(len(miceImage[0]), len(miceImage[1])))
        for i in range(len(miceImage)):
            x.clear()
            y.clear()
            plt.subplot(axis[i])
            for kp in miceImage[i].copy():
                j, k = kp.pt
                x.append(j)
                y.append(k)
            maximumVla = max(y)
            inverseY = [(maximumVla - eachY) for eachY in y]
            plt.scatter(x,inverseY)
                # plt.pause(0.5)
            # plt.subplot(axis[i])
            # u,v = np.meshgrid(x,y)
            # q = plt.quiver(x,y,u,v)
            # plt.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length 10', labelpos='E')
            self.mice.append([x,y])


# Main
def Main(video_path, micePosition):

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_counter = 0
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_counter = frame_counter + 1
        if (frame_counter % fps) != 0:
            continue
        miceBW = MakeFrameToBlackWhite(frame, micePosition)
        miceBW = Convolution(miceBW)
        miceBWCopy = miceBW.copy()
        miceSkel = Skeletonization(miceBWCopy).copy()

        miceKeypoints = SIFT(miceSkel).copy()

        # miceSkelSIFT = SIFT(miceSkel)
        cv2.imshow("M1", miceBW[0])
        cv2.imshow("M2", miceBW[1])
        cv2.imshow("S1", miceSkel[0])
        cv2.imshow("S2", miceSkel[1])

        mPlot = Plot([ax1, ax2])
        mPlot.Plot(miceKeypoints)
        plt.savefig('tmp.png')
        plotImage = cv2.imread('tmp.png', cv2.IMREAD_COLOR)
        cv2.imshow("Plot", plotImage)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()
    

# Start point
if __name__ == '__main__':
    video_path = 'mice2.mpg'
    
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, frame_bw = cv2.threshold(frame_gray, 130, 255, cv2.THRESH_BINARY)
    
    micePosition = FindMicePosition(frame_bw)
    
    cap.release()
    Main(video_path, micePosition)