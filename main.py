import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from networktables import NetworkTables
import numpy as np

NetworkTables.initialize(server='roborio-4738-frc.local')

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                     max_num_hands=2,
                     min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

buffer = [[0, 0]]
bufferLength = 1000

increm = 0

def isPointerPinch(handLms):
    handLms = list(enumerate(handLms.landmark))
    pointer = np.array([handLms[8][1].x, handLms[8][1].y])
    thumb = np.array([handLms[4][1].x, handLms[4][1].y])
    return np.all(np.abs(np.subtract(pointer, thumb)) < 0.075)

def isMiddlePinch(handLms):
    handLms = list(enumerate(handLms.landmark))
    middle = np.array([handLms[12][1].x, handLms[12][1].y])
    thumb = np.array([handLms[4][1].x, handLms[4][1].y])
    return np.all(np.abs(np.subtract(middle, thumb)) < 0.075)

def getDriveVector(img, handLms):
     # Middle of the screen
    middle = np.array([(img.shape[1]/2)/img.shape[1], (img.shape[0]/2)/img.shape[1]])
    # Middle of the hand
    hand = np.array([handLms.landmark[9].x, handLms.landmark[9].y])
    # Distance between the middle of the screen and the middle of the hand
    distance = np.subtract(hand, middle)
    # Normalize the distance
    distance = np.clip(distance, -1, 1)
    return distance

def getRotation(handLms):
    global increm
    #get the rotation of the hand based off of points 5, 7, and 0
    handLms = list(enumerate(handLms.landmark))
    #get the x and y coordinates of the points
    indexBase = np.array([handLms[7][1].x, handLms[7][1].y])
    pinkyBase = np.array([handLms[17][1].x, handLms[17][1].y])
    wrist = np.array([handLms[0][1].x, handLms[0][1].y])
    #Get the midpoint between the index and pinky base
    midPoint = np.add(indexBase, pinkyBase)/2
    #Get the vector between the wrist and the midpoint
    wristToMid = np.subtract(midPoint, wrist)
    #Get the angle between the wrist to mid vector and the x axis
    angle = np.arctan2(wristToMid[1], wristToMid[0])
    #return the angle
    return angle

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = w-int(lm.x *w), h-int(lm.y*h)
        
        # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        driveVector = getDriveVector(img, results.multi_hand_landmarks[0]) * 2
        cv2.arrowedLine(img, (int(img.shape[1]/2), int(img.shape[0]/2)), (int(img.shape[1]/2 + driveVector[0]*100), int(img.shape[0]/2 + driveVector[1]*100)), (255, 0, 0), 4)

        rotation = getRotation(results.multi_hand_landmarks[0])
        print(rotation)

    img = cv2.flip(img, 1)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

