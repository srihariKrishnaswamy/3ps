import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture(0)

num = 0
chessboardSize = (9, 6)
frameSize = (1080, 1920)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm 
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
num = 0
while cap.isOpened():
    succes, img = cap.read()
    k = cv.waitKey(5)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # print(img.shape)
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    if ret == True:
        print("FOUND corners")
        cv.imwrite('images/img' + str(num) + '.png', img)
        num += 1
    else:
        print("did not find corners")
    cv.imshow('Img',img)
    time.sleep(0.1)
cap.release()
cv.destroyAllWindows()