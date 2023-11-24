import cv2
import numpy as np
import time

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

def outlineRed(frame):
    # return frame
    frame = cv2.GaussianBlur(frame, (175, 175), cv2.BORDER_DEFAULT)
    frame = cv2.GaussianBlur(frame, (175, 175), cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # make this a little looser red bound
    lower_red = np.array([0, 120, 120])
    upper_red = np.array([5, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 100
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        biggest_box = [0, 0, 0, 0]
        if w > 400 and h > 400:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if (w * h > biggest_box[2] * biggest_box[3]):
                biggest_box = [x, y, w, h]
    return frame


while True:
    is_successful, frame = capture.read()

    if not is_successful:
        print("Error: Could not read frame.")
        break

    frame = outlineRed(frame)
    cv2.imshow('Altered', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)

capture.release()
cv2.destroyAllWindows()