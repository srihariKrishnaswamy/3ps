import cv2
import numpy as np
import time

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

REAL_AREA = 20 # will be like square cm or smth

def outlineRect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # dialations
    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        print("CONTOUR AREA: " + str(cv2.contourArea(largest_contour)))
        mask = np.zeros_like(edges)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

        result_frame = cv2.bitwise_and(frame, frame, mask=mask)

        return result_frame

    return edges


while True:
    is_successful, frame = capture.read()

    if not is_successful:
        print("Error: Could not read frame.")
        break

    frame = outlineRect(frame)
    cv2.imshow('Altered', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)

capture.release()
cv2.destroyAllWindows()