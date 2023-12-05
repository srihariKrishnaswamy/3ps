import cv2
import numpy as np
import time

focal_length = 5 # my mac has a 50mm focal length
capture = cv2.VideoCapture(0)
frame_height, frame_width = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
object_real_width = 10 #cm
object_real_height = 10 #cm

if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

def get_distance_from_camera_rectangular(frame, cnt, image_size, real_width, real_height):
    x, y, w, h = cv2.boundingRect(cnt)
    apparent_width = w
    apparent_height = h
    average_apparent_size = (apparent_width + apparent_height) / 2.0
    distance = (((real_width + real_height) / 2) * image_size) / (average_apparent_size * focal_length) # make this take width and height into account (this is really sketch as is)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    distance_text = f"Distance: {distance:.2f} cm"
    cv2.putText(frame, distance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print(distance_text)
    return frame

def outlineRect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # dialations
    kernel = np.ones((15, 15), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    # aspect_ratio_threshold = 1.0 # looking at square contours - NEED TO FIGURE OUT HOW TO SEGMENT OUT THE NON-RECTANGULAR ONES
    # contours = [cnt for cnt in contours if aspect_ratio_threshold > (cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3]) > 1 / aspect_ratio_threshold]
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(edges)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        result_frame = cv2.bitwise_and(frame, frame, mask=mask)
        cnt_area = cv2.contourArea(largest_contour)
        result_frame = get_distance_from_camera_rectangular(result_frame, largest_contour, frame_width, object_real_width, object_real_height)
        # print("CONTOUR AREA: " + str(cnt_area))
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