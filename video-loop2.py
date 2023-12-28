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



def outlineHough(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # dialations
    kernel = np.ones((15, 15), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        frame = get_distance_from_camera_hough(frame, cnt)
    return frame

def get_distance_from_camera_edge(frame, cnt, image_size, real_width, real_height):
    x, y, w, h = cv2.boundingRect(cnt)
    cnt_area = cv2.contourArea(cnt)
    distance = (((real_width + real_height) / 2) * image_size) / (cnt_area * focal_length) # make this take width and height into account (this is really sketch as is)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    distance_text = f"Distance: {distance:.2f} cm"
    cv2.putText(frame, distance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print(distance_text)
    return frame

def get_distance_from_camera_hough(frame, cnt):
    mask = np.zeros_like(frame)
    cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_mask, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame

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


# def outlineRect(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     # dialations
#     kernel = np.ones((15, 15), np.uint8)
#     edges = cv2.dilate(edges, kernel, iterations=1)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
#     # aspect_ratio_threshold = 1.0 # looking at square contours - NEED TO FIGURE OUT HOW TO SEGMENT OUT THE NON-RECTANGULAR ONES
#     # contours = [cnt for cnt in contours if aspect_ratio_threshold > (cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3]) > 1 / aspect_ratio_threshold]
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         mask = np.zeros_like(edges)
#         cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
#         result_frame = cv2.bitwise_and(frame, frame, mask=mask)
#         # result_frame = get_distance_from_camera_edge(result_frame, largest_contour, frame_width, object_real_width, object_real_height)

#         # result_frame = get_distance_from_camera_rectangular(result_frame, largest_contour, frame_width, object_real_width, object_real_height)
#         # print("CONTOUR AREA: " + str(cnt_area))
#         return result_frame
#     return edges


def outlineRect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((15, 15), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    for cnt in contours:
        # Process each contour here. For example:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # If you want to calculate and display the distance for each contour, you can call the distance function here.
        # frame = get_distance_from_camera_rectangular(frame, cnt, frame_width, object_real_width, object_real_height)

    return frame

while True:
    is_successful, frame = capture.read()

    if not is_successful:
        print("Error: Could not read frame.")
        break

    # frame = outlineHough(frame)
    frame = outlineRect(frame)
    cv2.imshow('Altered', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)

capture.release()
cv2.destroyAllWindows()
