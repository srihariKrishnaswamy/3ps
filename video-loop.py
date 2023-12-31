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
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 50, 200)
    # dialations
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    res = get_distance_from_camera_hough(edges, frame)
    return res

def get_distance_from_camera_hough(edges, og): # AFTER TESTING: I like probabalistic better so that's what's in there
    height, width, channels = og.shape
    print(height, width)
    blank_frame = np.zeros((height, width, 1), dtype=np.uint8)

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(len(linesP)):
            l = linesP[i][0]
            cv2.line(blank_frame, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)
        contours, _ = cv2.findContours(blank_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_area_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            # get average color of countour and instead of getting max area, if it's red in the og frame then take that as the countour we want
            if area > max_area:
                max_area = area
                max_area_contour = contour
            # avg_color = get_avg_color(contour, og, blank_frame) # this takes a lota tuning, wanna replace the top if block with this
            # if satisfies_threshold(avg_color) and area > max_area:
            #     max_area = area
            #     max_area_contour = contour
        if max_area_contour is not None:
            blank_frame = np.zeros((height, width, 1), dtype=np.uint8)
            cv2.drawContours(blank_frame, [max_area_contour], -1, (255, 255, 0), thickness=2)
            kernel = np.ones((9, 9), np.uint8)
            dilated = cv2.dilate(blank_frame, kernel, iterations=1)
            dilated, corners = get_corners(dilated)
            print(max_area, corners)
            return dilated
            # now get the houghlines on this and get pitch roll and yaw & then x y & z distance
    return blank_frame
    # tmrw: get the edges that make up the largest contour and do the bottom calculation: 
    # distance procedure: we know it's a square so we gotta get all the angles and edge distances for the 'square' in the frame
    # and since its a square we can use these angle and edge distances to get its orientation difference from the camera
    # and then factoring in this orientation difference we can get x y & z distance

def get_avg_color(contour, frame, bw):
    mask = np.zeros_like(bw, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    pixels_inside_contour = cv2.bitwise_and(frame, frame, mask=mask)
    average_color = np.mean(pixels_inside_contour, axis=(0, 1))  
    print("average_color " + str(average_color))
    return average_color

def satisfies_threshold(color):
    low = (0, 0, 50)
    high = (20, 20, 255)
    for i in range(len(color)):
        if color[i] < low[i] or color[i] > high[i]: return False
    return True

def get_corners(edges_frame):
    contours, _ = cv2.findContours(edges_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges_frame = cv2.cvtColor(edges_frame, cv2.COLOR_GRAY2BGR)
    four_sided_contour = None
    four_corners = None
    vertices = 4
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == vertices:
            four_sided_contour = approx
            break
    if four_sided_contour is not None:
        # print("got a 4 sided")
        four_corners = four_sided_contour.reshape(-1, 2)
        for x, y in four_corners:
            cv2.circle(edges_frame, (x, y), 10, (0, 0, 255), -1) 
    return edges_frame, four_corners

while True:
    is_successful, frame = capture.read()
    if not is_successful:
        print("Error: Could not read frame.")
        break
    frame = outlineHough(frame)
    cv2.imshow('Altered', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.1)
capture.release()
cv2.destroyAllWindows()