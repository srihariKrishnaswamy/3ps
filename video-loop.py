import cv2
import numpy as np
import time

# INTRINSIC CAMERA PARAMS (OBTAIN VIA RUNNING CAMERA CALIBRATION)
fx = 2.44171380e+03
fy = 2.39069151e+03
cx = 6.30504437e+02
cy = 5.70956578e+02
dist = [-1.38149680e-01, 3.57171479e+00 , 1.19686067e-02 , -6.03857380e-02 , -1.40045591e+01]
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

focal_length_mm = 50 # my mac has a 50mm focal length
side_length = .12
capture = cv2.VideoCapture(0)
frame_height, frame_width = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
object_real_width = .10 #10 cm
object_real_height = .10 #10 cm

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
            if corners is not None: 
                # print(max_area, corners)
                dilated, pitch, roll, yaw = get_pry(dilated, corners)
                x, y, z = get_xyz_disp(pitch, roll, yaw, corners, camera_matrix, dist)
            return dilated
            # now get the houghlines on this and get pitch roll and yaw & then x y & z distance
    return blank_frame

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

def get_pry(frame, corners, focal_length_mm=focal_length_mm, fx=fx, fy=fy, cx=cx, cy=cy, square_side_length=side_length):
    object_points = np.array([
        [-square_side_length / 2, -square_side_length / 2, 0],
        [square_side_length / 2, -square_side_length / 2, 0],
        [square_side_length / 2, square_side_length / 2, 0],
        [-square_side_length / 2, square_side_length / 2, 0]
    ], dtype=np.float32)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    image_coordinates = np.array(corners, dtype=np.float32)
    undistorted_image_coordinates = cv2.undistortPoints(
        image_coordinates.reshape(-1, 1, 2),
        K, np.zeros(4)
    )
    print("undist: ", undistorted_image_coordinates)
    _, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, undistorted_image_coordinates, K, np.zeros(4)
    )
    print("translation mat: ", translation_vector)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
    pitch, roll, yaw = np.degrees(euler_angles)
    print("Pitch:", pitch)
    print("Roll:", roll)
    print("Yaw:", yaw)
    #### GETTING CENTER OF SQUARE #####
    frame = draw_axes(frame, corners, pitch, roll, yaw)
    return frame, pitch, roll, yaw

def get_xyz_disp(pitch, roll, yaw, corners, mtx, dist, real_world_side_length=object_real_width):
    corners_array = np.array(corners, dtype=np.float32)
    corners_array = corners_array.reshape(-1, 1, 2)
    undistorted_corners = cv2.undistortPoints(corners_array, mtx, np.array(dist))
    undistorted_corners = undistorted_corners.reshape((4, 2))
    square_corners_3d = np.hstack((undistorted_corners, np.zeros((4, 1))))
    square_corners_3d *= real_world_side_length
    rotation_matrix = cv2.Rodrigues(np.array([pitch, roll, yaw]))[0]
    # Apply the rotation matrix to the 3D coordinates
    rotated_square_corners_3d = np.dot(rotation_matrix, square_corners_3d.T).T
    x_displacement, y_displacement, z_displacement = np.mean(rotated_square_corners_3d, axis=0)
    print(x_displacement, y_displacement, z_displacement)
    return x_displacement, y_displacement, z_displacement

def rotation_matrix_to_euler_angles(rotation_matrix): # bruh no clue what's going on here
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def draw_axes(frame, corners, pitch, roll, yaw, scale=100):
    # DRAWING CENTER POINT
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[3]
    x4, y4 = corners[2]
    m1 = (y4 - y1) / (x4 - x1)
    b1 = y1 - (m1 * x1)
    m2 = (y3 - y2) / (x3 - x2)
    b2 = y2 - (m2 * x2)
    center_x = int((b2 - b1) / (m1 - m2))
    center_y = int(m1 * center_x + b1)
    center = (center_x, center_y)
    cv2.circle(frame, center, 5, (0, 255, 0), -1)
    print("CENTER: ", center)
    # PRY stuff for axes
    axis_length = scale

    # Calculate the 2D coordinates of the X, Y, and Z axes based on angles
    x_axis_end = (center[0] + axis_length * np.cos(yaw), center[1] + axis_length * np.sin(yaw))
    y_axis_end = (center[0] - axis_length * np.sin(yaw), center[1] + axis_length * 10 * np.cos(yaw))
    z_axis_end = (center[0] + axis_length * np.cos(yaw + np.pi/2), center[1] + axis_length * np.sin(yaw + np.pi/2))

    # Convert the coordinates to integers
    x_axis_end = tuple(np.int32(x_axis_end))
    y_axis_end = tuple(np.int32(y_axis_end))
    z_axis_end = tuple(np.int32(z_axis_end))
    print("AXES ENDS: ", x_axis_end, y_axis_end, z_axis_end)

    # Draw the axes on the image
    cv2.line(frame, center, x_axis_end, (0, 0, 255), 2)  # X-axis (red)
    cv2.line(frame, center, y_axis_end, (0, 255, 0), 2)  # Y-axis (green)
    cv2.line(frame, center, z_axis_end, (255, 0, 0), 2)  # Z-axis (blue)
    return frame

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