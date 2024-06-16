import cv2
import numpy as np
from picamera2 import Picamera2
from time import sleep

def undistort_image(image):
    # Camera matrix and distortion coefficients (example values, replace with actual calibration data)
    camera_matrix = np.array([[550, 0, 640],
                            [0, 550, 360],
                            [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.065, 0.0, -0.0, 0.0, 0], dtype=np.float32)  # k1, k2, p1, p2

    # Get the optimal new camera matrix
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop the image based on the valid region of interest (roi)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    return undistorted_image

def perspective_image(image, n):
    h,w = image.shape[:2]
    pts1 = np.float32([[w/3,h/3],[w/3,h/3*2],[w/3*2,h/3],[w/3*2,h/3*2]])
    pts2 = np.float32([[w/3,h/3],[w/3-n,h/3*2],[w/3*2,h/3],[w/3*2+n,h/3*2]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    _image = cv2.warpPerspective(image, M, (w,h))

    return perspective_image


def extract_parking_area(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    if w < 500 or h < 500:
        return image
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image


def find_parking_lot_cells(image):
    h, w = image.shape[:2]
    cell_coords = [[w/13*12, h/9*8], [w/13*12, h/9*6], [w/13*12, h/9*3], [w/13*10, h/9*1],
                   [w/13*8, h/9*1], [w/13*5, h/9*1], [w/13*3, h/9*1], [w/13*1, h/9*3],
                   [w/13*1, h/9*6], [w/13*1, h/9*8], [w/13*5, h/9*8], [w/13*5, h/9*6],
                   [w/13*5, h/9*4], [w/13*8, h/9*4], [w/13*8, h/9*6], [w/13*8, h/9*8]]
    for i in range(0, 16):
        for j in range(0, 2):
            cell_coords[i][j] = round(cell_coords[i][j])
    return cell_coords


def process_parking_lot(image, parking_centers):
    parking_status = []
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])
    for center in parking_centers:
        center_x, center_y = center
        parking_space = image[center_y-20:center_y+20, center_x-20:center_x+20]
        avg_color_per_row = np.average(parking_space, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        avg_color_hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]
        if lower_green[0] <= avg_color_hsv[0] <= upper_green[0] and \
           lower_green[1] <= avg_color_hsv[1] <= upper_green[1] and \
           lower_green[2] <= avg_color_hsv[2] <= upper_green[2]:
            parking_status.append("Empty")
        else:
            parking_status.append("Full")
    return parking_status


def display_parking_lot(parking_lot_image, parking_status, parking_spaces_on_image):
    parking_lot_display = parking_lot_image.copy()
    for i, status in enumerate(parking_status):
        color = (0, 0, 255) if status != "Empty" else (235, 206, 135) if i in [2, 4, 5, 7] else (0, 255, 0)
        cv2.rectangle(parking_lot_display, parking_spaces_on_image[i][0], parking_spaces_on_image[i][1], color, -1)
    empty_spaces = parking_status.count("Empty")
    cv2.putText(parking_lot_display, f"Empty spaces:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(parking_lot_display, f"{empty_spaces}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.imshow('Parking Lot Model', parking_lot_display)
    sleep(0.1)

if __name__ == "__main__":
    # Define the coordinates of parking spaces on the reference image
    parking_spaces_on_image = [
        [(670, 390), (794, 474)],  # 1
        [(670, 300), (794, 384)],  # 2
        [(670, 135), (794, 229)],  # 3
        [(580, 5), (664, 129)],    # 4
        [(480, 5), (574, 129)],    # 5
        [(225, 5), (319, 129)],    # 6
        [(135, 5), (219, 129)],    # 7
        [(5, 135), (129, 229)],    # 8
        [(5, 300), (129, 384)],    # 9
        [(5, 390), (129, 474)],    # 10
        [(272, 390), (396, 474)],  # 11
        [(272, 300), (396, 384)],  # 12
        [(272, 210), (396, 294)],  # 13
        [(402, 210), (526, 294)],  # 14
        [(402, 300), (526, 384)],  # 15
        [(402, 390), (526, 474)]   # 16
    ]

    # Initialize and configure the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1280, 720)}))
    picam2.start()
    
    # Allow the camera to warm up
    sleep(2)

    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()
        frame = cv2.flip(frame,-1)

        perspective_frame=perspective_image(frame, 10)

        # Undistort the captured frame
        undistorted_frame = undistort_image(perspective_frame)

        # Extract the parking area from the undistorted frame
        parking_area_frame = extract_parking_area(undistorted_frame)
        cv2.imshow('cam', parking_area_frame)
        
        # Find the centers of the parking cells
        parking_centers = find_parking_lot_cells(parking_area_frame)

        # Process the parking lot to determine the status of each space
        parking_status = process_parking_lot(parking_area_frame, parking_centers)

        # Load a reference image of the parking lot
        parking_lot_image = cv2.imread("parking_lot_image.jpg")

        # Display the parking lot with the status of each space
        display_parking_lot(parking_lot_image, parking_status, parking_spaces_on_image)

        # Exit the loop if the 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Clean up and close windows
    cv2.destroyAllWindows()
    picam2.stop()