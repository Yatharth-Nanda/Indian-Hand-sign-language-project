
import csv
import copy
import argparse
import tkinter as tk
import itertools
import numpy as np
from collections import Counter
from collections import deque


import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



import os
import cv2

from model.keypoint_classifier import KeyPointClassifier

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode



# variable for folder name
folder_name = ""

def get_screen_dimensions():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()  # Close the Tkinter window
    return screen_width // 2, screen_height  # Return half width and full height

def logging_csv(character, preprocess_landmark_list):
    csv_path = (r"D:\yatharth files\SEM 3-2 2024\isl_sop_codes\final_sop_19thfeb\keypoint"
                r".csv")
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        formatted_landmark_list = [format(x, '.16f') for x in preprocess_landmark_list]
        writer.writerow([character] + formatted_landmark_list)

    return


def pre_process_landmark(both_hand_landmarks):
    length_landmarks = len(both_hand_landmarks)
    if length_landmarks == 0 or length_landmarks > 2 or both_hand_landmarks is None:
        #print("ERROR ERRROR ERROR ERROR ERROR ERRROR ERRORER ER WEREWOREREOA ER OAER OEWA OAEWRO EW E EWO ")
        return None
    elif (length_landmarks == 1):  # process only one set of landmarks wrt its wrist , keep the other as zero

        first_key = next(iter(both_hand_landmarks))
        temp_landmark_list = copy.deepcopy(both_hand_landmarks[first_key])

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(
                temp_landmark_list):  # indexing the landmarks and getting the values at the same time
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization

        max_value = max(list(map(abs, temp_landmark_list)))
        # print("max value is "+str(max_value))
        # print("base_x:", base_x)
        # print("base_y:", base_y)

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        temp_zeros = [0] * 42

        # where to add 42 zeros to the flattened array
        '''I want to keep the order of my hands to be left than right so I will be creating my array accordingly'''
        if (first_key == "Left"):
            temp_landmark_list.extend(temp_zeros)
        else:
            temp_landmark_list = temp_zeros + temp_landmark_list

        # print("processed landmark list is ")
        # print(temp_landmark_list)
        return temp_landmark_list

    elif (length_landmarks == 2):

        temp_landmark_list_left = copy.deepcopy(both_hand_landmarks["Left"])
        temp_landmark_list_right = copy.deepcopy(both_hand_landmarks["Right"])
        temp_landmark_list = temp_landmark_list_left + temp_landmark_list_right
        # Taking the average of the base points
        base_x = (temp_landmark_list_left[0][0] + temp_landmark_list_right[0][0]) / 2.0
        base_y = (temp_landmark_list_left[0][1] + temp_landmark_list_right[0][1]) / 2.0

        for landmark in temp_landmark_list:
            landmark[0] -= base_x
            landmark[1] -= base_y

        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))
        # print("max value is " + str(max_value))
        # print("base_x:", base_x)
        # print("base_y:", base_y)

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        # print("processed landmark list is ")
        # print(temp_landmark_list)
        return temp_landmark_list


def extract_label(string_representation):
    # Split the string by comma and space characters
    parts = string_representation.split(", ")

    # Initialize variable to store category name
    category_name = None

    # Iterate over each part
    for part in parts:
        # Find the part that contains 'category_name'
        if 'category_name' in part:
            # Extract the category name
            category_name = part.split("='")[1].rstrip("')]")

    return category_name


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def calc_landmark_list(image_local, hand_landmarks_local):
    image_width, image_height = image_local.shape[1], image_local.shape[0]
    landmark_point = []

    # Keypoint
    for landmark in (hand_landmarks_local):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


if __name__ == "__main__":


    # STEP 2: Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, min_hand_detection_confidence=0.5,running_mode=VisionRunningMode.IMAGE,
                                           min_hand_presence_confidence=0.5 , min_tracking_confidence=0.5,
                                           num_hands=2)
    hands = vision.HandLandmarker.create_from_options(options)

    cap_device = 0  # Camera device index (e.g., 0 for the first camera)
    cap = cv.VideoCapture(cap_device)

    # Get camera resolution
    camera_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    camera_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Set video capture object to capture full-length frame
    cap.set(cv.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, camera_height)

    keypoint_classifier = KeyPointClassifier()

    cap = cv.VideoCapture(0)

    while True:
        # Capture frame from camera
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        ret, frame = cap.read()
        if not ret:
            break

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        debug_image=frame

        results = hands.detect(image)
        both_hand_landmarks = {}
        if results.hand_landmarks is not None:

            for hand_landmarks, which_hand in zip(results.hand_landmarks,
                                                  results.handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                both_hand_landmarks[extract_label(str(which_hand))] = landmark_list

            preprocessed_landmarks = pre_process_landmark(both_hand_landmarks)

            if (preprocessed_landmarks is not None) :
            #call the keypoint classifier
                input_data_reshaped = np.reshape(preprocessed_landmarks,
                                             (1, -1))  # Reshape to a row vector with one row and multiple columns

                hand_sign_id = keypoint_classifier(input_data_reshaped)

            # Draw rectangle at the bottom of the frame
                cv.rectangle(frame, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                #print(hand_sign_id)

                #print(prediction)
            # Print gesture label in the rectangle
                cv.putText(frame, chr(hand_sign_id+65), (10, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display frame
        cv.imshow('Hand Gesture Recognition', frame)



    # Release resources
    cap.release()
    cv.destroyAllWindows()







