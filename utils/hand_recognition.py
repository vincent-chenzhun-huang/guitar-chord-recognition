import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
from scripts.data_preparation import crop_hand
from pathlib import PurePath, Path
from config.settings import DEFAULT_DATASET_PATH, DEFAULT_IMAGES_PATH, DEFAULT_LABELS_PATH
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def capture_hand(cam_num=0, min_detection_confidence=0.7, min_tracking_confidence=0.7, capture_rate=50,
                 output_path=DEFAULT_DATASET_PATH, max_files=200, start_over=False):
    cap = cv2.VideoCapture(cam_num)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(
      min_detection_confidence=min_detection_confidence,
      min_tracking_confidence=min_tracking_confidence) as hands:
        count = 1
        file_count = determine_next_file_name(PurePath(f'{str(output_path)}/images')) + 1 if not start_over else 0
        capture = True
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    dict_obj = MessageToDict(hand_landmarks)
                    h_l = dict_obj['landmark']
                    # we are interested in the position 8 12 16 20
                    candidate_fingertips = [h_l[i] for i in [8, 12, 16, 20]]
                    print(f'index finger: {candidate_fingertips[0]} '
                          f'middle finger: {candidate_fingertips[1]} '
                          f'ring finger: {candidate_fingertips[2]} '
                          f'pinky: {candidate_fingertips[3]} ')
                    # mp_drawing.draw_landmarks(
                    #     image,
                    #     hand_landmarks,
                    #     mp_hands.HAND_CONNECTIONS,
                    #     mp_drawing_styles.get_default_hand_landmarks_style(),
                    #     mp_drawing_styles.get_default_hand_connections_style())
                    if count == 0 and capture:
                        print(f'-------------------writing to disk {file_count}.jpg----------------------')
                        # save the frame to the destination
                        hand_img, label = crop_hand(image, h_l, margin=0.4)
                        print(f"Pure Path: --------- {str(PurePath(f'{str(output_path)}/images/{file_count}.jpg'))} ------------")
                        cv2.imwrite(str(PurePath(f'{str(output_path)}/images/{file_count}.jpg')), hand_img)
                        cv2.imwrite(str(PurePath(f'{str(output_path)}/original_images/{file_count}.jpg')), image)

                        file_count += 1
                        if file_count == max_files:
                            capture = False
                    count = (count + 1) % capture_rate

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


def determine_next_file_name(dir_path):

    if os.listdir(dir_path):
        max_file_number = sorted([int(f.split('.')[-2]) for f in os.listdir(dir_path) if
                                  os.path.isfile(os.path.join(dir_path, f))])[-1]
    else:
        max_file_number = 0

    return max_file_number


if __name__ == '__main__':
    capture_hand()
