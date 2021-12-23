# show camera feed
import numpy as np
import cv2 as cv
from collections import Counter
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def open_camera(capture_rate=50):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
    else:
        while True:
            ret, frame = cap.read()

            if not ret:
                print('Cannot receive frame (stream end?), exiting...')
                break
            # rotated_image = rotate_image(frame, top_orientation_in_degrees)
            # annotated_with_hands = get_hand_rectangle(cv2.flip(rotated_image, 1))
            annotated_with_hands = single_frame_operation(frame)
            current_frame = cv.resize(frame, (300, 300)) if not np.any(annotated_with_hands) else annotated_with_hands
            cv.imshow('frame', current_frame)
            if cv.waitKey(1) == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def single_image(image_path):
    img = cv.imread(image_path)
    annotated_image = single_frame_operation(img)
    cv.imshow('lines', annotated_image)
    cv.waitKey(0)


def single_frame_operation(img, mirror=False, output_size=300):
    top_orientation = get_string_orientation(img)
    top_orientation_in_degrees = top_orientation * (180.0 / 3.141592653589793238463)
    rotated_image = rotate_image(img, top_orientation_in_degrees)

    # image_to_be_annotated = rotated_image if mirror else cv2.flip(rotated_image, 1)
    result = get_hand_rectangle(rotated_image, mirror=mirror)
    annotated_with_hands, rectangle = result[0], result[1]
    if rectangle:
        rec_p1 = (rectangle['top left']['x'], rectangle['top left']['y'])
        rec_p2 = (rectangle['bottom right']['x'], rectangle['bottom right']['y'])
        cv2.rectangle(annotated_with_hands, rec_p1, rec_p2, (0, 244, 0))
        hand_rectangle = annotated_with_hands[rec_p1[1]: rec_p2[1], rec_p1[0]: rec_p2[0]]
        hand_rectangle = pad_to_square(hand_rectangle)
        hand_rectangle = cv2.resize(hand_rectangle, (output_size, output_size), interpolation=cv2.INTER_AREA)
        print(hand_rectangle.shape)
    else:
        hand_rectangle = np.zeros(img.shape)

    return hand_rectangle


def get_string_orientation(frame):
    edges = cv.Canny(frame, 100, 150, None, 3)

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    cdst = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # for each line, get their orientation
    orientation = np.arctan(np.float64(lines[:, :, 3] - lines[:, :, 1]) / np.float64(lines[:, :, 2] - lines[:, :, 0]))

    orientation = orientation.reshape(orientation.shape[0], )

    values, bins, patches = plt.hist(orientation, density=True, bins=60)

    # get the top two orientations
    indices = np.argsort(values)[::-1]
    top_orientation = (bins[indices[0]], bins[indices[0] + 1])

    # get the lines with orientations in such intervals
    interval1 = within_bin(orientation, top_orientation)
    line_filter = list(interval1)
    top_lines = lines[line_filter]
    top_orientations = orientation[line_filter]

    # hist, bins = np.histogram(orientation, bins=20)
    if top_lines is not None:
        for i in range(0, len(top_lines)):
            l = top_lines[i][0]
            cv.line(cdst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    # get the average value of the top orientation
    return np.mean(top_orientations)
    # plt.show()
    # return cv.flip(frame, 1)


def within_bin(orientation, or_bin):
    upper_bound = orientation < or_bin[1]
    lower_bound = orientation > or_bin[0]
    return upper_bound * lower_bound


def get_hand_rectangle(image, mirror=False):
    with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=1,
      min_detection_confidence=0.4) as hands:
        # Convert the BGR image to RGB before processing.
        if not mirror:
            image = cv.flip(image, 1)
        # google mediapipe wants the frame mirrored
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_height, image_width, _ = image.shape
        # annotated_image = image.copy()
        if not results.multi_hand_landmarks:
            return (image, None) if mirror else (cv.flip(image, 1), None)
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # print('hand_landmarks:', hand_landmarks)
            hand_handedness_dict = MessageToDict(results.multi_handedness[i])
            hand_handedness = hand_handedness_dict['classification'][0]['label']
            hand_rectangle = None
            if hand_handedness == 'Left':
                print('got a left hand')
                dict_obj = MessageToDict(hand_landmarks)
                h_l = dict_obj['landmark']
                hand_rectangle = crop_hand(image, h_l)
                # mp_drawing.draw_landmarks(
                #     image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style())

        # cv.imshow('image', cv2.flip(image, 1))
        # cv2.waitKey(0)
        return (image, hand_rectangle) if mirror else (cv.flip(image, 1), hand_rectangle)


def crop_hand(image_data, hand, margin=0.2):
    # cv.imshow('test_image', image_data)
    # cv.waitKey(0)
    img_width, img_height = image_data.shape[1], image_data.shape[0]
    x_rec_right = round(max(hand, key=lambda pos: pos['x'])['x'] * img_width)
    y_rec_up = round(min(hand, key=lambda pos: pos['y'])['y'] * img_height)
    y_rec_down = round(max(hand, key=lambda pos: pos['y'])['y'] * img_height)

    box_width = x_rec_right
    box_height = y_rec_down - y_rec_up

    # return mirrored output
    return {
        'bottom right': {
            'x': img_width - 1 - 0,
            'y': max(y_rec_down + round(box_height * margin), box_height - 1)
        },
        'top left': {
            'x': img_width - 1 - x_rec_right - round(img_width * margin),
            'y': max(y_rec_up - round(box_height * margin * 2), 0)
        }
    }


def pad_to_square(cropped_img):
    # it is assumed that the width is
    h, w = cropped_img.shape[0], cropped_img.shape[1]
    color = [0, 0, 0]
    if h > w:
        # if height > width, pad width
        delta_w = h - w
        left, right = delta_w // 2, delta_w - delta_w // 2
        new_im = cv2.copyMakeBorder(cropped_img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        delta_h = w - h
        top, bottom = delta_h // 2, delta_h - delta_h // 2
        new_im = cv2.copyMakeBorder(cropped_img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)
    return new_im


if __name__ == '__main__':
    # open_camera()
    single_image('/Users/vincenthuang/Development/guitar-chord-recognition/dataset/test/image1.jpg')
