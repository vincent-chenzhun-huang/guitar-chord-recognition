import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
from scripts.data_preparation import crop_hand

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

if __name__ == '__main__':

    # For static images:
    IMAGE_FILES = ['/Users/vincenthuang/Development/guitar-chord-recognition/dataset/test/image4.jpg']
    with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )

                hand = MessageToDict(hand_landmarks)['landmark']
                new_img, label = crop_hand(image, hand)
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                cv2.imshow('image', new_img)
                cv2.imshow('original', annotated_image)
                print(f'labels: {label}')
                cv2.waitKey(0)

            cv2.imwrite(
                str(idx) + '.png', cv2.flip(annotated_image, 1))
