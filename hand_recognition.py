import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def capture_hand(cam_num=0, min_detection_confidence=0.7, min_tracking_confidence=0.7):
    cap = cv2.VideoCapture(cam_num)
    with mp_hands.Hands(
      min_detection_confidence=min_detection_confidence,
      min_tracking_confidence=min_tracking_confidence) as hands:
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
            image.flags.writeable = True
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
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == '__main__':
    capture_hand()
