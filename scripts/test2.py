import cv2


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(1)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def check_data():
    image = cv2.imread('/data/test/image9.jpg')
    hand = [(3757, 2283), (4696, 2802)]  # in the format of (x, y)
    labelled = cv2.rectangle(image, hand[0], hand[1], (0, 0, 0), 2)

    hand_img = image[hand[0][1]: hand[1][1], hand[0][0]: hand[1][0]]

    fingers = [(574,142), (270,163), (127,171)]

    for finger in fingers:
        hand_img = cv2.circle(hand_img, finger, 0, (0, 0, 0), 2)
    cv2.imshow('hand', hand_img)
    cv2.waitKey(0)


def main():
    check_data()
    # show_webcam()
if __name__ == '__main__':
    main()
