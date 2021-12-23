import cv2
import json
if __name__ == '__main__':
    image = cv2.imread('../dataset/hand/images/51.jpg')
    with open('../dataset/hand/labels/51.json') as label_file:
        data = json.load(label_file)
    w = len(image[0])
    h = len(image)
    for finger in data['fingers']:
        x = finger['x']
        y = finger['y']
        real_x = round(x * w)
        real_y = round(y * h)

        image = cv2.circle(image, (real_x, real_y), radius=0, color=(255, 255, 255), thickness=2)

    cv2.imshow('image', image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
