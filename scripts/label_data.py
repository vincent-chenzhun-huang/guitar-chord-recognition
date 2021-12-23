import cv2
import json
import os
from pathlib import PurePath, Path
from config.settings import DEFAULT_IMAGES_PATH, DEFAULT_LABELS_PATH


def label_data(file_name, input_path=DEFAULT_IMAGES_PATH, output_path=DEFAULT_LABELS_PATH):
    image = cv2.imread(str(PurePath(f'{str(input_path)}/{file_name}.jpg'))) # file_name does not have the jpg suffix
    w = len(image[0])
    h = len(image)

    cv2.namedWindow(file_name)
    finger_pos = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            finger_dict = {
                'x': x / w,
                'y': y / h,
            }
            finger_pos.append(finger_dict)
            print(f'position: ({x}, {y})')

    while True:
        cv2.imshow(file_name, image)
        cv2.setMouseCallback(file_name, onMouse)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            break

        if key == ord('s'):
            # if the image is not good enough, skip it and not create a label
            return
    finger_label = {
        'fingers': finger_pos
    }

    with open(Path(f'{str(output_path)}/{file_name}.json'), 'w+') as label_json:
        print('------------wrote to json--------------')
        json.dump(finger_label, label_json, indent=4)

    cv2.destroyAllWindows()

    return finger_pos


def label_images(data_path=DEFAULT_IMAGES_PATH, output_path=DEFAULT_LABELS_PATH, start_index=1):
    if os.listdir(data_path):
        max_file_number = sorted([int(f.split('.')[-2]) for f in os.listdir(data_path) if
                                  os.path.isfile(os.path.join(data_path, f))])[-1]
    else:
        raise IOError('The directory is empty')

    for file_number in range(start_index, max_file_number + 1):
        label_data(f'{file_number}', data_path, output_path)


if __name__ == '__main__':
    label_images(start_index=72)
