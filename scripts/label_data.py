import cv2
import json
import os
from pathlib import PurePath, Path
from config.settings import DEFAULT_DATASET_PATH

FINGERS = 4
FRETS = 8
STRINGS = 6

LABEL_NAMES = {
    FINGERS: 'fingers',
    FRETS: 'frets',
    STRINGS: 'strings'
}


def label_data(file_name, input_path=DEFAULT_DATASET_PATH, output_path=DEFAULT_DATASET_PATH, maximum_points=FINGERS):
    print(f'open image: {str(input_path)}/{file_name}.jpg')
    image = cv2.imread(str(PurePath(f'{str(input_path)}/{file_name}.jpg')))  # file_name does not have the jpg suffix
    w = len(image[0])
    h = len(image)

    cv2.namedWindow(file_name)
    points = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            finger_dict = {
                'x': x,
                'y': y,
            }
            if len(points) == maximum_points:
                print(f'the maximum is {maximum_points} points, you cannot add anything any more')
            else:
                points.append(finger_dict)
                print(f'position: ({x}, {y})')

    # for i in range(maximum_points):
    while True:
        cv2.imshow(file_name, image)
        cv2.setMouseCallback(file_name, onMouse)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('v'):
            # the labelling goes in order of index, middle, ring, finger then pinky, if a finger
            # is not pressed on the fretboard, press v, we give the coordinate a -1, -1
            points.append({
                'x': -1.000,
                'y': -1.000
            })
            print(f'position: (-1, -1)')

        if key == ord('b'):
            remaining_pts_count = maximum_points - len(points)
            if remaining_pts_count > 0:
                for j in range(remaining_pts_count):
                    points.append({
                        'x': -1.000,
                        'y': -1.000
                    })
            break

        if key == ord('s'):
            # if the image is not good enough, skip it and not create a label
            return

    # finger_label = {
    #     label_names[maximum_points]: points
    # }

    output_file_name = f'{file_name}_{LABEL_NAMES[maximum_points]}'

    with open(Path(f'{str(output_path)}/{output_file_name}.csv'), 'w+') as label_csv:
        print('------------write to csv--------------')
        # json.dump(finger_label, label_csv, indent=4)
        for point in points:
            label_csv.write(f'{point["x"]},{point["y"]}\n')

    cv2.destroyAllWindows()

    return points


def label_images(data_path=DEFAULT_DATASET_PATH, output_path=DEFAULT_DATASET_PATH, start_index=206, label_type=FINGERS):
    # if os.listdir(data_path):
    #     max_file_number = sorted([int(f.split('.')[-2]) for f in os.listdir(data_path) if
    #                               os.path.isfile(os.path.join(data_path, f))])[-1]
    # else:
    #     raise IOError('The directory is empty')

    # for file_number in range(start_index, max_file_number + 1):
    #     label_data(f'{file_number}', data_path, output_path)
    filename = f'image{start_index}'
    file_path = f'{data_path}/{filename}.jpg'
    if os.path.isfile(file_path):
        print(f'Start labelling {LABEL_NAMES[label_type]}')
        while True:
            label_data(filename, data_path, output_path, maximum_points=label_type)
            start_index += 1
            filename = f'image{start_index}'
            file_path = f'{data_path}/{filename}.jpg'
            if not os.path.isfile(file_path):
                break


if __name__ == '__main__':
    label_images(start_index=212, label_type=STRINGS)
