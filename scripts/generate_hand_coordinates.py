from config.settings import DEFAULT_DATASET_PATH
import os
import cv2


def add_hand_coordinate():
    all_image_numbers = [file.replace('image', '').replace('.jpg', '') for file in os.listdir(DEFAULT_DATASET_PATH) if
                         file.endswith('.jpg')]
    # add a image_num_hand.csv file

    for num in all_image_numbers:
        image_file_name = f'image{num}.jpg'
        hand_file_name = f'image{num}_hand.csv'
        image_shape = cv2.imread(f'{DEFAULT_DATASET_PATH}/{image_file_name}').shape
        w, h = image_shape[0], image_shape[1]
        if w and h:
            with open(f'{DEFAULT_DATASET_PATH}/{hand_file_name}', 'w+') as hand_csv:
                hand_csv.write(f'0,0\n{w - 1},{h - 1}')
        else:
            raise Exception('no such file')


if __name__ == '__main__':
    add_hand_coordinate()
