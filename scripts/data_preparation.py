import cv2


def crop_hand(original_image_data, hand, margin=0.1, img_size=(300, 300, 3), output_path='dataset/hand'):
    """
    Algorithm to crop out the fingertip
        - The positions are normalized in proportion to the image feed dimension.
        - Extract the rectangular region over the hand
        - Then apply a specified margin m
        - Update the finger coordinates according to the relative position in the rectangle
        - Crop out the image
        - Resize the new image to form a square image

    Function to produce a new image with the hand on it
    :param original_image_data: the original 2d BGR image to be processed
    :param hand: a list with the positions of the fingers
    :param margin: a margin to be added to the image
    :param img_size: The size of the image to be returned
    :param output_path: The path to the folder storing the labels, default to dataset/hand
    """
    image_data = cv2.cvtColor(original_image_data, cv2.COLOR_BGR2GRAY)
    x_rec_left = min(hand, key=lambda pos: pos['x'])['x']
    x_rec_right = max(hand, key=lambda pos: pos['x'])['x']

    y_rec_up = min(hand, key=lambda pos: pos['y'])['y']
    y_rec_down = max(hand, key=lambda pos: pos['y'])['y']

    width = x_rec_right - x_rec_left
    height = y_rec_down - y_rec_up

    '''
        A ----------- B
        |             |
        |             |
        C ----------- D
    '''

    a_x = max(x_rec_left - margin * width, 0)
    a_y = max(y_rec_up - margin * height, 0)
    d_x = min(x_rec_right + margin * width, 1)
    d_y = min(y_rec_down + margin * height, 1)

    true_width = len(image_data[0])
    true_height = len(image_data)

    cropped_image = original_image_data[int(a_y * true_height): int(d_y * true_height) + 1,
                                        int(a_x * true_width):int(d_x * true_width) + 1,
                                        :]

    # extract compute new locations of fingertips and save to json file

    finger_points = [{'x': (point['x'] - a_x) / (width * (1 + 2 * margin)),
                      'y': (point['y'] - a_y) / (height * (1 + 2 * margin))}
                     for point in hand]

    label = {'fingers': finger_points}

    return cropped_image, label
