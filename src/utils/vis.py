import cv2
import random
import numpy as np

from src.utils.image import read_image


def display_in_window(image, tag="Resizable Image"):
    """
    Display the image in a window.
    :param image: numpy.ndarray，Pictures read through opencv
    :param tag: str, window label
    :return:
    """
    # Display the image and set the window properties to resizable.
    cv2.namedWindow(tag, cv2.WINDOW_NORMAL)
    cv2.imshow(tag, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_detection_results_on_image(image, detection_results):
    """
    Display the results of object huggingface on the picture.
    :param image: image path or image read through opencv
    :param detection_results: The result of object huggingface has a similar format:
    [{'score': confidence(float), 'label': category label(str),
    'box': {'xmin': x coordinate of upper left corner(int),
            'ymin': y coordinate of upper left corner(int),
            'xmax': x coordinate of lower right corner(int),
            'ymax': y coordinate of lower right corner(int)}}, {...}, ...]
    :return: image with huggingface results
    """
    if isinstance(image, str):
        image = read_image(image, mode="bgr")

    # Create a copy of the image to avoid modifying the original image.
    output_image = np.copy(image)

    # Each category is individually assigned a random color.
    random_colors = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for index, label in
                     enumerate(set(item['label'] for item in detection_results))}

    # Iterate through each huggingface result.
    for result in detection_results:
        label = result['label']
        score = result['score']
        box = result['box']

        xmin, ymin = box['xmin'], box['ymin']
        xmax, ymax = box['xmax'], box['ymax']

        # draw a rectangular box.
        cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), random_colors[label], 2)

        # display the label and confidence above the rectangular box.
        label_text = f"{label}: {score:.2f}"
        cv2.putText(output_image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, random_colors[label], 2)

    return output_image
