import numpy as np
import cv2
import matplotlib.pyplot as plt

def labelize_image(image:np.array, label:np.array) -> np.array:
    """Labelize image (put the Mitral Valve detected fron label in white)

    Args:
        image (np.array): image
        label (np.array): label

    Returns:
        np.array: labelized image
    """
    labelized_image = label.copy()
    # label = label.astype(np.uint8)  #convert to an unsigned byte
    # label*=255
    # labelized_image[label == 255] = 255
    cv2.imshow("labelized", labelized_image)
    cv2.waitKey(0)
    return 


def visualize(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i].reshape(128,128))
        plt.axis('off')
    plt.show()