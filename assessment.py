import cv2
import numpy as np


class Assessment:
    @staticmethod
    def damage_percentage(image1, image2):
        img1 = cv2.imread(image1, 0)
        img2 = cv2.imread(image2, 0)

        diff = cv2.absdiff(img1, img2)
        percentage = np.mean(diff) / 255 * 100

        return percentage
