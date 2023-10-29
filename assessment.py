import cv2
import numpy as np


class Assessment:
    @staticmethod
    def building_damage_percentage(image1, image2):
        img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        diff = cv2.absdiff(img1, img2)
        percentage = np.mean(diff) / 255 * 100

        return percentage

class BuildingDamageAssessor:
    def calculate_percentage(self, image1_path, image2_path):
        percentage = Assessment.building_damage_percentage(image1_path, image2_path)
        rounded_percentage = round(percentage, 2)
        return rounded_percentage
