import cv2
import os
import numpy as np
from skimage import io
from colormath.color_objects import LabColor
from pyciede2000 import ciede2000
from colormath.color_objects import LabColor, HSVColor
from colormath.color_conversions import convert_color


def hsv_to_lab(color):
    hsv_color = HSVColor(color[0], color[1] / 100, color[2] / 100)
    lab_color = convert_color(hsv_color, LabColor)
    return lab_color.lab_l, lab_color.lab_a, lab_color.lab_b


def color_distance_hsv(image_color, target_color):
    return cv2.norm(image_color, target_color, cv2.NORM_L2)

def color_distance_cie2000(color1, color2):
    color1_lab = (color1[0], color1[1], color1[2])
    color2_lab = (color2[0], color2[1], color2[2])
    delta_e = ciede2000(color1_lab, color2_lab)
    return delta_e

def find_images_by_color(input_folder, output_folder, target_color_hsv, threshold=20):
    target_color_lab = hsv_to_lab(target_color_hsv)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = io.imread(image_path)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            color_distance = color_distance_cie2000(target_color_lab, hsv_to_lab(image_hsv.mean(axis=0).mean(axis=0)))
            if color_distance['delta_E_00'] < threshold:
                print("Найдено изображение", filename)
                output_path = os.path.join(output_folder, filename)
                io.imsave(output_path, image)
            else:
                print("изображение не подошло", filename)

def complementary_hsv(hsv_color):
    """
    Находит комплементарный цвет в пространстве HSV для заданного цвета.
    """
    h_complementary = (hsv_color[0] + 180) % 360
    s_complementary = hsv_color[1]
    v_complementary = hsv_color[2]
    return (h_complementary, s_complementary, v_complementary)

input_folder = "background_removed_images_with_bgrem"
output_folder = "complimentFrinder"
target_color_hsv = (47, 2, 84)
complementary_hsv1 = complementary_hsv(target_color_hsv)
threshold = 50
find_images_by_color(input_folder, output_folder, complementary_hsv1, threshold)