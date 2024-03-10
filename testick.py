from colormath.color_objects import LabColor, HSVColor
from colormath.color_conversions import convert_color

def hsv_to_lab(h, s, v):
    hsv_color = HSVColor(h, s / 100, v / 100)
    lab_color = convert_color(hsv_color, LabColor)
    return lab_color.lab_l, lab_color.lab_a, lab_color.lab_b

hsv_color = (11, 6, 30)
lab_color = hsv_to_lab(*hsv_color)
print(lab_color)
