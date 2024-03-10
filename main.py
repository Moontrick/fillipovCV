import os
import random
import cv2
import cv2 as cv
import requests
from rembg import remove
import numpy as np
from urllib.parse import urljoin, urlparse
#Сохранение фото
def save_img(data, folder_path):
    for group in data['groups']:
        for item in group['items']:
            img_url = item['image']
            try:
                img_response = requests.get(img_url)
                img_name = os.path.basename(urlparse(img_url).path)
                img_path = os.path.join(folder_path, img_name)
                with open(img_path, 'wb') as img_file:
                    img_file.write(img_response.content)
                print(f"Изображение сохранено: {img_path}")
            except Exception as e:
                print(f"Ошибка при загрузке изображения {img_url}: {e}")


#Кол-во фото в папке
def count_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    num_files = len(files)
    print(num_files)
#Получение jsonфайла
def download_images(url, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print('Папка создана')
    response = requests.get(url)
    print(response.status_code)
    if response.status_code == 200:
        data = response.json()
        save_img(data, folder_path)

def remove_background(image_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    img = cv.imread(image_path)
    gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    bl = cv.medianBlur(gr, 3)
    canny = cv.Canny(bl, 1, 100)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (230, 3))
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)

    contours = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    mask = np.zeros_like(gr)
    cv.drawContours(mask, contours, -1, (255), thickness=cv.FILLED)

    alpha = np.zeros_like(gr)
    alpha[mask == 255] = 255

    # Создание 4-канального изображения (BGRA)
    bgra = cv.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha])

    # Удаление черного фона
    result = cv.bitwise_and(bgra, bgra, mask=alpha)
    filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, result)

def remove_init(folder_path, out_path):
    files = os.listdir(folder_path)
    for file in files:
        remove_background(folder_path+ '/'+ file, out_path)




def blend_images(obj_img, bg_img):
    # Изменяем размер объекта до размера фона
    obj_img_resized = cv2.resize(bg_img, (obj_img.shape[1], obj_img.shape[0]))

    # Создаем многомасштабную пирамиду для объекта и фона
    obj_pyr = [obj_img.copy()]
    bg_pyr = [obj_img_resized.copy()]
    for _ in range(5):  # 5 уровней пирамиды
        obj_pyr.append(cv2.pyrDown(obj_pyr[-1]))
        bg_pyr.append(cv2.pyrDown(bg_pyr[-1]))

    # Вклеиваем объект на каждом уровне пирамиды фона
    for i in range(len(obj_pyr)):
        # Получаем текущий уровень пирамиды для объекта и фона
        obj_level = obj_pyr[i]
        bg_level = bg_pyr[i]

        # Изменяем размер объекта до размера фона текущего уровня
        obj_level_resized = cv2.resize(obj_level, (bg_level.shape[1], bg_level.shape[0]))

        # Копируем объект на фон
        blended_img = bg_level.copy()
        blended_img[:obj_level_resized.shape[0], :obj_level_resized.shape[1]] = obj_level_resized

        # Обновляем фон для следующего уровня
        bg_pyr[i] = blended_img

    # Возвращаем окончательное изображение с вклеенным объектом
    return bg_pyr[-1]




def removeBGwithRembg(image_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    input = cv2.imread(image_path)
    output = remove(input)
    filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, output)

def removewithRembg_init(folder_path, out_path):
    files = os.listdir(folder_path)
    for file in files:
        removeBGwithRembg(folder_path+ '/'+ file, out_path)








url_to_scrape = 'https://shop.mango.com/ws-product-lists/v1/channels/shop/countries/us/catalogs/prendas_she.camisetas_she?language=en'
folder_path = 'downloaded_images'
#download_images(url_to_scrape, folder_path)
count_files_in_folder(folder_path)

output_folder = 'background_removed_images'
#remove_init(folder_path, output_folder)
output_folder1 = 'background_removed_images_with_bgrem'
#removewithRembg_init(folder_path, output_folder1)


def create_mask(image):
    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем пороговый метод для выделения объекта
    _, thresholded = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Ищем контуры на изображении
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем пустое изображение для маски
    mask = np.zeros_like(gray)

    # Заполняем маску контурами
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    return mask

def background_insertion(directory='background_removed_images_with_bgrem/', background="back.jpg", imageCount=5):
    new_directory = 'newBackground/'
    os.makedirs(new_directory, exist_ok=True)

    files = os.listdir(directory)
    random_images = random.sample(files, imageCount)

    for image in random_images:
        image_path = os.path.join(directory, image)

        foreground = cv2.imread(image_path)
        foreground_height, foreground_width = foreground.shape[:2]

        background_img = cv2.imread(background)
        background_resized = cv2.resize(background_img, (foreground_width, foreground_height))

        foreground_pyramid = tuple(cv2.pyrDown(foreground) for _ in range(3))
        background_pyramid = tuple(cv2.pyrDown(background_resized) for _ in range(3))
        result = None

        for i in range(3):
            background_height, background_width = background_pyramid[i].shape[:2]
            roi = cv2.resize(foreground_pyramid[i], (background_width, background_height))

            foreground_gray = cv2.cvtColor(foreground_pyramid[i], cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(foreground_gray, 1, 255, cv2.THRESH_BINARY)

            foreground_masked = cv2.bitwise_and(roi, roi, mask=mask)
            background_masked = cv2.bitwise_and(background_pyramid[i], background_pyramid[i], mask=cv2.bitwise_not(mask))
            blended = cv2.addWeighted(foreground_masked, 1, background_masked, 1, 0)

            if i < 2:
                blended = cv2.resize(blended, (foreground_pyramid[i + 1].shape[1], foreground_pyramid[i + 1].shape[0]))
                blended = cv2.pyrUp(blended)

            result = blended

        # Save result
        new_image_path = os.path.join(new_directory, image)
        cv2.imwrite(new_image_path, result)

background_insertion()
