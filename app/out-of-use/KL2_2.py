import json
import os
import shutil
from copy import deepcopy
import matplotlib
matplotlib.use('TkAgg')
import cv2
import dicom2nifti  # to convert DICOM files to the NIfTI format
import matplotlib.pyplot as plt
import nibabel as nib  # nibabel to handle nifti files
import numpy as np
import pydicom  # pydicom to handle dicom files
import glob
from config.settings import path_routing
from pydicom.errors import InvalidDicomError

# Цвета для масок в зависимости от первой буквы
color_dict = {
    'I': (255, 0, 0),  # Красный
    'P': (0, 255, 0),  # Зеленый
    'S': (0, 0, 255),  # Синий
    'U': (255, 255, 0) # Желтый
}

def create_nifti_from_contrast_volumes(root_path, postpr_path):
    print("Запуск функции create_nifti_from_contrast_volumes")
    # walk the path and creates nifti from dicom in each survey then returns list of survey paths and list of paths to DICOM in marked_mri
    survey_list = []
    raw_data_list = []
    survey_name = ''
    survey_path = ''
    for root, dirs, files in os.walk(root_path):
        # search for DICOM
        for fname in files:

            file_path = os.path.join(root, fname)
            try:
                dicom_content = pydicom.dcmread(file_path)
            except InvalidDicomError:
                print("Невозможно прочитать файл: " + fname)
                continue
            try:
                orientation = np.round(dicom_content.ImageOrientationPatient)
            except AttributeError:
                print('Отсутствует или не верно указана ориентация файла')
                continue

            # search for volume with contrast
            orientation_bool = all(orientation[:3] == [1, 0, 0]) and all(orientation[3:] == [0, 1, 0])
            mode_bool = 'flair' in dicom_content.SeriesDescription.lower() or 'fluid' in dicom_content.SeriesDescription.lower()
            if mode_bool and (orientation_bool or 'tra' in dicom_content.SeriesDescription.lower()):
                file_path_for_nifti = deepcopy(file_path)
                survey_name = root_path.split(os.path.sep)[-1]

                # make directories in processed data
                os.makedirs(os.path.join(postpr_path, survey_name, 'DICOM'), exist_ok=True)
                os.makedirs(os.path.join(postpr_path, survey_name, 'NIFTI'), exist_ok=True)

                shutil.copy(file_path_for_nifti, os.path.join(postpr_path, survey_name, 'DICOM'))
                # add survey path to list
        print(survey_name)  # debug

    dicom2nifti.convert_directory(os.path.join(postpr_path, survey_name, 'DICOM'),
                                      os.path.join(postpr_path,survey_name, 'NIFTI'), compression=True, reorient=True)



def get_nifti_meta(nifti_path):
    print("Запуск функции get_nifti_meta")
    # create two variables from one nifti meta
    nifti_path = os.path.join(nifti_path, 'NIFTI') if nifti_path.split(os.path.sep)[-1] != 'NIFTI' else nifti_path
    try:
        epi_img = nib.load(os.path.join(nifti_path, os.listdir(nifti_path)[0]))
    except IndexError:
        print('Отсутствуют файлы NIFTI')
        return

    epi_img_data = epi_img.get_fdata()
    return epi_img, epi_img_data


def set_needed_file(postpr_path, root_path):
    for root, dirs, files in os.walk(postpr_path):
        if 'NIFTI' in dirs and root.split(os.path.sep)[-1] in os.listdir(root_path):
            os.makedirs(os.path.join(root, 'NIFTI_T1'), exist_ok=True)
            os.makedirs(os.path.join(root, 'DICOM_T1'), exist_ok=True)
            folder_ = root.split(f'{os.path.sep}')[-1]
            if folder_ in os.listdir(root_path):
                path = os.path.join(root_path, folder_, 'DICOM', '**')
                for file_ in glob.glob(path, recursive=True):
                    if file_.split(os.path.sep)[-1].startswith('IM'):
                        dicom_content = pydicom.dcmread(file_)
                        orientation = np.round(dicom_content.ImageOrientationPatient)  # ориентация
                        tra_in = 'tra' in dicom_content.SeriesDescription.lower()
                        t1_in = 'T1'.lower() in dicom_content.SeriesDescription.lower()
                        if t1_in and ((all(orientation[:3] == [1, 0, 0]) and all(orientation[3:] == [0, 1, 0])) or tra_in):
                            path = os.path.join(postpr_path + os.path.sep + folder_, 'DICOM_T1',
                                                file_.split(os.path.sep)[-1])
                            shutil.copy(file_, path)
            print(folder_)
            dicom2nifti.convert_directory(os.path.join(postpr_path, folder_, 'DICOM_T1'),
                                          os.path.join(postpr_path, folder_, 'NIFTI_T1'), compression=True,
                                          reorient=True)


def create_demyelination_areas_markups(epi_img, markup_path):
    print("Запуск функции create_demyelination_areas_markups")
    # read jsons and create a list of matrixes with coordinates for one nifti
    points_lps = []
    # walk through .json filenames and add their location to list
    for root, dirs, files in os.walk(markup_path):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, 'r') as f:
                    local_point_list = []
                    json_data = json.load(f)
                    # form a 4x1 np matrix using 'orientation' and 'position' from each .json
                    for point in json_data["markups"][0]["controlPoints"]:
                        orientation_matrix = np.reshape(np.array(point['orientation']), (3, 3))
                        position_matrix = np.reshape(np.array(point["position"]), (3, 1))
                        local_point_list.append(
                            np.vstack(((orientation_matrix @ position_matrix), [1])))
                    points_lps.append((file,local_point_list))
    new_points_lps = []
    for file, point in points_lps:
        local_list = []
        for local_point in point:
            index_coords = np.round(np.dot(np.linalg.inv(epi_img.affine), np.array(local_point)))
            local_list.append(index_coords)
        new_points_lps.append((file,local_list))
    return new_points_lps


def create_yolo_data(epi_img, epi_img_data, new_points_lps, yolo_path):
    print("Запуск функции create_yolo_data")
    path_to_Yolo_data = os.path.join(yolo_path, 'Yolo')
    os.makedirs(path_to_Yolo_data, exist_ok=True)
    os.makedirs(os.path.join(path_to_Yolo_data, 'annotation'), exist_ok=True)
    os.makedirs(os.path.join(path_to_Yolo_data, 'images'), exist_ok=True)
    os.makedirs(os.path.join(path_to_Yolo_data, 'labels'), exist_ok=True)

    mask_index = 0  # Счетчик для масок
    mask_files = {}  # Словарь для хранения файлов масок по срезам

    transformations = {
        "original": lambda img: img,
        "flip_x": lambda img: np.flipud(img),
        "flip_y": lambda img: np.fliplr(img),
    }
    color_to_class = {
        '25500': 1,  # Красный
        '02550': 2,  # Зеленый
        '00255': 3,  # Синий
        '2552550': 4  # Желтый
    }

    for i in range(epi_img.shape[2]):
        slice_has_masks = False
        annotations = []  # Список для аннотаций

        for file, points in new_points_lps:
            if len(points) > 0 and int(points[0][2].item()) == i:  # Явное извлечение значения из numpy-матрицы
                slice_has_masks = True
                mask_image = np.zeros((epi_img.shape[0], epi_img.shape[1]), dtype=np.uint8)  # Черная маска

                # Создание контуров вокруг областей внутри точек
                contour = [(int(point[1].item()), int(point[0].item())) for point in points]  # Явное извлечение значения из numpy-матрицы
                first_letter = file[0]
                color = color_dict.get(first_letter, (255, 255, 255))  # Белый по умолчанию
                cv2.fillPoly(mask_image, [np.array(contour, dtype=np.int32)], 255)

                for transform_name, transform_func in transformations.items():
                    transformed_mask = transform_func(mask_image)
                    color_name = ''.join(map(str, color))
                    mask_filename = f"slice_{i}_mask_{mask_index}_{color_name}_{transform_name}.png"
                    mask_index += 1

                    if i not in mask_files:
                        mask_files[i] = []
                    mask_files[i].append(mask_filename)

                    plt.imshow(transformed_mask, cmap='gray')
                    plt.axis('off')
                    plt.savefig(os.path.join(path_to_Yolo_data, 'annotation', mask_filename), bbox_inches='tight', pad_inches=0, transparent=True)

                    # Преобразование маски в полигоны и добавление аннотаций
                    contours, _ = cv2.findContours(transformed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    H, W = transformed_mask.shape
                    for cnt in contours:
                        polygon = []
                        for point in cnt:
                            x, y = point[0]
                            polygon.append(x / W)
                            polygon.append(y / H)
                        if polygon:
                            annotations.append(f"{color_to_class[color_name]} " + " ".join(map(str, polygon)))

        for transform_name, transform_func in transformations.items():
            transformed_slice = transform_func(epi_img_data[:, :, i])
            original_slice_filename = f"slice_{i}_{transform_name}.png"
            plt.imshow(transformed_slice, cmap=plt.cm.gray)
            plt.axis('off')
            plt.savefig(os.path.join(path_to_Yolo_data, 'images', original_slice_filename), bbox_inches='tight', pad_inches=0)

            # Создание и заполнение txt файла для каждого среза
            with open(os.path.join(path_to_Yolo_data, 'labels', f"slice_{i}_{transform_name}.txt"), 'w') as f:
                for annotation in annotations:
                    f.write(annotation + "\n")


def mask_to_polygons(path_to_mask):
    print("Запуск функции mask_to_polygons")
    path_to_mask = os.path.join(path_to_mask,'Yolo')
    output_dir = os.path.join(path_to_mask, 'labels')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

    input_dir = os.path.join(path_to_mask,'annotation')
    # Определение классов на основе цветов масок
    color_to_class = {
        '25500': 1,  # Красный
        '02550': 2,  # Зеленый
        '00255': 3,  # Синий
        '2552550': 4  # Желтый
    }

    for j in os.listdir(input_dir):
        image_path = os.path.join(input_dir, j)
        if j.endswith('png'):
            # Загрузка маски и определение её класса на основе цвета
            mask_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            color_code = j.split('_')[-1].split('.')[0]
            class_id = color_to_class.get(color_code, -1)

            if class_id == -1:
                print(f"Unknown color code: {color_code}")
                continue

            # Преобразование бинарной маски в полигоны
            _, mask = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)
            H, W = mask.shape
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            polygons = []
            for cnt in contours:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

            # Запись полигонов в текстовый файл в формате YOLO
            with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
                for polygon in polygons:
                    if polygon:  # Убедитесь, что полигон не пустой
                        f.write(f"{class_id} " + " ".join(map(str, polygon)) + "\n")

def read_file(filepath):
    print("Запуск функции read_file")
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(filepath, content):
    print("Запуск функции write_file")
    with open(filepath, 'a', encoding='utf-8') as file:
        file.write(content)

def merge_mask_txt_files(labels_path):
    print("Запуск функции merge_mask_txt_files")
    for root, dirs, files in os.walk(labels_path):
        for filename in files:
            if filename.startswith('slice_') and filename.endswith('.txt'):
                slice_num = filename.split('_')[1]
                if 'mask' in filename:
                    content = read_file(os.path.join(root, filename))
                    write_file(os.path.join(root, f'slice_{slice_num}.txt'), content + "\n")
                    os.remove(os.path.join(root, filename))
    print("Объединение файлов завершено и файлы с масками удалены.")

def raw_data_processing(path_to_all,path_out_all):

    for folder_ in os.listdir(path_to_all):
        #Прописываем пути
        path_ = os.path.join(path_to_all,folder_)
        path_out = os.path.join(path_out_all,folder_)
        #Ищем подходящие NIFTI
        create_nifti_from_contrast_volumes(root_path=path_,postpr_path=path_out_all)
        #Получаем данные из NIFTI
        try:
            epi_img, epi_img_data = get_nifti_meta(nifti_path=path_out)
        except FileNotFoundError:
            print(f'Ошибка чтения NIFTI из raw_data_processing \nИмя файла : {path_out}')
            continue
        #Выгрузка данных из json
        new_points_lps = create_demyelination_areas_markups(epi_img, markup_path=path_)
        #Создание папки для yolo
        create_yolo_data(epi_img, epi_img_data, new_points_lps, yolo_path=path_out)
        #Перезапись массива точек в классовые файлы
        mask_to_polygons(path_to_mask=path_out)
        #перезапись классовых файлов в единый txt
        merge_mask_txt_files(labels_path=os.path.join(path_out, 'Yolo', 'labels'))




if __name__ == '__main__':
    raw_data_processing(path_routing.marked_mri_path, path_routing.processed_data_path)

