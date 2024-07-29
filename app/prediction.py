import glob

import pandas as pd
from pydicom.errors import InvalidDicomError
import os
import pydicom
import numpy as np
import dicom2nifti
import shutil
from copy import deepcopy
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
import csv
import matplotlib

from app.countpixel import count
from app.parcellation import wmparc
from app.yolo_pipeline import yolo_train
from app.yolo_pipeline.yolo_train import example_case
from config.settings import path_routing
from app.fast_surfer_morphometry import get_nifti_meta, fastsurfer
import dicom2nifti.settings as settings
settings.disable_validate_orthogonal()
settings.enable_resampling()
settings.set_resample_spline_interpolation_order(1)
settings.set_resample_padding(-1000)
matplotlib.use('TkAgg')
import subprocess
from matplotlib.pyplot import get_cmap


def folder_creator(num_res):
    name = 'research'

    os.makedirs(path_routing.morphometry_path, exist_ok=True)
    num_folder = ''
    if len(os.listdir(path_routing.morphometry_path)):
        list_ = list(filter(lambda x: os.path.isdir(os.path.join(path_routing.morphometry_path, x)),
                            os.listdir(path_routing.morphometry_path)))
        # Фильтрация только тех папок, которые имеют ожидаемый формат
        list_ = [x for x in list_ if x.startswith(name) and x[len(name):].lstrip('_').isdigit()]
        if list_:
            last = sorted(list_, key=lambda x: int(x.split('_')[-1]))[-1]
            num_folder = str(int(last.split('_')[-1]) + 1)
        else:
            num_folder = '0'
    else:
        num_folder = '0'
    os.makedirs(os.path.join(path_routing.morphometry_path, name + '_' + num_res), exist_ok=True)
    path_to_folder = os.path.join(path_routing.morphometry_path, name + '_' + num_res)
    return path_to_folder

def get_nifti_meta(input_data):
    # Check if input is a path or a Nifti1Image object
    if isinstance(input_data, str):
        nifti_path = input_data
        nifti_path = os.path.join(nifti_path, 'NIFTI') if nifti_path.split(os.path.sep)[-1] != 'NIFTI' else nifti_path

        file_path = ''
        for root, dirs, files in os.walk(nifti_path):
            for fname in files:
                if ('fluid' in fname.lower() or 'flair' in fname.lower()) and (
                        fname.endswith('.nii.gz') or fname.endswith('.nii')):
                    file_path = os.path.join(root, fname)
                    break
            if file_path:
                break

        if not file_path:
            print('Отсутствуют файлы DICOM с меткой T2')
            return None, None

        try:
            epi_img = nib.load(file_path)
        except IndexError:
            print('Отсутствуют файлы DICOM')
            return None, None

    elif isinstance(input_data, nib.Nifti1Image):
        epi_img = input_data

    else:
        raise ValueError("Input must be a path to a NIFTI directory or a Nifti1Image object")

    epi_img_data = epi_img.get_fdata()  # Correct method name
    return epi_img, epi_img_data


def get_volume(path_to_dicom, path_to_model, path_to_final_folder):
    to_dicom = os.path.join(path_to_final_folder, 'DICOM')
    to_nifti = os.path.join(path_to_final_folder, 'NIFTI')
    to_image = os.path.join(path_to_final_folder, 'SLICES')
    to_predict = os.path.join(path_to_final_folder, 'PREDICTED')
    to_mask = os.path.join(path_to_final_folder, 'MASK')
    to_wmparc = os.path.join(path_to_final_folder, 'WMPARC')

    os.makedirs(to_dicom, exist_ok=True)
    os.makedirs(to_nifti, exist_ok=True)
    os.makedirs(to_image, exist_ok=True)
    os.makedirs(to_predict, exist_ok=True)
    os.makedirs(to_mask, exist_ok=True)
    os.makedirs(to_wmparc, exist_ok=True)

    # номер исследования
    num_research = path_to_dicom.split(os.path.sep)[-1]
    # Поиск и сохранение DICOM
    print("Текущее исследование: " + num_research)
    for root, dirs, files in os.walk(path_to_dicom):
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
                continue
            # search for volume with contrast
            orientation_bool = all(orientation[:3] == [1, 0, 0]) and all(orientation[3:] == [0, 1, 0])
            name = dicom_content.SeriesDescription.lower()
            flair = 'FLAIR'.lower() in dicom_content.SeriesDescription.lower()
            fluid = 'fluid' in dicom_content.SeriesDescription.lower()
            t1 = 't1' in dicom_content.SeriesDescription.lower()
            mode_bool = flair or fluid or t1
            if mode_bool and (orientation_bool or 'tra' in dicom_content.SeriesDescription.lower()):
                file_path_for_nifti = deepcopy(file_path)
                name = np.where(np.array([flair, fluid, t1]))[0][0]
                new_name = file_path_for_nifti + '_' + str(name)
                os.rename(file_path, new_name)
                shutil.copy(new_name, to_dicom)

    # перевод DICOM в NIFTI
    dicom2nifti.convert_directory(to_dicom, to_nifti, compression=True, reorient=True)

    try:
        # Найти все файлы NIfTI в указанной папке
        nifti_files = glob.glob(os.path.join(to_nifti, '*.nii')) + glob.glob(os.path.join(to_nifti, '*.nii.gz'))
        if not nifti_files:
            print(f"No NIfTI files found in folder: {to_nifti}")
            return

        keywords = ['flair', 'fluid']

        # Затем пройдите по всем файлам и найдите те, которые содержат ключевые слова
        matching_files = [file for file in nifti_files if any(keyword in file.lower() for keyword in keywords)]

        # Проверьте, найдены ли какие-либо файлы
        if not matching_files:
            print(f"No NIfTI files found with keywords: {', '.join(keywords)}")
            return

        # Загрузите первый найденный файл из списка matching_files
        nifti_img = nib.load(matching_files[0])

        # Получите метаданные этого файла
        _, data_nifti = get_nifti_meta(matching_files[0])

    except TypeError:
        create_csv(path_to_final_folder, num_research, to_nifti, True)
        return

    # Сохранение срезов как изображений
    nifti_data = nifti_img.get_fdata()
    for i in range(nifti_data.shape[2]):
        slice_img = nifti_data[:, :, i]
        output_file = os.path.join(to_image, f"slice_{i}.png")
        plt.imshow(slice_img, cmap='gray')
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)

    # Обработка изображений YOLO
    for path_to_image in glob.glob(os.path.join(to_image, '*.png')):
        result = example_case(path_to_image, path_to_model, save_image_dir=to_predict, to_mask=to_mask)
        if result is not None:
            image_, overlay = result
            plt.imshow(cv2.cvtColor(image_, cv2.COLOR_BGR2RGB))  # Display original image

    create_csv(path_to_final_folder, num_research, to_nifti, to_wmparc,to_mask)

def make_zero_affine(path):
    date = nib.load(path)
    new_data = nib.nifti1.Nifti1Image(date.get_fdata(), affine=date.header.get_base_affine())
    path_without_name = os.path.sep.join(path.split(os.path.sep)[:-1])
    name = path.split(os.path.sep)[-1].split('.')
    name[0] = name[0] + '_zero_affine'

    new_path = os.path.join(path_without_name, '.'.join(name))
    nib.save(new_data, new_path)

    return new_path


def create_csv(path_to_nifti, number_research, to_nifti, to_wmparc, to_mask,error_=False):
    print(f"Checking for NIfTI files in folder: {to_nifti}")
    nifti_files = glob.glob(os.path.join(to_nifti, '*.nii')) + glob.glob(os.path.join(to_nifti, '*.nii.gz'))
    print(f"Found NIfTI files: {nifti_files}")

    if not nifti_files:
        print(f"No NIfTI files found in folder: {to_nifti}")
        return

    keywords = ['T1', 't1', 'tra']
    matching_files = [file for file in nifti_files if any(keyword in file.lower() for keyword in keywords)]
    print(f"NIfTI files matching keywords {keywords}: {matching_files}")

    if not matching_files:
        print(f"No NIfTI files found with keywords: {', '.join(keywords)}")
        return

    heaviest_file = max(matching_files, key=os.path.getsize)
    print(f"Loading NIfTI file: {heaviest_file}")
    nifti_img = nib.load(heaviest_file)
    print(heaviest_file)

    _, data_nifti = get_nifti_meta(heaviest_file)
    path = make_zero_affine(heaviest_file)
    print(f"Calling fastsurfer with path: {path} and research number: {number_research}")
    docker_subject_id = fastsurfer([path], number_research)
    processed_data_folder = path_to_nifti
    global_path_for_file = os.path.sep.join(processed_data_folder.split(os.path.sep)[:-1])


    wmparc_path = os.path.normpath(os.path.join(path_to_nifti, to_nifti, docker_subject_id, 'mri', 'wmparc.DKTatlas.mapped.mgz'))
    if not os.path.exists(wmparc_path):
        print(f"wmparc.DKTatlas.mapped.mgz file not found at path: {wmparc_path}")
        return

    print(f"Found wmparc.DKTatlas.mapped.mgz at path: {wmparc_path}")

    t2_file_candidates = glob.glob(os.path.join(path_to_nifti, to_nifti, '*T2*')) + glob.glob(os.path.join(path_to_nifti, to_nifti, '*Flair*'.lower()))
    t1_file_candidates = os.path.join(path_to_nifti, to_nifti, docker_subject_id, 'mri', 'T1.mgz')

    if not t2_file_candidates:
        print("T2 or FLAIR files not found")
        return

    t2_file = t2_file_candidates[0]
    t1_file = t1_file_candidates

    print(f"Calling wmparc function with: {wmparc_path}, T2 file: {t2_file}, T1 file: {t1_file}")
    wmparc(wmparc_path, t1_file,t2_file , to_wmparc)


    csv_file_path = os.path.join(global_path_for_file, 'brain_volumes.csv')
    count(to_wmparc,to_mask,csv_file_path,docker_subject_id,path)


def all_research(path_to_all_research):
    # Список уже существующих папок в path_routing.morphometry_path
    existing_folders = os.listdir(path_routing.morphometry_path)

    # Путь к общему CSV файлу
    global_csv_path = os.path.join(path_routing.morphometry_path, 'brain_volumes.csv')
    brain_volumes_df = pd.DataFrame()

    for folder_ in os.listdir(path_to_all_research):
        path_to_one_research = os.path.join(path_to_all_research, folder_)
        research_name = f'research_{folder_}'

        # Проверка, обработана ли уже эта папка
        if research_name in existing_folders:
            # Проверить, есть ли файлы в папке to_wmparc
            to_wmparc_folder = os.path.join(path_routing.morphometry_path, research_name, 'WMPARC')
            if os.path.exists(to_wmparc_folder) and os.listdir(to_wmparc_folder):
                print(f"Исследование {research_name} уже обработано и имеет файлы в папке WMPARC.")
            else:
                print(f"Исследование {research_name} уже существует, но папка WMPARC пуста. Перезапуск анализа...")
                # Создать папку для исследования, если её нет
                if not os.path.exists(os.path.join(path_routing.morphometry_path, research_name)):
                    path_to_final_folder = folder_creator(folder_)
                else:
                    path_to_final_folder = os.path.join(path_routing.morphometry_path, research_name)

                # Вызов функции для обработки данных
                get_volume(path_to_one_research, path_routing.path_to_model, path_to_final_folder)

                # Вызов функции для создания CSV
                create_csv(path_to_final_folder, research_name, path_to_final_folder, to_wmparc_folder, error_=True)

        else:
            path_to_final_folder = folder_creator(folder_)
            to_wmparc_folder = os.path.join(path_to_final_folder, 'WMPARC')
            get_volume(path_to_one_research, path_routing.path_to_model, path_to_final_folder)
            create_csv(path_to_final_folder, research_name, path_to_final_folder, to_wmparc_folder, error_=True)

all_research(path_routing.marked_mri_path)
