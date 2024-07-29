import json
import os
import shutil
from copy import deepcopy

import cv2
import dicom2nifti
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
import glob
from config.settings import path_routing

# path_to_all = PROJECT_DIR
path_to_database = path_routing.processed_data_path
path_to_all = path_routing.train_set_path

os.makedirs(path_routing.train_set_path,exist_ok = True)
os.makedirs(os.path.join(path_routing.train_set_path, 'images'),exist_ok = True)
os.makedirs(os.path.join(path_routing.train_set_path,'labels'),exist_ok = True)

def create_train_set():
    for root, dirs, files in os.walk(path_to_database):
        if 'images' in root or 'labels' in root:
            for file_ in files:
                path_to_image = os.path.join(root, file_)
                # Используем os.path.sep для разделителя пути, чтобы код работал на разных операционных системах
                new_name = path_to_image.split(os.path.sep)[-1][:-4] + '_' + root.split(os.path.sep)[-3] + path_to_image[-4:]
                p = os.path.join(os.path.join(path_to_all, path_to_image.split(os.path.sep)[-2]), new_name)
                shutil.copy2(path_to_image, p)

    #Создает нужные папки для ращделения
    devide_ = ['train','val','test']
    path_to_all_image = os.path.join(path_routing.train_set_path,'images')
    path_to_all_labels = os.path.join(path_routing.train_set_path,'labels')
    path_to_main = [path_to_all_image,path_to_all_labels]
    [os.makedirs(os.path.join(path_to_,devide_i),exist_ok=True) for path_to_ in path_to_main for devide_i in devide_]

    train_val_test = [.8,.1,.1]
    size_ = len(os.listdir(path_to_all_image)) - 3
    temp = glob.glob(path_to_all_image + '/*.png')
    temp.sort(key=lambda x: x.split(os.path.sep)[-1][:-4].split('_')[-1])
    for i, file_ in enumerate(temp):
        req_file = glob.glob(os.path.join(path_to_all_labels, file_.split(os.path.sep)[-1][
                                                              :-4] + '.txt'))  # получение 2-х динаковых файло (img,label)
        img_file = file_.split(os.path.sep)[-1]
        # temp_value - переменная для разделения на выборки
        temp_value = devide_[0] if i < int(size_ * train_val_test[0]) else devide_[1] if i < int(
            size_ * (train_val_test[0] + train_val_test[1])) else devide_[2]
        # разделение
        shutil.move(os.path.join(path_to_all_image, img_file), os.path.join(path_to_all_image, temp_value, img_file))
        shutil.move(req_file[0], os.path.join(path_to_all_labels, temp_value, req_file[0].split(os.path.sep)[-1]))

create_train_set()