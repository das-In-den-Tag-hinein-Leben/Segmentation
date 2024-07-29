import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nibabel as nib
import csv
from matplotlib.colors import ListedColormap
from skimage import color
from collections import defaultdict
import cv2
import numpy as np


labels_rgb_dict = {
    (0, 0, 0): 'Background',                        # 0
    (0, 0, 255): 'Non-ventricular',                 # 4
    (127, 255, 212): '3rd-Ventricle',               # 5
    (240, 230, 140): '4th-Ventricle',               # 6
    (176, 48, 96): '5th-Ventricle',                 # 7
    (48, 176, 96): 'Right-Accumbens-Area',          # 8
    (48, 176, 96): 'Left-Accumbens-Area',           # 9
    (103, 255, 255): 'Right-Amygdala',              # 10
    (103, 255, 255): 'Left-Amygdala',               # 11
    (238, 186, 243): 'Pons',                        # 12
    (119, 159, 176): 'Brain-Stem',                  # 13
    (122, 186, 220): 'Right-Caudate',               # 14
    (122, 186, 220): 'Left-Caudate',                # 15
    (96, 204, 96): 'Right-Cerebellum-Exterior',     # 16
    (96, 204, 96): 'Left-Cerebellum-Exterior',      # 17
    (220, 247, 164): 'Right-Cerebellum',            # 18
    (220, 247, 164): 'Left-Cerebellum',             # 19
    (60, 60, 60): '3rd-Ventricle-(Posterior-part)', # 22
    (220, 216, 20): 'Right-Hippocampus',            # 23
    (220, 216, 20): 'Left-Hippocampus',             # 24
    (196, 58, 250): 'Right-Inf-Lat-Vent',           # 25
    (196, 58, 250): 'Left-Inf-Lat-Vent',            # 26
    (120, 18, 134): 'Right-Lateral-Ventricle',      # 27
    (120, 18, 134): 'Left-Lateral-Ventricle',       # 28
    (12, 48, 255): 'Right-Pallidum',                # 29
    (12, 48, 225): 'Left-Pallidum',                 # 30
    (236, 13, 176): 'Right-Putamen',                # 31
    (236, 13, 176): 'Left-Putamen',                 # 32
    (0, 118, 14): 'Right-Thalamus-Proper',          # 33
    (0, 118, 14): 'Left-Thalamus-Proper',           # 34
    (165, 42, 42): 'Right-Ventral-DC',              # 35
    (165, 42, 42): 'Left-Ventral-DC',               # 36
    (160, 32, 240): 'Right-vessel',                 # 37
    (160, 32, 240): 'Left-vessel',                  # 38
    (56, 192, 255): 'Right-periventricular-white-matter',   # 39
    (56, 192, 255): 'Left-periventricular-white-matter',    # 40
    (255, 225, 225): 'Optic-Chiasm',                # 41
    (184, 237, 194): 'Cerebellar-Vermal-Lobules-I-V',      # 42
    (180, 231, 250): 'Cerebellar-Vermal-Lobules-VI-VII',   # 43
    (225, 183, 231): 'Cerebellar-Vermal-Lobules-VIII-X',   # 44
    (180, 180, 180): 'Left-Basal-Forebrain',         # 45
    (180, 180, 180): 'Right-Basal-Forebrain',        # 46
    (245, 255, 200): 'Right-Temporal-White-Matter',  # 47
    (255, 230, 255): 'Right-Insula-White-Matter',    # 48
    (245, 245, 245): 'Right-Cingulate-White-Matter', # 49
    (220, 255, 220): 'Right-Frontal-White-Matter',   # 50
    (220, 220, 220): 'Right-Occipital-White-Matter', # 51
    (200, 255, 255): 'Right-Parietal-White-Matter',  # 52
    (250, 220, 200): 'Corpus-Callosum',              # 53
    (245, 255, 200): 'Left-Temporal-White-Matter',   # 54
    (255, 230, 255): 'Left-Insula-White-Matter',     # 55
    (245, 245, 245): 'Left-Cingulate-White-Matter',  # 56
    (220, 255, 220): 'Left-Frontal-White-Matter',    # 57
    (220, 220, 220): 'Left-Occipital-White-Matter',  # 58
    (200, 255, 255): 'Left-Parietal-White-Matter',   # 59
    (140, 125, 255): 'Right-Claustrum',              # 60
    (140, 125, 255): 'Left-Claustrum',               # 61
    (255, 62, 150): 'Right-ACgG-anterior-cingulate-gyrus', # 62
    (255, 62, 150): 'Left-ACgG-anterior-cingulate-gyrus',  # 63
    (160, 82, 45): 'Right-AIns-anterior-insula',     # 64
    (160, 82, 45): 'Left-AIns-anterior-insula',      # 65
    (165, 42, 42): 'Right-AOrG-anterior-orbital-gyrus',     # 66
    (165, 42, 42): 'Left-AOrG-anterior-orbital-gyrus',      # 67
    (205, 91, 69): 'Right-AnG-angular-gyrus',        # 68
    (205, 91, 69): 'Left-AnG-angular-gyrus',         # 69
    (100, 149, 237): 'Right-Calc-calcarine-cortex', # 70
    (100, 149, 237): 'Left-Calc-calcarine-cortex',  # 71
    (135, 206, 235): 'Right-CO-central-operculum',  # 72
    (135, 206, 235): 'Left-CO-central-operculum',   # 73
    (250, 128, 114): 'Right-Cun-cuneus',            # 74
    (250, 128, 114): 'Left-Cun-cuneus',             # 75
    (255, 255, 0): 'Right-Ent-entorhinal-area',     # 76
    (255, 255, 0): 'Left-Ent-entorhinal-area',      # 77
    (221, 160, 221): 'Right-FO-frontal-operculum',  # 78
    (221, 160, 221): 'Left-FO-frontal-operculum',   # 79
    (0, 238, 0): 'Right-FRP-frontal-pole',          # 80
    (0, 238, 0): 'Left-FRP-frontal-pole',           # 81
    (205, 92, 92): 'Right-FuG-fusiform-gyrus',      # 82
    (205, 92, 92): 'Left-FuG-fusiform-gyrus',       # 83
    (176, 48, 96): 'Right-GRe-gyrus-rectus',        # 84
    (176, 48, 96): 'Left-GRe-gyrus-rectus',         # 85
    (152, 251, 152): 'Right-IOG-inferior-occipital-gyrus',  # 86
    (152, 251, 152): 'Left-IOG-inferior-occipital-gyrus',   # 87
    (50, 205, 50): 'Right-ITG-inferior-temporal-gyrus',      # 88
    (50, 205, 50): 'Left-ITG-inferior-temporal-gyrus',       # 89
    (0, 100, 0): 'Right-LiG-lingual-gyrus',        # 90
    (0, 100, 0): 'Left-LiG-lingual-gyrus',         # 91
    (173, 216, 230): 'Right-LOrG-lateral-orbital-gyrus',  # 92
    (173, 216, 230): 'Left-LOrG-lateral-orbital-gyrus',   # 93
    (153, 50, 204): 'Right-MCgG-middle-cingulate-gyrus',  # 94
    (153, 50, 204): 'Left-MCgG-middle-cingulate-gyrus',   # 95
    (160, 32, 240): 'Right-MFC-medial-frontal-cortex',     # 96
    (160, 32, 240): 'Left-MFC-medial-frontal-cortex',      # 97
    (0, 206, 208): 'Right-MFG-middle-frontal-gyrus',       # 98
    (0, 206, 208): 'Left-MFG-middle-frontal-gyrus',        # 99
    (51, 50, 135): 'Right-MOG-middle-occipital-gyrus',     # 100
    (51, 50, 135): 'Left-MOG-middle-occipital-gyrus',      # 101
    (135, 50, 74): 'Right-MOrG-medial-orbital-gyrus',      # 102
    (135, 50, 74): 'Left-MOrG-medial-orbital-gyrus',       # 103
    (218, 112, 214): 'Right-MTG-middle-temporal-gyrus',     # 104
    (218, 112, 214): 'Left-MTG-middle-temporal-gyrus',      # 105
    (64, 224, 208): 'Right-OL-orbital-lobule',        # 106
    (64, 224, 208): 'Left-OL-orbital-lobule',         # 107
    (173, 255, 47): 'Right-OpIFG-opercular-part-of-the-inferior-frontal-gyrus',    # 108
    (173, 255, 47): 'Left-OpIFG-opercular-part-of-the-inferior-frontal-gyrus',     # 109
    (255, 127, 80): 'Right-OrIFG-orbital-part-of-the-inferior-frontal-gyrus',      # 110
    (255, 127, 80): 'Left-OrIFG-orbital-part-of-the-inferior-frontal-gyrus',       # 111
    (255, 140, 0): 'Right-PoCG-postcentral-gyrus',   # 112
    (255, 140, 0): 'Left-PoCG-postcentral-gyrus',    # 113
    (255, 105, 180): 'Right-PoCG-postcentral-gyrus', # 114
    (255, 105, 180): 'Left-PoCG-postcentral-gyrus',  # 115
    (153, 50, 204): 'Right-PoCG-postcentral-gyrus',  # 116
    (153, 50, 204): 'Left-PoCG-postcentral-gyrus',   # 117
    (106, 90, 205): 'Right-PreCG-precentral-gyrus',  # 118
    (106, 90, 205): 'Left-PreCG-precentral-gyrus',   # 119
    (160, 82, 45): 'Right-PrG-precentral-gyrus',     # 120
    (160, 82, 45): 'Left-PrG-precentral-gyrus',      # 121
    (205, 92, 92): 'Right-PrG-precentral-gyrus',     # 122
    (205, 92, 92): 'Left-PrG-precentral-gyrus',      # 123
    (0, 206, 209): 'Right-PT-parietal-lobule',       # 124
    (0, 206, 209): 'Left-PT-parietal-lobule',        # 125
    (0, 100, 0): 'Right-SFG-superior-frontal-gyrus', # 126
    (0, 100, 0): 'Left-SFG-superior-frontal-gyrus',  # 127
    (102, 205, 170): 'Right-SOG-superior-occipital-gyrus',  # 128
    (102, 205, 170): 'Left-SOG-superior-occipital-gyrus',   # 129
    (139, 69, 19): 'Right-SOrG-superior-orbital-gyrus',     # 130
    (139, 69, 19): 'Left-SOrG-superior-orbital-gyrus',      # 131
    (173, 255, 47): 'Right-SFG-superior-frontal-gyrus',      # 132
    (173, 255, 47): 'Left-SFG-superior-frontal-gyrus',       # 133
    (173, 216, 230): 'Right-TL-temporal-lobule',      # 134
    (173, 216, 230): 'Left-TL-temporal-lobule',       # 135
    (153, 50, 204): 'Right-TO-temporal-occipital-junction', # 136
    (153, 50, 204): 'Left-TO-temporal-occipital-junction',  # 137
    (255, 62, 150): 'Right-TrIFG-triangular-part-of-the-inferior-frontal-gyrus',    # 138
    (255, 62, 150): 'Left-TrIFG-triangular-part-of-the-inferior-frontal-gyrus',     # 139
    (0, 139, 69): 'Right-TrIFG-triangular-part-of-the-inferior-frontal-gyrus',      # 140
    (0, 139, 69): 'Left-TrIFG-triangular-part-of-the-inferior-frontal-gyrus',       # 141
    (221, 160, 221): 'Right-TP-temporal-pole',       # 142
    (221, 160, 221): 'Left-TP-temporal-pole',        # 143
    (102, 205, 170): 'Right-ORB-inf-orbital-sulcus', # 144
    (102, 205, 170): 'Left-ORB-inf-orbital-sulcus',  # 145
    (139, 69, 19): 'Right-ORB-inf-orbital-sulcus',   # 146
    (139, 69, 19): 'Left-ORB-inf-orbital-sulcus',    # 147
    (218, 112, 214): 'Right-ORB-mid-orbital-sulcus', # 148
    (218, 112, 214): 'Left-ORB-mid-orbital-sulcus',  # 149
    (255, 127, 80): 'Right-ORB-mid-orbital-sulcus',  # 150
    (255, 127, 80): 'Left-ORB-mid-orbital-sulcus',   # 151
    (102, 205, 170): 'Right-ORB-sup-orbital-sulcus', # 152
    (102, 205, 170): 'Left-ORB-sup-orbital-sulcus',  # 153
    (139, 69, 19): 'Right-ORB-sup-orbital-sulcus',   # 154
    (139, 69, 19): 'Left-ORB-sup-orbital-sulcus',    # 155
    (218, 112, 214): 'Right-ORB-sup-orbital-sulcus',  # 156
    (218, 112, 214): 'Left-ORB-sup-orbital-sulcus',   # 157
    (255, 127, 80): 'Right-ORB-sup-orbital-sulcus',   # 158
    (255, 127, 80): 'Left-ORB-sup-orbital-sulcus',    # 159
}


def count(image_dir, mask_dir, csv_filename,docker_subject_id,path):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
    mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]
    a = nib.load(path).header.get_zooms()
    print(a)

    study_number = os.path.basename(docker_subject_id)
    data_dict = defaultdict(list)

    for mask_path in mask_paths:
        # Extract slice number from mask file name
        slice_number = os.path.basename(mask_path).split('_')[1]

        # Find corresponding image path based on slice number
        for image_path in image_paths:
            if f'slice_{slice_number}' in os.path.basename(image_path):
                image = cv2.imread(image_path)
                mask = cv2.imread(mask_path, 0)

                if mask is None:
                    continue

                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                color_mask = np.zeros_like(image)
                color_mask[mask > 0] = [255, 255, 255]

                masked_roi = cv2.bitwise_and(image, image, mask=mask)
                mask_pixels = np.count_nonzero(mask)

                color_counts = defaultdict(int)

                for row in masked_roi:
                    for pixel in row:
                        blue, green, red = pixel
                        pixel_rgb = (red, green, blue)
                        if pixel_rgb in labels_rgb_dict:
                            color_counts[pixel_rgb] += 1

                black_pixel_count = color_counts[(0, 0, 0)]

                class_number = int(os.path.basename(mask_path).split('_')[2].split('.')[0])
                for color, count in color_counts.items():
                    if color != (0, 0, 0):
                        data_dict[(study_number, class_number)].append(
                            (study_number, class_number, color, count, black_pixel_count))

    with open(csv_filename, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')

        if csv_file.tell() == 0:
            csv_writer.writerow(['Study Number', 'Class', 'Colors', 'Pixel Counts'])

        for (study_number, class_number), class_data in data_dict.items():
            if not class_data:
                continue

            row = [study_number, class_number]
            for _, _, color, count, black_pixel_count in class_data:
                color_name = labels_rgb_dict.get(color, 'Unknown Color')
                scaled_count = count * a[0] * a[1]
                row.append(color_name)
                row.append(scaled_count)

            csv_writer.writerow(row)

    for (study_number, class_number), class_data in data_dict.items():
        if not class_data:
            continue

        for _, _, color, count, _ in class_data:
            color_name = labels_rgb_dict.get(color, 'Unknown Color')
            scaled_count = count * a[0] * a[1]
            print(f"Class {class_number}, Color {color_name}: {scaled_count} pixels")

# count(r"E:\1\Segmentation\database\for_test\research_2505-007-SCREENING\WMPARC",r'E:\1\Segmentation\database\for_test\research_2505-007-SCREENING\MASK',r'E:\1\Segmentation\database\for_test\brain_volumes.csv')
