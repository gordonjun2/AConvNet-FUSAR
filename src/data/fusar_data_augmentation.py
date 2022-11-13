import splitfolders
import glob
import os
import albumentations as A
import numpy as np
import math
import fusar
import cv2
import random
import shutil

def center_crop(data, size=256):
    h, w, _ = data.shape

    y = (h - size) // 2
    x = (w - size) // 2

    return data[y: y + size, x: x + size]

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dataset_root = os.path.join(project_root, 'dataset', 'fusar')
raw_root = os.path.join(dataset_root, 'raw')

output_root = os.path.join(dataset_root, 'augmented')

class_len = {}
max_len = 0

for target in fusar.target_name['fusar']:
    src_path = os.path.join(raw_root, target)
    image_path_list = glob.glob(os.path.join(src_path, '*'))

    # print('Target Name: ', target)
    # print('Images Count: ', len(image_path_list))
    # print('\n')

    if len(image_path_list) >= max_len:
        max_len = len(image_path_list) 

    class_len[target] = len(image_path_list)

class_aug_len_per_image = {}

class_max_len = max_len * 1.2

for key in class_len.keys():
    class_aug_len_per_image[key] = (class_max_len - class_len[key]) / class_len[key]

# Data augmentation

if not os.path.isdir(output_root):
    os.makedirs(output_root)

# Declare an augmentation pipeline

transform = A.Compose([
    A.Rotate(limit=180, p=0.5),
    A.Flip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5)
])

for target in fusar.target_name['fusar']:

    print('Performing image augmentation on class ' + target + '...')

    src_path = os.path.join(raw_root, target)
    image_path_list = glob.glob(os.path.join(src_path, '*'))

    prob = class_aug_len_per_image[target] - math.floor(class_aug_len_per_image[target])

    for image_path in image_path_list:
        
        # # Read an image with OpenCV
        image = cv2.imread(image_path)   

        # image = center_crop(image)
        # scale_percent = 50
        # width = int(image.shape[1] * scale_percent / 100)
        # height = int(image.shape[0] * scale_percent / 100)

        dim = (128, 128)
        resized_image = cv2.resize(image, dim)

        output_sub_folder = output_root + "/" + target + "/"

        if not os.path.isdir(output_sub_folder):
            os.makedirs(output_sub_folder)

        cv2.imwrite(output_sub_folder + image_path.split('\\')[-1], resized_image)      

        aug_count = 0

        if math.floor(class_aug_len_per_image[target]) >= 1:
            for j in range(1, math.floor(class_aug_len_per_image[target]) + 1):
                # Augment an image
                transformed = transform(image=image)
                transformed_image = transformed["image"]
                resized_image = cv2.resize(transformed_image, dim)

                aug_filename = image_path.split('\\')[-1].split('.tiff')[0] + '_aug_' + str(j) + '.tiff'
                cv2.imwrite(output_sub_folder + aug_filename, resized_image)

                aug_count = aug_count + 1
        
        random_value = random.uniform(0, 1)

        if random_value <= prob:
            # Augment an image
            transformed = transform(image=image)
            transformed_image = transformed["image"]
            resized_image = cv2.resize(transformed_image, dim)

            aug_filename = image_path.split('\\')[-1].split('.tiff')[0] + '_aug_' + str(aug_count + 1) + '.tiff'
            cv2.imwrite(output_sub_folder + aug_filename, resized_image)

print("\nSplitting dataset into train-test sets...")

output_split_root = output_root + '_split'

splitfolders.ratio(output_root, output=output_split_root, seed=1337, ratio=(0.8, 0.2))        # Split the dataset into train-val-test

# print("Deleting unsplit data directory...")
# shutil.rmtree(output_root)

