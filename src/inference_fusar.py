import matplotlib.pyplot as plt

import numpy as np

import json
import glob
import sys
import os

from tqdm import tqdm
import torchvision
import torch

from utils import common
from data import preprocess
from data import loader
import model

from sklearn import metrics
from data import fusar

import matplotlib.pyplot as plt
import seaborn as sns

import cv2

def center_crop(data, size=88):
    h, w, _ = data.shape

    y = (h - size) // 2
    x = (w - size) // 2

    return data[y: y + size, x: x + size]

def infer(_m, image):

    _m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)

    prediction = _m.inference(image)
    prediction = _softmax(prediction)

    pred_acc, pred_label = torch.max(prediction.data, 1)

    return pred_acc.cpu().tolist()[0], pred_label.cpu().tolist()[0]

config = common.load_config(os.path.join(common.project_root, 'experiments/config/AConvNet-FUSAR.json'))
model_name = config['model_name']

_fusar = fusar.FUSAR(
    name='fusar', is_train=False, chip_size=128, patch_size=128, use_phase=False, stride=1
)

# Load in image and label here

image_path = os.path.join(common.project_root, 'dataset\\fusar\\sample\\Fishing\\bulkcarrier_and_fishing_combined_2.jpg')

# Read in the image
label_dict, image = _fusar.read_inference(image_path)

# Display image
plt.imshow(np.squeeze(image[0], axis=(2,)), interpolation='nearest')
plt.show()

gt_label = label_dict['class_id']
gt_label_name = label_dict['target_type']

image = center_crop(image[0])

image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
image = image_transform(image)
image = image[None, :, :, :]

m = model.Model(
    classes=config['num_classes'], channels=config['channels'],
)

model_path = os.path.join(common.project_root, f'experiments/model/{model_name}/old/model-061.pth')

m.load(model_path)
pred_acc, pred_label = infer(m, image)

print('-RESULT-\n')
print('Ground Truth Label: ' + gt_label_name + ' (' + str(gt_label) + ')')

pred_label_name = fusar.serial_number_inv_map[pred_label]
print('Predicted Label: ' + pred_label_name + ' (' + str(pred_label) + ')')
print('Predicted Accuracy: ' + str(pred_acc))