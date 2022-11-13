from absl import logging
from absl import flags
from absl import app

from tqdm import tqdm

from torch.utils import tensorboard

import torchvision
import torch

import numpy as np

import json
import os

from data import preprocess
from data import loader
from utils import common
import model

from matplotlib import pyplot as plt

flags.DEFINE_string('experiments_path', os.path.join(common.project_root, 'experiments'), help='')
flags.DEFINE_string('config_name', 'config/AConvNet-FUSAR.json', help='')
FLAGS = flags.FLAGS


common.set_random_seed(12321)


def load_dataset(path, is_train, name, batch_size):
    transform = [preprocess.CenterCrop(88), torchvision.transforms.ToTensor()]
    if is_train:
        transform = [preprocess.RandomCrop(88), torchvision.transforms.ToTensor()]
    _dataset = loader.Dataset(
        path, name=name, is_train=is_train,
        transform=torchvision.transforms.Compose(transform)
    )
    data_loader = torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=is_train, num_workers=1
    )
    return data_loader


@torch.no_grad()
def validation(m, ds):
    num_data = 0
    corrects = 0

    # Test loop
    m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(tqdm(ds)):
        images, labels = data

        predictions = m.inference(images)
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(m.device)).sum().item()

    accuracy = 100 * corrects / num_data
    return accuracy

# def inference(m, file):

#     m.net.eval()
#     _softmax = torch.nn.Softmax(dim=1)



#     predictions = m.inference(image)
#     predictions = _softmax(predictions)
#     _, predictions = torch.max(predictions.data, 1)
#     label = label.type(torch.LongTensor)

#     return accuracy


def run(epochs, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, experiments_path=None):

    m = model.Model(
        classes=classes, dropout_rate=dropout_rate, channels=channels,
        lr=lr, lr_step=lr_step, lr_decay=lr_decay,
        weight_decay=weight_decay
    )

    model_path = os.path.join(experiments_path, f'model/{model_name}')

    try:
        print('Loading pretrained model...')
        m.load(os.path.join(model_path, 'old', 'model-061.pth'))
        print('Pretrained model loaded...')
    except:
        print('Pretrained model not found...')
        return

def main(_):
    logging.info('Start')
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    config = common.load_config(os.path.join(experiments_path, config_name))

    dataset = config['dataset']
    classes = config['num_classes']
    channels = config['channels']
    epochs = config['epochs']
    batch_size = config['batch_size']

    lr = config['lr']
    lr_step = config['lr_step']
    lr_decay = config['lr_decay']

    weight_decay = config['weight_decay']
    dropout_rate = config['dropout_rate']

    model_name = config['model_name']

    image_path = ''

    run(epochs, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, experiments_path)

    logging.info('Finish')


if __name__ == '__main__':
    app.run(main)
