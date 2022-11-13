from absl import logging
from absl import flags
from absl import app

from multiprocessing import Pool
from PIL import Image
import numpy as np

import json
import glob
import os

import fusar

flags.DEFINE_string('image_root', default='dataset', help='')
flags.DEFINE_string('dataset', default='fusar', help='')
flags.DEFINE_boolean('is_train', default=True, help='')
flags.DEFINE_integer('chip_size', default=100, help='')              # If training, use 100. If testing, use 128.
flags.DEFINE_integer('patch_size', default=94, help='')             # If training, use 94. If testing, use 128.
flags.DEFINE_boolean('use_phase', default=False, help='')

FLAGS = flags.FLAGS

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def data_scaling(chip):
    r = chip.max() - chip.min()
    return (chip - chip.min()) / r


def log_scale(chip):
    return np.log10(np.abs(chip) + 1)


def generate(src_path, dst_path, is_train, chip_size, patch_size, use_phase, dataset):
    if not os.path.exists(src_path):
        return
    if not os.path.exists(dst_path):
        os.makedirs(dst_path, exist_ok=True)
    print(f'Target Name: {os.path.basename(dst_path)}')

    _fusar = fusar.FUSAR(
        name=dataset, is_train=is_train, chip_size=chip_size, patch_size=patch_size, use_phase=use_phase, stride=1
    )

    image_list = glob.glob(os.path.join(src_path, '*'))

    for path in image_list:
        label, _images = _fusar.read(path)

        for i, _image in enumerate(_images):
            name = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(dst_path, f'{name}-{i}.json'), mode='w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=2)

            # _image = log_scale(_image)
            np.save(os.path.join(dst_path, f'{name}-{i}.npy'), _image)
            # Image.fromarray(data_scaling(_image)).convert('L').save(os.path.join(dst_path, f'{name}-{i}.bmp'))


def main(_):
    dataset_root = os.path.join(project_root, FLAGS.image_root, FLAGS.dataset)
    aug_root = os.path.join(dataset_root, 'augmented_split')

    mode = 'train' if FLAGS.is_train else 'val'

    output_root = os.path.join(dataset_root, mode)
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    arguments = [
        (
            os.path.join(aug_root, mode, target),
            os.path.join(output_root, target),
            FLAGS.is_train, FLAGS.chip_size, FLAGS.patch_size, FLAGS.use_phase, FLAGS.dataset
        ) for target in fusar.target_name[FLAGS.dataset]
    ]

    with Pool(10) as p:
        p.starmap(generate, arguments)


if __name__ == '__main__':
    app.run(main)
