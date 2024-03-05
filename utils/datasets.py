import os
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, data_dir, data_name, transform=None, train_mode=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train_mode = train_mode
        file_path = Path(f'{self.data_dir}/{data_name}/files.txt')
        if not file_path.exists():
            self.load_label(f'{self.data_dir}/{data_name}/')
        self.samples = open(file_path).read().splitlines()

    def __getitem__(self, idx):
        image = Image.open(f'{self.samples[idx]}.jpg')
        image = image.convert('RGB')
        label = sio.loadmat(f'{self.samples[idx]}.mat')
        pt2d = label['pt2d']
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        if self.train_mode:
            k = np.random.random_sample() * 0.2 + 0.2
            x_min -= 0.6 * k * abs(x_max - x_min)
            y_min -= 2 * k * abs(y_max - y_min)
            x_max += 0.6 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)
        else:
            k = 0.20
            x_min -= 2 * k * abs(x_max - x_min)
            y_min -= 2 * k * abs(y_max - y_min)
            x_max += 2 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)

        image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        pre_pose_params = label['Pose_Para'][0]
        pose = pre_pose_params[:3]
        pitch, yaw, roll = pose[0], pose[1], pose[2]

        rnd = np.random.random_sample()
        if self.train_mode and rnd < 0.5:
            yaw = -yaw
            roll = -roll
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        rnd = np.random.random_sample()

        if self.train_mode and rnd < 0.05:
            image = image.filter(ImageFilter.BLUR)

        if self.transform is not None:
            image = self.transform(image)

        if self.train_mode:
            return image, torch.FloatTensor(self.get_rotation(pitch, yaw, roll))
        else:
            return image, torch.FloatTensor([pitch, yaw, roll])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_rotation(x, y, z):
        rotate_x = np.array([[1, 0, 0],
                             [0, np.cos(x), -np.sin(x)],
                             [0, np.sin(x), np.cos(x)]])
        # y
        rotate_y = np.array([[np.cos(y), 0, np.sin(y)],
                             [0, 1, 0],
                             [-np.sin(y), 0, np.cos(y)]])
        # z
        rotate_z = np.array([[np.cos(z), -np.sin(z), 0],
                             [np.sin(z), np.cos(z), 0],
                             [0, 0, 1]])

        rotation = rotate_z.dot(rotate_y.dot(rotate_x))
        return rotation

    @staticmethod
    def load_label(data_dir):
        f_counter, rej_counter = 0, 0
        file = open(f'{data_dir}files.txt', 'w')

        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f[-4:].lower().endswith('.jpg'):
                    mat_path = os.path.join(root, f.replace('.jpg', '.mat'))
                    mat = sio.loadmat(mat_path)
                    pre_pose_ = mat['Pose_Para'][0]
                    pose = pre_pose_[:3]
                    pitch = pose[0] * 180 / np.pi
                    yaw = pose[1] * 180 / np.pi
                    roll = pose[2] * 180 / np.pi
                    if all(abs(angle) <= 99 for angle in (pitch, yaw, roll)):
                        if f_counter > 0:
                            file.write('\n')
                        file.write(os.path.join(root, f[:-4]))
                        f_counter += 1
                    else:
                        rej_counter += 1

# data_dir = '../../../Datasets/HPE'
# Datasets(data_dir, transform=None)[0]
