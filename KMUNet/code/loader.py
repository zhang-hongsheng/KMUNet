from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
import os
from PIL import Image
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation
import torchvision.transforms.functional
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import ndimage
from utils import *
from torchvision import transforms


class My_Train_Transforms:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=360),
        ])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        if random.random() < 0.5:
            image = transforms.functional.vflip(image)
            label = transforms.functional.vflip(label)

        angle = random.randint(0, 360)
        image = transforms.functional.rotate(image, angle)
        label = transforms.functional.rotate(label, angle)

        image = np.array(image)
        image = image.transpose(2, 0, 1)
        label = np.array(label)
        label = np.where(label > 127, 1, 0)

        return {'image': image, 'label': label, 'filename': sample['filename']}


class My_Test_Transforms:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = self.transform(image)  # 应用变换
        return {'image': image, 'label': label, 'filename': sample['filename']}


class BaseDataSets_ISIC(Dataset):
    def __init__(self, base_dir=None, split="train", transform=None, num=None, ops_weak=None, ops_strong=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.transform = transform

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        # 加载数据集
        if self.split == "train":
            for filename in os.listdir(os.path.join(self._base_dir, 'train_images')):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    self.sample_list.append(filename)

        elif self.split == "val":
            for filename in os.listdir(os.path.join(self._base_dir, 'val_images')):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    self.sample_list.append(filename)

        elif self.split == "test":
            for filename in os.listdir(os.path.join(self._base_dir, 'test_images')):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    self.sample_list.append(filename)

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            image_path = os.path.join(self._base_dir, 'train_images', case)
            label_path = os.path.join(self._base_dir, 'train_labels', case)

        elif self.split == "val":
            image_path = os.path.join(self._base_dir, 'val_images', case)
            label_path = os.path.join(self._base_dir, 'val_labels', case)

        elif self.split == "test":
            image_path = os.path.join(self._base_dir, 'test_images', case)
            label_path = os.path.join(self._base_dir, 'test_labels', case)

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        sample = {'image': image, 'label': label, 'filename': case}

        sample = self.transform(sample)
        return sample


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split="train", num=None, ops_weak=None, ops_strong=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            for filename in os.listdir(self._base_dir + '/train_images'):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    self.sample_list.append(filename)

        elif self.split == "val":
            for filename in os.listdir(self._base_dir + '/val_images'):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    self.sample_list.append(filename)

        elif self.split == "test":
            for filename in os.listdir(self._base_dir + '/test_images'):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    self.sample_list.append(filename)

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            image_path = os.path.join(self._base_dir, 'train_images', f"{case}")
            label_path = os.path.join(self._base_dir, 'train_labels', f"{case}")

        elif self.split == "val":
            image_path = os.path.join(self._base_dir, 'val_images', f"{case}")
            label_path = os.path.join(self._base_dir, 'val_labels', f"{case}")

        elif self.split == "test":
            image_path = os.path.join(self._base_dir, 'test_images', f"{case}")
            label_path = os.path.join(self._base_dir, 'test_labels', f"{case}")

        image = np.array(Image.open(image_path).convert('RGB'))
        image = np.transpose(image, (2, 0, 1))
        label = Image.open(label_path).convert('L')
        label = np.array(label)
        label = np.where(label > 127, 1, 0)

        sample = {'image': image, 'label': label, 'filename': case}
        return sample


class BaseDataSets_BCSS(Dataset):
    def __init__(self, base_dir=None, split="train", num=None, transform=lambda x: x, ops_weak=None, ops_strong=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.transform = transform

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            for filename in os.listdir(self._base_dir + '/train_images'):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    self.sample_list.append(filename)

        elif self.split == "val":
            for filename in os.listdir(self._base_dir + '/val_images'):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    self.sample_list.append(filename)

        elif self.split == "test":
            for filename in os.listdir(self._base_dir + '/test_images'):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    self.sample_list.append(filename)

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            image_path = os.path.join(self._base_dir, 'train_images', f"{case}")
            label_path = os.path.join(self._base_dir, 'train_labels', f"{case}")

        elif self.split == "val":
            image_path = os.path.join(self._base_dir, 'val_images', f"{case}")
            label_path = os.path.join(self._base_dir, 'val_labels', f"{case}")

        elif self.split == "test":
            image_path = os.path.join(self._base_dir, 'test_images', f"{case}")
            label_path = os.path.join(self._base_dir, 'test_labels', f"{case}")

        image = np.array(Image.open(image_path).convert('RGB'))
        image = np.transpose(image, (2, 0, 1))
        label = Image.open(label_path).convert('L')
        label = np.array(label)
        label[label == 29] = 0
        label[label == 150] = 1
        label[label == 76] = 2
        label[label == 255] = 3
        label[label == 75] = 4

        sample = {'image': image, 'label': label, 'filename': case}
        sample = self.transform(sample)
        return sample


class TestDataSets(Dataset):
    def __init__(self, base_dir=None, split="train", num=None, transform=lambda x: x, ops_weak=None, ops_strong=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.transform = transform

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        for filename in os.listdir(self._base_dir):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                self.sample_list.append(filename)

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image_path = os.path.join(self._base_dir, f"{case}")

        image = np.array(Image.open(image_path).convert('RGB'))
        image = np.transpose(image, (2, 0, 1))

        sample = {'image': image, 'name': case}
        sample = self.transform(sample)
        return sample


class Medical_dataset(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=lambda x: x,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.transform = transform

        for sub_dir in ['LUAD', 'LUSC']:
            image_dir = os.path.join(self._base_dir, self.split, sub_dir, 'images')
            for filename in os.listdir(image_dir):
                self.sample_list.append(f"{sub_dir}+{filename}")

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        sub_dir, filename = case.split('+')

        image_path = os.path.join(self._base_dir, self.split, sub_dir, 'images', filename)
        label_path = os.path.join(self._base_dir, self.split, sub_dir, 'labels', filename)

        image = np.array(Image.open(image_path).convert('RGB'))
        image = np.transpose(image, (2, 0, 1))
        label = Image.open(label_path).convert('L')
        label = np.array(label)
        if sub_dir == 'LUAD':
            label = np.where(label > 127, 1, 0)
        elif sub_dir == 'LUSC':
            label = np.where(label > 127, 2, 0)
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        return sample


class isic_loader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train=True, Test=False):
        super(isic_loader, self)
        self.train = train
        if train:
          self.data   = np.load(path_Data+'data_train.npy')
          self.mask   = np.load(path_Data+'mask_train.npy')
        else:
          if Test:
            self.data   = np.load(path_Data+'data_test.npy')
            self.mask   = np.load(path_Data+'mask_test.npy')
          else:
            self.data   = np.load(path_Data+'data_val.npy')
            self.mask   = np.load(path_Data+'mask_val.npy')          
        
        self.data   = dataset_normalized(self.data)
        self.mask   = np.expand_dims(self.mask, axis=3)
        self.mask   = self.mask/255.

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_rot_flip(img, seg)
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)
        
        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        img = img.permute( 2, 0, 1)
        seg = seg.permute( 2, 0, 1)

        return img, seg
    
    def random_rot_flip(self,image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def random_rotate(self,image, label):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label

    def __len__(self):
        return len(self.data)
