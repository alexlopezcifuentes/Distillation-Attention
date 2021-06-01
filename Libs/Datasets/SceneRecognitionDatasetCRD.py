import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

SceneRecognitionDataset.py
Python file to create the SceneRecogntionDatasetCRD class. This is the class used for all the three Scene
Recognition Datasets (ADE20K, SUN397, MIT97) when CRD is used.

It is itented to be a combination from the SceneRecogntionDataset class and the original CIFAR 10 implementation. 
For more details refer to the original implementation by the original author
https://github.com/HobbitLong/RepDistiller

Partially developed by Anonymous Code Author.
"""

class SceneRecognitionDatasetCRD(Dataset):
    """Class for ADE20K dataset."""

    def decodeADE20K(self, filenames_file):
        with open(filenames_file) as class_file:
            for line in class_file:
                name, label = line.split()
                self.filenames.append(name)
                self.labels.append(label)
                self.labelsindex.append(self.classes.index(label))

    def decodeMIT67(self, filenames_file):
        with open(filenames_file) as class_file:
            for line in class_file:
                _, label, name = line.split('/')
                self.filenames.append(os.path.join(label, name[:name.find('.jpg')]))
                self.labels.append(label)
                self.labelsindex.append(self.classes.index(label))

    def decodeSUN397(self, filenames_file):
        with open(filenames_file) as class_file:
            for line in class_file:
                _, label, name = line.split('/')
                self.filenames.append(os.path.join(label, name[:name.find('.jpg')]))
                self.labels.append(label)
                self.labelsindex.append(self.classes.index(label))

    def __init__(self, CONFIG, set, mode, tencrops=False):
        """
        Intialiaze dataset
        :param CONFIG: Configuration file
        :param root_dir: Root dir of data
        :param set: Set that is used
        :param mode: Mode to use the set, either train or val
        :param tencrops: Boolean to use or not ten crop evaluation
        """
        # Extract main path and set (Train or Val)
        self.image_dir = CONFIG['DATASET']['ROOT']
        self.set = set.lower()
        self.mode = mode.lower()
        # Set boolean variable of ten crops for validation
        self.ten_crop = tencrops

        # Decode dataset scene categories
        self.classes = list()
        class_file_name = os.path.join(self.image_dir, "scene_names.txt")

        with open(class_file_name) as class_file:
            for line in class_file:
                self.classes.append(line.split()[0])

        self.nclasses = self.classes.__len__()

        # Create list for filenames and scene ground-truth labels
        self.filenames = list()
        self.labels = list()
        self.labelsindex = list()
        filenames_file = os.path.join(self.image_dir, self.set + ".txt")

        if CONFIG['DATASET']['NAME'] == 'ADE20K':
            self.decodeADE20K(filenames_file)
        elif CONFIG['DATASET']['NAME'] == 'MIT67':
            self.decodeMIT67(filenames_file)
        elif CONFIG['DATASET']['NAME'] == 'SUN397':
            self.decodeSUN397(filenames_file)

        # Control Statements for data loading
        assert len(self.filenames) == len(self.labels)

        # ----------------------------- #
        #    Pytorch Transformations    #
        # ----------------------------- #
        self.mean = CONFIG['DATASET']['MEAN']
        self.STD = CONFIG['DATASET']['STD']
        self.resizeSize = CONFIG['MODEL']['RESIZE']
        self.outputSize = CONFIG['MODEL']['CROP']

        if self.mode == 'train':
            # Train Set Transformation
            self.transform = transforms.Compose([
                transforms.Resize(self.resizeSize),
                transforms.RandomCrop(self.outputSize),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.STD)
            ])
        else:
            if not self.ten_crop:
                self.transform = transforms.Compose([
                    transforms.Resize(self.resizeSize),
                    transforms.CenterCrop(self.outputSize),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.STD)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(self.resizeSize),
                    transforms.TenCrop(self.outputSize),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(self.mean, self.STD)(crop) for crop in crops])),
                ])

        # ----------------------------- #
        #           CRD Stuff           #
        # ----------------------------- #
        self.k = CONFIG['DISTILLATION']['K']

        # List cls_positive of num_classes elements
        # Get the samples for each class. Class[0] = [35, 45, 69]
        self.cls_positive = [[] for i in range(self.nclasses)]
        for i in range(self.nclasses):
            self.cls_positive[self.labelsindex[i]].append(i)

        # List cls_negative of num_classes elements
        # For each class, list of samples not corresponding to it.
        self.cls_negative = [[] for i in range(self.nclasses)]
        for i in range(self.nclasses):
            for j in range(self.nclasses):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        # Convert the list to arrays
        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(self.nclasses)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(self.nclasses)]

        # Again convert the list to array
        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __len__(self):
        """
        Function to get the size of the dataset
        :return: Size of dataset
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """

        """

        # Get RGB image path and load it
        img_name = os.path.join(self.image_dir, self.set, (self.filenames[idx] + ".jpg"))
        img = Image.open(img_name)

        # Convert it to RGB if gray-scale
        if img.mode is not "RGB":
            img = img.convert("RGB")

        img = self.transform(img)

        label = self.classes.index(self.labels[idx])

        # Create dictionary
        sample = {'Images': img, 'Labels': label}

        # CRD Stuff
        pos_idx = idx
        replace = True if self.k > len(self.cls_negative[label]) else False
        # Select randomly "k" elements from all the negative samples for class "target"
        neg_idx = np.random.choice(self.cls_negative[label], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

        crd_info = {'Idx': idx, 'Sample Idx': sample_idx}
        sample.update(crd_info)

        return sample
