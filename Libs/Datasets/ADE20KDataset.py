from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np
import torch


class ADE20KDataset(Dataset):
    """Class for ADE20K dataset."""

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
        self.image_dir = os.path.join(CONFIG['DATASET']['ROOT'], 'ADEChallengeData2016')
        self.set = set.lower()
        self.mode = mode.lower()
        # Set boolean variable of ten crops for validation
        self.ten_crop = tencrops

        # Decode dataset scene categories
        self.classes = list()
        class_file_name = os.path.join(self.image_dir, "Scene_Names.txt")

        with open(class_file_name) as class_file:
            for line in class_file:
                self.classes.append(line.split()[0])

        self.nclasses = self.classes.__len__()

        # Create list for filenames and scene ground-truth labels
        self.filenames = list()
        self.labels = list()
        self.labelsindex = list()
        filenames_file = os.path.join(self.image_dir, "sceneCategories_" + self.set + ".txt")

        with open(filenames_file) as class_file:
            for line in class_file:
                name, label = line.split()
                self.filenames.append(name)
                self.labels.append(label)
                self.labelsindex.append(self.classes.index(label))

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
        img_name = os.path.join(self.image_dir, "images", self.set, (self.filenames[idx] + ".jpg"))
        img = Image.open(img_name)

        # Convert it to RGB if gray-scale
        if img.mode is not "RGB":
            img = img.convert("RGB")

        img = self.transform(img)

        # Create dictionary
        self.sample = {'Images': img, 'Labels': self.classes.index(self.labels[idx])}

        return self.sample
