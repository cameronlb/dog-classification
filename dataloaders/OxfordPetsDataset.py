import os
import utils
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

# Dataset class for pytorch DataLoaders
class OxfordPetsDataset(Dataset):
    def __init__(self, img_root, transform, pretrained=None):
        self.img_root = img_root
        # file names to open images
        self.img_fnames = [f_name for f_name in os.listdir(self.img_root) if f_name.endswith('.jpg')]

        # classes corresponding to img via file name,
        # size = number of images
        self.img_labels = [utils.parse_breed(fname) for fname in self.img_fnames]

        # classes of all breeds via file name and sorted for retrieving index in __getitem__,
        # size = number of unique breeds (37)
        self.classes = sorted(set(self.img_labels))

        self.transform = transform

    def get_label_names(self):
        return self.classes

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # opening image from file name and root
        img_path = os.path.join(self.img_root, self.img_fnames[idx])
        image = utils.open_image(img_path)

        # getting the label from the file name and indexing classes
        breed_name = utils.parse_breed(self.img_fnames[idx])
        label = self.classes.index(breed_name)

        # applying transforms
        if self.transform:
            image = self.transform(image)
        return image, label