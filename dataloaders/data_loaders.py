import torch
import torchvision.transforms as transforms
from dataloaders import StanfordDogsDataset as StanfordDogs
from torch.utils.data import DataLoader

def get_data(data_dir, image_size, split_ratio):
    # Instantiate custom pytorch dataset

    # pretrained data transforms so data is useable by pretained model
    pretrained_transforms = transforms.Compose([transforms.Resize(image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # data transforms for custom model
    custom_data_transforms = transforms.Compose([transforms.Resize(image_size),
                                                 transforms.ToTensor()])

    full_dataset = StanfordDogs.StanfordDogsDataset(data_dir, pretrained_transforms)

    def make_train_test_data(data_set, split_ratio):
        # Test size based on train size
        # Create train and test datasets
        train_size = int(split_ratio * len(data_set))
        test_size = len(data_set) - train_size

        # Add seed to random_split() to get reproducible results, generator=torch.Generator().manual_seed(42)
        # random_split() creates non-overlapping data
        train_data, test_data = torch.utils.data.dataset.random_split(data_set, [train_size, test_size])

        return train_data, test_data

    train_data, test_data = make_train_test_data(full_dataset, split_ratio)
    print("Train data size: {}, Test data size: {}".format(len(train_data), len(test_data)))
    return train_data, test_data



def make_data_loaders(train_dataset, test_dataset, batch_size):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader
