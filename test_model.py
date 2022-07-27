import torch
from torchvision import transforms
from PIL import Image

from data_loader.StanfordDogsDataset import StanfordDogsDataset
from models.DogBreedClassifier import DogBreedClassifier



DATA_DIR = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"
dog_dataset = StanfordDogsDataset(DATA_DIR)
breed_labels = dog_dataset.breed_names

model = DogBreedClassifier()
model.load_state_dict(torch.load("model_state_dict.pth"))
print(model)
model = model.eval()

input_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

img = Image.open("images/dudley_standing_field_landscape.jpg")

# apply image transforms for model input
img_transformed = input_transforms(img)

# add batch dimension for model input shape
img_transformed = img_transformed.unsqueeze(0)
with torch.no_grad():
    output = model(img_transformed)
    print(output.shape)
    _, predictions = torch.max(output, 1)

    print(predictions.item())
    print(breed_labels)
    print(breed_labels[predictions.item()])

