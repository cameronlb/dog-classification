import torch

from models.DogBreedClassifier import DogBreedClassifier




model = DogBreedClassifier()
model.load_state_dict(torch.load("model_state_dict.pth"))

print(model)