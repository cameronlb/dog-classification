import torch
import torch.utils.data
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import plotly.express as px
import numpy as np

##### Custom imports #####
import image_utils
from data_loader.StanfordDogsDataset import StanfordDogsDataset
from models.DogBreedClassifier import DogBreedClassifier

##########################

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"

data_transforms = transforms.Compose([transforms.Resize((224, 224)),
								transforms.ToTensor(),
								transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
								])


dataset = StanfordDogsDataset(DATA_DIR, data_transforms)
label_names = dataset.breed_names
num_classes = len(label_names)

train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

images, labels = next(iter(train_data_loader))

print(images.shape)

##### LOAD MODEL #####
model = torch.jit.load("trained_model_scripted.pt")
# model = DogBreedClassifier()
model.load_state_dict(torch.load("model_state_dict.pth"))
model.to("cpu")

model.eval()

def image_to_input(img_path):
	"""function to convert image path to input for model"""
	pil_img = Image.open(img_path)

	fig = px.imshow(pil_img)
	fig.show()

	img_transforms = transforms.Compose([transforms.Resize((224, 224)),
									 transforms.ToTensor(),
									 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
									 ])

	pil_img = img_transforms(pil_img)

	print(pil_img[0].shape)


	input_tensor = pil_img.unsqueeze(0)
	print(input_tensor.shape)

	return input_tensor

model_input = image_to_input("images/dudley_with_sticks.jpg")

soft_max = nn.Softmax(dim=1)

model_output = model(model_input)
model_output = soft_max(model_output)
_, model_predictions = torch.max(model_output, 1)

print(model_predictions)

for idx, breed in enumerate(label_names):
	prediction = model_output[0, idx]
	print(f"{breed}, predicted with {prediction}.")

prediction = np.argmax(model_predictions[0])
print(prediction)
print(model_predictions[0, prediction])
print(label_names[prediction])