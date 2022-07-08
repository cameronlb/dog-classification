import torch

model.eval()

# Run the model on some test examples
with torch.no_grad():
	correct, total = 0, 0
	for images, labels in test_loader:
		images, labels = images.to(device), labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	print(f"Accuracy of the model on the {total} " +
		  f"test images: {100 * correct / total}%")

	wandb.log({"test_accuracy": correct / total})

# Save the model in the exchangeable ONNX format
torch.onnx.export(model, images, "model.onnx")
torch.save(model.state_dict(), "model_state_dict.pth")