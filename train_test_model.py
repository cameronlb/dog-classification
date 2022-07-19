import copy
import time

import torch

# pass dataloaders as dictionary where: dataloaders{ train: train_data, test: test_data}
def train_test_model(model, dataloaders, criterion, optimizer, num_epochs, device=None):
	since = time.time()

	val_acc_history = []

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print("Epoch {}/{}".format(epoch + 1, num_epochs))
		print("-" * 10)

		for phase in ["train", "test"]:
			if phase == "train":
				model.train()
				print(phase)
			else:
				model.eval()
				print(phase)

			running_loss = 0.0
			running_corrects = 0.0

			# Iterate over data
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# Zero the parameter gradients
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == "train"):
					if phase == "train":
						# Forward pass
						outputs = model(inputs)

						# loss function
						loss = criterion(outputs, labels)

						_, predictions = torch.max(outputs, 1)

					else:
						outputs = model(inputs)
						loss = criterion(outputs, labels)

					_, predictions = torch.max(outputs, 1)

					if phase == "train":
						# Backward pass
						loss.backward()

						# Optimize model
						optimizer.step()

				running_loss += loss.item() * inputs.size(0)

				running_corrects += torch.sum(predictions == labels.data)

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

			print("{} loss: {:.4f} accuracy: {:.4f}".format(phase, epoch_loss, epoch_acc))

			# deep copy model
			if phase == "test" and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
			if phase == "test":
				val_acc_history.append(epoch_acc)

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, val_acc_history


