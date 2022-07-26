import copy
import time

import torch

# pass dataloaders as dictionary where: dataloaders{ train: train_data, val: val_data}
import wandb


def train_val_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device=None):
	since = time.time()

	val_acc_history = []

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	best_train_loss = 0.0
	best_val_loss = 0.0
	num_train_examples = 0
	num_val_examples = 0

	for epoch in range(num_epochs):
		print("Epoch {}/{}".format(epoch + 1, num_epochs))
		print("-" * 10)
		wandb.log({"epoch": epoch + 1})

		for phase in ["train", "val"]:
			if phase == "train":
				model.train()
			else:
				model.eval()

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

						# logging
						num_train_examples += inputs.shape[0]
						wandb.log({f"train examples seen": num_train_examples, "epoch": epoch + 1})
						wandb.log({"train loss": loss})

					else:
						outputs = model(inputs)
						loss = criterion(outputs, labels)

						# logging
						num_val_examples += inputs.shape[0]
						wandb.log({f"val examples seen": num_val_examples, "epoch": epoch + 1})
						wandb.log({"val loss": loss})

					_, predictions = torch.max(outputs, 1)

					if phase == "train":
						# Backward pass
						loss.backward()

						# Optimize model
						optimizer.step()

				running_loss += loss.item() * inputs.size(0)

				running_corrects += torch.sum(predictions == labels.data)

				wandb.log({f"{phase} running loss": running_loss,
						   f"{phase} running correct predictions": running_corrects,
						   "epoch": epoch + 1})
			if phase == "train":
				scheduler.step()

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

			# Log stuff
			wandb.log({f"{phase} epoch loss": epoch_loss,
					   f"{phase} epoch accuracy": epoch_acc,
					   "epoch": epoch + 1})

			print("{} loss: {:.4f} accuracy: {:.4f}".format(phase, epoch_loss, epoch_acc))


			# deep copy model
			if phase == "val" and epoch_acc > best_acc:
				best_acc = epoch_acc
				print("new best accuracy achieved: {:.4f}".format(best_acc))
				wandb.log({"new best accuracy": best_acc, "epoch": epoch})
				best_model_wts = copy.deepcopy(model.state_dict())
			if phase == "val":
				val_acc_history.append(epoch_acc)

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	wandb.log({"best accuracy achieved": best_acc})

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, val_acc_history


