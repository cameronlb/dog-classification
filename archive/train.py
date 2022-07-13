import torch
import wandb
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, loss_fn, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_fn, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(dataloader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        print(f'Epoch {epoch}/{config.epochs - 1}')
        print('-' * 10)

        model.train()

        for _, (images, labels) in enumerate(dataloader):
            loss = train_batch(images, labels, model, optimizer, loss_fn)
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(model, loss, example_ct, epoch)

def train_batch(images, labels, model, optimizer, loss_fn):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    model.cuda()

    # Forward pass ➡
    outputs = model(images)
    loss = loss_fn(outputs, labels)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log(model, loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    wandb.watch(model)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
