from torch import optim, nn


def initialize_model(model, num_classes, config=None):
    # disable all gradients in model to false, so no training occurs on those layers
    for param in model.parameters():
        param.requires_grad = False

    # change fully connected layer of pretrained model
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    # Loop to find params/layers of model that have gradients set to true/active
    params_to_update = model.parameters()
    print("Params to learn: ")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

    if config == None:
        # Default optimizer
        print("No optimizer found in config, using default: SGD")
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    elif config.optimizer == "SGD":
        # pass in params/layers to optim to optimization only
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    elif config.optimizer == "ADAM":
        optimizer = optim.Adam(params_to_update, lr=0.001)

    return model, optimizer