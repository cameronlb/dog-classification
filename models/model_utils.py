from torch import optim, nn


def initialize_feature_extractor(model):
    """Function to initialize pretrained model as fixed feature extractor"""
    print(model)
    num_of_params = 0
    # efficient-net 213 params
    for param in model.parameters():
        num_of_params += 1
        if num_of_params > 150:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_params_to_train = num_of_params // 4
    print(f"number of params/layers: {num_of_params}")
    print(f"number of params/layers to train: {num_params_to_train}")

    return model

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

