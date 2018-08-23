import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import torch.utils.data as data
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Data directory filepath')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--gpu', action='store_true', help='Choose training model on gpu if available')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')
args, _ = parser.parse_known_args()

#Define your transforms for the training, validation, and testing sets
if args.data_dir:    
   data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
    
    #Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(root=args.data_dir + '/' + x, transform=data_transforms[x])
        for x in list(data_transforms.keys())
    }
    
    #Using the image datasets and the trainforms, define the dataloaders
   trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
   validloader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=32)
   testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)

if args.arch:
        arch = args.arch 

#load pre-trained model
def pre_train_model(arch='vgg19',hidden_units=4096,output_size=102):
    if arch=='vgg19':
        model = models.vgg19(pretrained=True)
    elif arch=='resnet':
        model = models.resnet18(pretrained=True)
    else:
        break
    for param in model.parameters():
        param.requires_grad = False
        
    features = list(model.classifier.children())[:-1]
    input_size = model.classifier[len(features)].in_features
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p=0.5)),
                             ('fc2', nn.Linear(hidden_units, hidden_units)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p=0.5)),
                                ('fc3', nn.Linear(hidden_units, output_size)),
                            ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model

#train model function
def train_model(image_datasets, arch='vgg19', hidden_units=4096, epochs=10, learning_rate=0.01, gpu=True, checkpoint=''):
    if args.arch:
        arch = args.arch     
    if args.hidden_units:
        hidden_units = args.hidden_units
    if args.epochs:
        epochs = args.epochs
    if args.learning_rate:
        learning_rate = args.learning_rate
    if args.gpu:
        gpu = args.gpu
    if args.checkpoint:
        checkpoint = args.checkpoint   
    
    output_size = len(image_datasets['train'].classes)
    model = pre_train_model(arch=arch,hidden_units=hidden_units,output_size=output_size)
    
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
    
    #Implement a function for the validation pass
    def validation(model, validloader, criterion):
        valid_loss = 0
        accuracy = 0

        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            valid_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return valid_loss, accuracy
   
    steps = 0
    running_loss = 0
    print_every = 40
    
    for e in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                model.train()
    #save the model as a checkpoint
    state_dict=model.state_dict()
    model.load_state_dict(state_dict)
    model.class_to_idx = image_datasets['train'].class_to_idx
    if checkpoint: 
        checkpoint_dict = { 'model_arch': arch,
            'hidden_units': hidden_units,
            'epochs': epochs,
            'optimizer_state':optimizer.state_dict,
            'class_to_idx': model.class_to_idx, 
            'state_dict': model.state_dict(),
        }
        torch.save(checkpoint_dict, checkpoint)
                    
    return model
