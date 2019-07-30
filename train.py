# Imports here
import matplotlib.pyplot as plt
import numpy as np
import torch
import PIL
import seaborn as sns

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import json
from get_input_args import get_input_args

def main():
    in_arg = get_input_args()
    # Function that checks command line arguments using in_arg  
    #check_command_line_arguments(in_arg)
    #data_dir = 'flowers'
    data_dir = in_arg.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    #import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # TODO: Build and tain your network
    
    vgg = "vgg16"
    if vgg == in_arg.arch: 
        model = models.vgg16(pretrained=True)
        v1 = 25088
        v2 = 10000
        v3 = in_arg.h_lay
    else:
        model = models.densenet161(pretrained=True)
        v1 = 2208
        v2 = 1000
        v3 = in_arg.h_lay
        #print("Sorry, Model Trained only for vgg16")

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(in_arg.device) 
    model.to(device) 
    #print(device)

    # Turn off gradients
    for param in model.parameters():
        param.requires_grad = False
    
    # Define new classifier
    classifier = nn.Sequential(nn.Linear(v1, v2),
                          nn.ReLU(), 
                          nn.Dropout(0.2), 
                          nn.Linear(v2, v3),
                          nn.ReLU(), 
                          nn.Dropout(0.2),                          
                          nn.Linear(v3, 102),                                                     
                          nn.LogSoftmax(dim=1))    
                            
    model.classifier = classifier    
    model.to(device) 
  
    lr = in_arg.lr_rate
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    print(device)
    
    # Train the model
    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device          
            inputs, labels = inputs.to(device), labels.to(device)         
            optimizer.zero_grad()        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
                
 # Check the loss on the Test dataset       
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:                        
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)                    
                        valid_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"valid loss: {valid_loss/len(validloader):.3f}.. "
                        f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
            
    # TODO: Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': v1,
                 'hidden_layer_size': v3,
                 'output_size': 102,
                 'classifier' : model.classifier,
                 'state_dict': model.state_dict(),
                 'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'modelcheckpoint.pth')
    print("Training finished")

# Call to main function to run the program
if __name__ == "__main__":
    main()   