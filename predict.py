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
    image_path = in_arg.dir
    
    #image_path = "flowers/test/100/image_07896.jpg"
    # Load the model with the loading function
    file_name = 'modelcheckpoint.pth'
    #saved_model
    model = load_model(file_name)  
    probs, classes = predict(image_path, model)
    print(probs)
    print(classes)
    
def get_input_args():
    parser = argparse.ArgumentParser()    
    # Create command line arguments  
    parser = argparse.ArgumentParser(description='Predict the model')
    parser.add_argument('--dir', type=str, help='Give path to the image, Eg:flowers/test/100/image_07896.jpg')
    return parser.parse_args()

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(filepath):            
     # Load the model and force the tensors to be on the CPU
     checkpoint = torch.load(filepath)
     model = models.vgg16(pretrained=True)   
     model.classifier = checkpoint['classifier']
     model.load_state_dict(checkpoint['state_dict'])
     model.class_to_idx = checkpoint['class_to_idx']      
      
     # Freeze the parameters so we dont backpropagate through them
     for param in model.parameters():
         param.requires_grad = False    
     return model  
   
def process_image(image):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_name = 'modelcheckpoint.pth'
    model = load_model(file_name)
    print(model)
    model.to(device) 
    
    # TODO: Process a PIL image for use in a PyTorch model
    test_image = PIL.Image.open(image)

    # Get original dimensions
    orig_width, orig_height = test_image.size

    # Find shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]        
    test_image.thumbnail(size=resize_size)

    # Find pixels 
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(test_image)/255 
   
    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)      
    return np_image    
    
def predict(image_path, model, topk=3):     
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to('cpu')
    #model.to(device)
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(np.array([image])).float()
   
    logps = model(image)
    ps = torch.exp(logps)    
    p, classes = ps.topk(topk, dim=1)
    top_p = p.tolist()[0]
    top_classes = classes.tolist()[0]    
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    
    #import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    labels = []
    for c in top_classes:
        labels.append(cat_to_name[idx_to_class[c]])
    return top_p, labels    
       
# Call to main function to run the program
if __name__ == "__main__":
    main()                          
    