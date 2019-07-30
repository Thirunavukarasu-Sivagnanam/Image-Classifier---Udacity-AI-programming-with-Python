# Imports python modules
import argparse

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments  
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--dir', type=str, action="store", default="flowers", help='Give path to the image')
    parser.add_argument('--arch', type=str, action="store", default="densenet161", help='choose architecture: vgg16 or densenet161')
    parser.add_argument('--lr_rate', type=float, action="store", default="0.001", help='choose learning rate')
    parser.add_argument('--h_lay', type=int, action="store", default="500", help='choose number of hidden layers')
    parser.add_argument('--epochs', type=int, action="store", default="1", help='give number of epochs')
    parser.add_argument('--device', type=str, action="store", default="cuda", help='select gpu(cuda) or cpu')
    
    return parser.parse_args()