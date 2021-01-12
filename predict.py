import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
from os.path import isdir
from os import listdir
from collections import OrderedDict
import json

def get_input_args():
    parser = argparse.ArgumentParser(description='prediction') 
    parser.add_argument('--image', type=str, default='flowers/test/1/image_06743.jpg', help='Path of image')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Loading checkpoint')
    parser.add_argument('--gpu', type=bool, default=False, help='GPU on or off')
    parser.add_argument('--topK', type=int, default=5, help='Get top K largest values')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='JSON File')
    
    return parser.parse_args()


def loading_checkpoint(image_path):
    checkpoint = torch.load(image_path)
    model = models.vgg19(pretrained=True)
    model.to('cuda')
    
    architecture = checkpoint['architecture']
    hidden_layers = checkpoint['hidden_layers']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])  
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    image_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])
                                         ])
    
    processed_Img = image_transforms(Image.open(image)).float()
    
    return processed_Img  


def predict(image_path, model):

    img = process_image(image_path).unsqueeze_(0).float()
    
    model = loading_checkpoint(model)
    with torch.no_grad():
        output = model.forward(img.cuda())
    
    probs, classes = torch.exp(output).topk(5)
    
    probs = (np.array(probs))[0] 
    
    to_class = []
    for key, value in model.class_to_idx.items():
        to_class.append(key)
    
    classes2 = []
    for C in (np.array(classes))[0]:
        classes2.append(to_class[C])
        
    return probs, classes2


args = get_input_args()

with open(args.cat_to_name, 'r') as f:
    cat_to_name = json.load(f)

probs, classes = predict(args.image, args.checkpoint)
for probs, classes in zip(probs, classes):
    print("Class_____%2s   Probality Percentage_____%f" % (classes, probs*100))
  
