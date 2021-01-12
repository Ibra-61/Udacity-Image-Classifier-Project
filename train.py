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

#Input Arguments
parser = argparse.ArgumentParser(description='Training') 
parser.add_argument('--data', type=str, default='flowers', help='Data containing train and test images')
parser.add_argument('--architecture', type = str, default = 'vgg19', help = 'Model Architecture')
parser.add_argument('--epochs', type = int, default = 10 ,help = 'Number of training epochs for model') 
parser.add_argument('--hidden_layers', type = int, default = [128, 64], help = 'Number of nodes in each hidden layer')   
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learningrate of model') 
parser.add_argument('--dropout', type = float, default = 0.5, help = 'Dropout of model') 
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Directory to save checkpoints')
parser.add_argument('--gpu', type = bool, default = False ,help = 'GPU on or off')

args = parser.parse_args()

data_dir = args.data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomSizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

#Load data
train_data = datasets.ImageFolder(data_dir + '/train', transform = train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform = test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform = valid_transforms)

#Define dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)

#Building model
model = models.vgg19(pretrained=True)

#Freezing parameters
for param in model.parameters():
    param.requires_grad = False
    
    
classifier = nn.Sequential(OrderedDict([
                                        ('dropout', nn.Dropout(0.5)),
                                        ('fc1', nn.Linear(25088,128)),
                                        ('relu1', nn.ReLU()),
                                        ('fc2', nn.Linear(128,64)),
                                        ('relu2', nn.ReLU()),
                                        ('fc3', nn.Linear(64,102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))

model.classifier = classifier

#Optimize
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

#Train Network

epochs = args.epochs  
steps = 0
running_loss = 0
print_every = 20
model.to('cuda')

for epoch in range(epochs):
    print("Epoch: {}/{}".format(epoch+1,epochs))
    
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    model.to('cuda:0')
                    outputs = model.forward(inputs)
                    batch_loss = criterion(outputs, labels)
                
                    valid_loss += batch_loss.item()
                
                    #Accuracy
                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
            
#Test
total = 0 
correct = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model.forward(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f'Accuracy: {correct/total*100}%')

#Save
model.class_to_idx = train_data.class_to_idx
checkpoint = {'architecture':args.architecture,
              'inputs':25088,
              'outputs':102,
              'hidden_layers': args.hidden_layers,
              'epochs':args.epochs,
              'learning_rate':args.learning_rate,
              'dropout':args.dropout,
              'classifier':classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'optimizer': optimizer.state_dict
             } 

torch.save(checkpoint, args.save_dir)

