import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import train
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Image')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
parser.add_argument('--topk', type=int, help='Return top K predictions')
parser.add_argument('--labels', type=str, help='JSON file with label names')
parser.add_argument('--gpu', action='store_true', help='Use the GPU to calculate the predictions if available')

args, _ = parser.parse_known_args()

# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image) 
    #resize the images where the shortest side is 256 pixels (referred to https://gist.github.com/tomvon/ae288482869b495201a0 for guidance)
    side=256
    percent = (side/float(pil_image.size[0]))
    height = int((float(pil_image.size[1])*float(percent)))
    pil_image = pil_image.resize((side,height), Image.ANTIALIAS)
    #crop out the center 224x224 portion of the image (referred to https://stackoverflow.com/questions/39805750/pil-crop-image-give-incorrect-height-result for guidance) 
    width, height = pil_image.size 
    new=224
    left = (width - new)/2
    top = (height - new)/2
    right = (width + new)/2
    bottom = (height + new)/2
    pil_image=pil_image.crop((left, top, right, bottom))
    
    #convert color channels 
    img_loader = transforms.Compose([transforms.ToTensor()])
    pil_image = img_loader(pil_image).float()
    np_image = np.array(pil_image)    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image

def predict(image, checkpoint, topk=5, labels='', gpu=True): 
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if args.image:
        image = args.image     
        
    if args.checkpoint:
        checkpoint = args.checkpoint

    if args.topk:
        topk = args.topk
            
    if args.labels:
        labels = args.labels

    if args.gpu:
        gpu = args.gpu
        
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    image= process_image(image)
    image = Variable(torch.cuda.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)
    image.to(device)
    checkpoint_dict = torch.load(checkpoint)
    arch = checkpoint_dict['model_arch']
    output_size = len(checkpoint_dict['class_to_idx'])
    hidden_units = checkpoint_dict['hidden_units']
    model = pre_train_model(arch=arch, hidden_units=hidden_units,output_size=output_size)
    model.to(device)
    model.eval()
    x=model(image)
    result = x.topk(topk)
    if gpu and torch.cuda.is_available():
        probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
        classes = result[1].data.cpu().numpy()[0]
    else:       
        probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
        classes = result[1].data.numpy()[0]
    if labels:
        with open(labels, 'r') as f:
            cat_to_name = json.load(f)
        labels = list(cat_to_name.values())
        classes = [labels[x] for x in classes]
    model.train()
    #print out the top K classes along with associated probabilities
    if args.image:
        print('Top K classes and associated probabilities:', list(zip(classes, probs)))
    return probs,classes

if args.image and args.checkpoint:
    predict(args.image, args.checkpoint)
