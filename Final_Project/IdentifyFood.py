import pandas as pd
import random
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def identify_food_name(image_path):
    device = 'cpu'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),       
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data= pd.read_csv(r"/home/user/Food_Classification/img/train_img.csv")
    classes=data['ClassName'].unique()
    num_classes = data['ClassName'].nunique()
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)

    
    # Load the model
    model.load_state_dict(torch.load(r'/home/user/Food_Classification/Model.pth',map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transformations
    input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict the class
    with torch.no_grad():
        output = model(input_image)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = classes[predicted_idx.item()]
        
    # Display the image with the predicted class
    plt.imshow(image)
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    plt.show()
    plt.show(block=True)  # 手動關閉視窗後執行下一步
    
    return predicted_class
    
def identify_food_intheDirAllImage():
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)

    unlabeled_dir = r'C:\Users\HP\.cache\kagglehub\datasets\bjoernjostein\food-classification\versions\2\test_images\test_images'
    unlabeled_dir = r'/home/user/Food_Classification/img'

    random_images = random.sample(os.listdir(unlabeled_dir), 10)

    # Load the model
    model.load_state_dict(torch.load(r'/home/user/Food_Classification/Model.pth',map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    # Loop through the random images
    for image_name in random_images:
        # Load the image
        image_path = os.path.join(unlabeled_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        
        # Predict the class
        with torch.no_grad():
            output = model(input_image)
            _, predicted_idx = torch.max(output, 1)
            predicted_class = classes[predicted_idx.item()]
        
        # Display the image with the predicted class
        plt.imshow(image)
        plt.title(f'Predicted Class: {predicted_class}')
        plt.axis('off')
        plt.show()
        plt.show(block=True)  # 手動關閉視窗後執行下一步
