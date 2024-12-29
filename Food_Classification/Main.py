# 正確率很差
# https://www.kaggle.com/code/oussamab25/food-class

# -------------------------------------------------------------------------
# Download latest version
# import kagglehub
# path = kagglehub.dataset_download("bjoernjostein/food-classification")
# print("Path to dataset files:", path)
# -------------------------------------------------------------------------


# import pandas as pd
# import torch
# import os
# from PIL import Image
# import torchvision.models as models
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import random
# import matplotlib.pyplot as plt

# data= pd.read_csv(r"C:\Users\HP\.cache\kagglehub\datasets\bjoernjostein\food-classification\versions\2\train_img.csv")
# classes=data['ClassName'].unique()
# num_classes = data['ClassName'].nunique()

# print(classes)
# print(num_classes)
# print(data.shape)
# print(data.head())

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),       
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, df, root_dir, transform=None, train=True):
#         self.df = df
#         self.root_dir = root_dir
#         self.transform = transform
#         self.train = train

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         img_name = self.df.iloc[idx, 0]
#         img_path = os.path.join(self.root_dir, img_name)
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         if self.train:
#             label = self.df.iloc[idx, 1]
#             return image, label
#         else:
#             return image
        
# train_dataset = CustomDataset(train_df, r'C:\Users\HP\.cache\kagglehub\datasets\bjoernjostein\food-classification\versions\2\train_images\train_images', transform=transform)
# val_dataset = CustomDataset(val_df, r'C:\Users\HP\.cache\kagglehub\datasets\bjoernjostein\food-classification\versions\2\train_images\train_images', transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
# model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model.to(device)

# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train() 
#     running_loss = 0.0 
#     with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
#         for batch_idx, (images, labels) in enumerate(pbar, 1):
#             label_to_idx = {label: idx for idx, label in enumerate(classes)}
#             numeric_labels = [label_to_idx[label] for label in labels]
#             labels_tensor = torch.tensor(numeric_labels, dtype=torch.long).to(device)
#             images = images.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels_tensor)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * images.size(0)
#             if batch_idx % 100 == 0:
#                 pbar.set_postfix({'Loss': loss.item()})
#     epoch_loss = running_loss / len(train_loader.dataset) 
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")





# model.eval() 
# total_correct = 0
# total_samples = 0
# running_loss = 0.0


# predicted_labels = []
# true_labels = []

# with torch.no_grad():
#     for images, labels in val_loader:
#         images = images.to(device)
#         label_to_idx = {label: idx for idx, label in enumerate(classes)}
#         numeric_labels = [label_to_idx[label] for label in labels]
#         labels_tensor = torch.tensor(numeric_labels, dtype=torch.long).to(device)
        
#         # Forward pass
#         outputs = model(images)
        
#         # Compute loss
#         loss = criterion(outputs, labels_tensor)
#         running_loss += loss.item() * images.size(0)
        
#         ## Get predicted labels
#         _, predicted = torch.max(outputs, 1)

#         # Update total correct and total samples
#         total_correct += (predicted == labels_tensor).sum().item()
#         total_samples += labels_tensor.size(0)
        
#         # Append predicted labels and true labels for further analysis
#         predicted_labels.extend(predicted.cpu().numpy())
#         true_labels.extend(labels_tensor.cpu().numpy())

# # Calculate average loss and accuracy
# average_loss = running_loss / len(val_dataset)
# accuracy = total_correct / total_samples

# print(f"Validation Loss: {average_loss:.4f}")
# print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# torch.save(model.state_dict(), 'Model.pth')













import pandas as pd
import random
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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