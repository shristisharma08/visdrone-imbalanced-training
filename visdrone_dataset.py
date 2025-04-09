#in this file , firstly , we are looping through the organized folder, then receiving a image , parrallely an label , then we are transforming the image and label , then returning it , along with the index 
#whenever an index is called , we are returning the image and the label 

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class VisDroneDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        # DOUBT : why are the list created 
        self.images = []
        self.labels = []

        dataset_name = "VisDrone2019-DET-train" if train else "VisDrone2019-DET-val"
        dataset_folder = os.path.join(self.root, dataset_name, "organized")
        labels_folder = os.path.join(self.root, dataset_name, "labels")

        print(f"🔍 Checking dataset folder: {dataset_folder}")

        # : if the folders doesnt exist , it is for debugging 
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f" Dataset folder not found: {dataset_folder}")
        if not os.path.exists(labels_folder):
            raise FileNotFoundError(f" Labels folder not found: {labels_folder}")


            # : category means , classes folder that is made , 
        for category in os.listdir(dataset_folder):
            category_path = os.path.join(dataset_folder, category) #C:\Users\csio\Desktop\Deep Learning\IB-Loss-main\VisDrone2019-DET-train\organized\bicycle
            # : it is to check if category is a folder of images 
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    if img_name.endswith(".jpg"):
                        img_path = os.path.join(category_path, img_name) #C:\Users\csio\Desktop\Deep Learning\IB-Loss-main\VisDrone2019-DET-train\organized\bicycle\0000071_07349_d_0000012.jpg
                        label_path = os.path.join(labels_folder, img_name.replace(".jpg", ".txt"))#C:\Users\csio\Desktop\Deep Learning\IB-Loss-main\VisDrone2019-DET-train\labels\0000071_07349_d_0000012.txt
                        
                        self.images.append(img_path)
                        # DOUBT :: but we have organised the images based on the labels only , so how can images exist while labels cant??
                        self.labels.append(label_path if os.path.exists(label_path) else None)
    #length of image 
    def __len__(self):
        return len(self.images)

    #get item 
    # DOUBT : how we gave index , ANSWER : in the "organized" file 
    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.labels[index]

        print(f"\n📦 Fetching item at index: {index}")
        print(f"   ├─ Image path: {img_path}")
        print(f"   └─ Label path: {label_path if label_path else ' No label path'}")

    # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        print(f"    Image loaded and transformed — Shape: {image.shape} | Type: {type(image)}")

    # Default class_id = 0 (if label not found)
        class_id = 0
        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                label_data = f.readlines()
            if label_data:
                class_id = int(label_data[0].split(' ')[0])
                print(f"     Class ID found in label: {class_id}")
            else:
                print("     Label file is empty.")
        else:
            print("     Label file not found. Using default class ID = 0")

        label = torch.tensor(class_id, dtype=torch.long)
        print(f"    Label tensor created: {label} | Shape: {label.shape} | Type: {type(label)}")

        return image, label
        


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = VisDroneDataset(root="C:/Users/csio/Desktop/Deep Learning/IB-Loss-main", train=True, transform=transform)
    print(f" Loaded {len(dataset)} training images")

     # 👇 Triggering __getitem__() with index 0 (change index if needed)
    image, label = dataset[0]