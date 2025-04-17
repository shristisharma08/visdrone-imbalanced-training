import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from torchvision.datasets import ImageFolder  # loads images class-wise from each folder

class IMBALANCEVisDrone(ImageFolder):  # extends ImageFolder so it inherits everything including how images are loaded
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, transform=None):
        print("\n--- Initializing IMBALANCEVisDrone ---")
        super(IMBALANCEVisDrone, self).__init__(root, transform)
        print("Loaded dataset from:", root)
        print("Initial number of samples:", len(self.samples))
        print("Class names (alphabetical):", self.classes)
        print("Class-to-index mapping:", self.class_to_idx)
        
        np.random.seed(rand_number)
        print("Random seed set to:", rand_number)
        
        self.cls_num = len(self.classes)
        print("Number of classes:", self.cls_num)
        
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        print("Images to keep per class:", img_num_list)

        self.gen_imbalanced_data(img_num_list)
        print("Final number of samples after imbalance:", len(self.samples))
        print("Updated class distribution:", self.get_cls_num_list())
        print("--- Initialization Complete ---\n")

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        print("\n--- Calculating Image Numbers per Class ---")
        img_max = len(self.samples) / cls_num
        print("Max images per class before imbalance:", img_max)

        img_num_per_cls = []
        if imb_type == 'exp':
            print("Using exponential imbalance with factor:", imb_factor)
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
                print(f"Class {cls_idx} -> {int(num)} images")
        elif imb_type == 'step':
            print("Using step imbalance with factor:", imb_factor)
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
                print(f"Class {cls_idx} -> {int(img_max)} images")
            for cls_idx in range(cls_num // 2, cls_num):
                img_num_per_cls.append(int(img_max * imb_factor))
                print(f"Class {cls_idx} -> {int(img_max * imb_factor)} images")
        else:
            print("Using uniform distribution")
            img_num_per_cls.extend([int(img_max)] * cls_num)

        print("--- Finished Calculating Image Numbers ---\n")
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        print("\n--- Generating Imbalanced Dataset ---")
        targets_np = np.array(self.targets, dtype=np.int64)
        print("Original number of samples:", len(targets_np))

        unique, counts = np.unique(targets_np, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} samples")

        classes = np.unique(targets_np)
        print("Unique classes found:", classes)

        new_samples = []
        self.num_per_cls_dict = {}

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            print(f"\nClass {the_class}:")
            print(f"  Total found: {len(idx)} samples")
            print(f"  Will keep:  {the_img_num} samples")
            np.random.shuffle(idx)
            print(f"  Shuffled indices: {idx.tolist()[:10]}...")

            selected_idx = idx[:the_img_num]
            print(f"  Selected indices: {selected_idx.tolist()}")

            for i in selected_idx:
                new_samples.append(self.samples[i])

        self.samples = new_samples
        self.targets = [s[1] for s in new_samples]

        print("\nNew total number of samples:", len(self.samples))
        final_targets = np.array(self.targets)
        final_unique, final_counts = np.unique(final_targets, return_counts=True)
        for u, c in zip(final_unique, final_counts):
            print(f"  Class {u}: {c} samples")

        print("--- Finished Generating Imbalanced Dataset ---\n")

    def get_cls_num_list(self):
        return [self.num_per_cls_dict[i] for i in range(self.cls_num)]


if __name__ == '__main__':
    print("\n--- Main Block Execution ---")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    print("Transform defined.")

    trainset = IMBALANCEVisDrone(root="C:/Users/csio/Desktop/Deep Learning/IB-Loss-main/VisDrone2019-DET-train/organized", 
                                 imb_type='exp', 
                                 imb_factor=0.01, 
                                 rand_number=0, 
                                 transform=transform)
    print("Dataset created.")

    print("Accessing first image and label for debug:")
    trainloader = iter(trainset)
    data, label = next(trainloader)

    print("Image shape:", data.shape)
    print("Label index:", label)
    print("Label name:", trainset.classes[label])

    import pdb; pdb.set_trace()
