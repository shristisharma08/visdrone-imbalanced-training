#we will , see all the label files , then match them with the images , 
#in the next step , we will open the label file , look for the first class id , match it with the class mapping , 
#make the folder , and move the images in the folder 
import os
import shutil

dataset_path = r"C:\Users\csio\Desktop\Deep Learning\IB-Loss-main\VisDrone2019-DET-train"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")
output_path = os.path.join(dataset_path, "organized")

class_mapping = {
    "0": "pedestrian",
    "1": "people",
    "2": "bicycle",
    "3": "car",
    "4": "van",
    "5": "truck",
    "6": "tricycle",
    "7": "awning-tricycle",
    "8": "bus",
    "9": "motor",
}

#os.listdir returns the folder names in alphabatical order 
for labels_file in os.listdir(labels_path):
    if labels_file.endswith(".txt"):
        image_name = labels_file.replace(".txt", ".jpg")
        labels_file_path = os.path.join(labels_path, labels_file)

        with open(labels_file_path, "r") as f:
            first_line = f.readline().strip()
            if not first_line:
           
                continue  
                    # : it is taking the first class id and then moving the image to the designated folder if it is present or else making a folder 
            class_id = first_line.split(" ")[0]

        if class_id in class_mapping:
            class_name = class_mapping[class_id]
            class_folder = os.path.join(output_path, class_name)
            

           
            os.makedirs(class_folder, exist_ok=True)

            src = os.path.join(images_path, image_name)
            dst = os.path.join(class_folder, image_name)

            if os.path.exists(src):
                shutil.move(src, dst)
                

print(" Dataset successfully organized by first object class ID!")
