#we will , see all the label files , then match them with the images , 
#in the next step , we will open the label file , look for the maximum class id , match it with the class mapping , 
#make the folder , and move the images in the folder 
import os
import shutil

dataset_path = r"C:\Users\csio\Desktop\Deep Learning\IB-Loss-main\VisDrone2019-DET-train" #main folder which has all the images and labels 
images_path = os.path.join(dataset_path, "images") # path of folder which has all the images 
labels_path = os.path.join(dataset_path, "labels") # path of labels which has all the labels 
output_path = os.path.join(dataset_path, "organized") # path of organized folder which has all the output files(files which were send from the images folder to the organized class )

class_mapping = { #maps each class id with the class name 
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


for labels_file in os.listdir(labels_path): #parse all the label files in this path , 
    if labels_file.endswith(".txt"):
        image_name = labels_file.replace(".txt", ".jpg") 
        labels_file_path = os.path.join(labels_path, labels_file) #the path of each label files 

        with open(labels_file_path, "r") as f: #open file and read it , read all the lines 
            lines = f.readlines()
            if not lines:
                continue  # skip if file is empty

            # Extract class IDs from all lines and find max
            class_ids = [int(line.strip().split()[0]) for line in lines if line.strip()] #a list of all the class id in the file 
            # we used int , because the 0th column in the label file is a string 
            max_class_id = max(class_ids) #maximum class id in the label file

        max_class_id_str = str(max_class_id)# as in the class mapping , the class id is string , 
        if max_class_id_str in class_mapping:
            max_class_name = class_mapping[max_class_id_str]
            class_folder = os.path.join(output_path, max_class_name)
           
            os.makedirs(class_folder, exist_ok=True)

            src = os.path.join(images_path, image_name)
            dst = os.path.join(class_folder, image_name)

            if os.path.exists(src):
                shutil.move(src, dst)
                

print(" Dataset successfully organized ")
