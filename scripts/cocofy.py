import os
import json
from PIL import Image
from tqdm import tqdm

def reformat_HABBOF(source_path):
    frames_path = os.path.join(source_path, "frames")
    annotations_path = os.path.join(source_path, "annotations")
    sequence_list = sorted(os.listdir(frames_path))

    for sequence in sequence_list:
        sequence_path = os.path.join(frames_path, sequence)
        json_path = os.path.join(annotations_path, f"{sequence}.json")

        frame_list = sorted(os.listdir(sequence_path))
        annotations_list = []
        images_list = []

        print(f"Creating {sequence}.json...")
        for file_name in tqdm(frame_list):
            file_path = os.path.join(sequence_path, file_name)

            if ".jpg" in file_name:
                image = Image.open(file_path)
                image_dict = {
                    "file_name" : file_name,
                    "id" : os.path.splitext(file_name)[0],
                    "width" : image.width,
                    "height" : image.height
                }
                images_list.append(image_dict)

            if ".txt" in file_name:
                with open(file_path, 'r') as bb_list:
                    person_id = 1
                    for line in bb_list:
                        bb = [float(value) for value in line.split()[1:]]
                        annotation_dict = {
                            "area" : bb[2] * bb[3],
                            "bbox" : bb,
                            "category_id" : 1,
                            "image_id" : os.path.splitext(file_name)[0],
                            "is_crowd" : 0,
                            "segmentation" : [],
                            "person_id" : person_id
                        }
                        annotations_list.append(annotation_dict)
                        person_id += 1
        
        data_dump = {
            "annotations" : annotations_list,
            "images" : images_list,
            "category" : [
                {
                    "id" : 1,
                    "name" : "person",
                    "supercategory" : "person"
                }
            ]
        }
        
        with open(json_path, 'w') as json_file:
            json.dump(data_dump, json_file, indent = 4)
        print("---------------------------------------------")
    
    print("All JSON files were succesfully created!")

if __name__ == "__main__":
    source_dir = "HABBOF"
    source_path = os.path.join("./datasources/", source_dir)

    if source_dir == "HABBOF":
        print("Reformating HABBOF dataset:")
        reformat_HABBOF(source_path)