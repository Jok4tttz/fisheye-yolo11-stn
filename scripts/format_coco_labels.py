import os
import shutil
import json
import math
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics.utils import ops
from utils import reformat_annotation

def write_labels(data):
    label_target_path = data["label_target_path"]

    object_class = data["object_class"]
    bb_annotation = [data["center_x"], data["center_y"], data["bb_width"], data["bb_height"], data["rotation"]]

    image_width = data["image_width"]
    image_height = data["image_height"]

    labels_dir = os.path.dirname(label_target_path)

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    if os.path.exists(label_target_path):
        with open(label_target_path, 'a') as line:
            line.write(f"\n{reformat_annotation(bb_annotation, image_width, image_height, object_class)}")
    else:
        with open(label_target_path, 'w') as line:     
            line.write(reformat_annotation(bb_annotation, image_width, image_height, object_class))
        tqdm.write(f"{label_target_path} was successfuly created!")
        tqdm.write('-' * 100)

if __name__ == '__main__':
    source_path = os.path.normpath("datasources/COCO2017/annotations")
    dataset_labels_source = os.path.normpath("datasets/labels/coco_2017")
    dataset_images_source = os.path.normpath("datasets/images/coco_2017")
    set_list = sorted(os.listdir(source_path))

    labels_records = []
    print("Loading labels data...")
    for set in set_list:
        set_path = os.path.join(source_path, set)
        dataset_labels_set = os.path.join(dataset_labels_source, set)
        dataset_images_set = os.path.join(dataset_images_source, set)
        label_list = sorted(os.listdir(set_path))

        tqdm.write(f"\nCreating '{set}' label records...")
        tqdm.write("-" * 100)
        for label in tqdm(label_list):
            label_source_path = os.path.join(set_path, label)
            label_target_path = os.path.join(dataset_labels_set, label)

            image_id = os.path.splitext(label)[0]
            image_filename = f"{image_id}.jpg"
            related_image_path = os.path.join(dataset_images_set, image_filename)
            related_image = Image.open(related_image_path)
            image_width = related_image.width
            image_height = related_image.height

            with open(label_source_path, 'r') as label_file:
                for line in label_file:
                    annotations = line.split()
                    
                    # tqdm.write(f"{annotations}")
                    format_dict = {
                        "label_target_path" : label_target_path,
                        "object_class" : annotations[0],
                        "center_x" : float(annotations[1]),
                        "center_y" : float(annotations[2]),
                        "bb_width" : float(annotations[3]),
                        "bb_height" : float(annotations[4]),
                        "rotation" : 0,
                        "image_width" : image_width,
                        "image_height" : image_height
                    }
                    labels_records.append(format_dict)
    
    labels_data = pd.DataFrame(labels_records)

    tqdm.pandas()
    print("\nWriting Files...")
    print('-' * 100)
    labels_data.progress_apply(write_labels, axis = 1)
    print("All files has been succesfully written!")