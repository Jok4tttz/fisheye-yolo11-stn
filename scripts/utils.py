import os
import shutil
import math
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from ultralytics.utils import ops

def create_frames_dataframe(json_records):
    formatted_records = []

    for sequence_dict in json_records:
        sequence_name = sequence_dict["sequence_name"]
        sequence_path = sequence_dict["sequence_path"]
        ROIs_path = sequence_dict["ROIs_path"]
        ROI_path = None

        if ROIs_path is not None:
            ROI_path = f"{os.path.join(ROIs_path, sequence_name)}.png"

        for data in sequence_dict["records"]:
            frame_id = data["id"]
            file_name = data["file_name"]
            frame_source_path = os.path.join(sequence_path, file_name)
            frame_width = data["width"]
            frame_height = data["height"]
            
            formatted_record = {
                "frame_id" : frame_id,
                "sequence" : sequence_name,
                "frame_source_path" : frame_source_path,
                "ROI_path" : ROI_path,
                "frame_width" : frame_width,
                "frame_height" : frame_height
            }
            formatted_records.append(formatted_record)
    dataframe = pd.DataFrame.from_records(formatted_records)
    return dataframe
# -----------------------------------------------------------------------
def create_annotations_dataframe(json_records):
    formatted_records = []
    for sequence_dict in json_records:
        sequence_name = sequence_dict["sequence_name"]

        for data in sequence_dict["records"]:
            frame_id = data["image_id"]
            center_x = data["bbox"][0]
            center_y = data["bbox"][1]
            bb_width = data["bbox"][2]
            bb_height = data["bbox"][3]
            rotation = data["bbox"][4]

            formatted_record = {
                "frame_id" : frame_id,
                "sequence" : sequence_name,
                "center_x" : center_x,
                "center_y" : center_y,
                "bb_width" : bb_width,
                "bb_height" : bb_height,
                "rotation" : rotation
            }
            formatted_records.append(formatted_record)
    dataframe = pd.DataFrame.from_records(formatted_records)
    return dataframe
# -----------------------------------------------------------------------
def assign_target_path(data, dataset_root, dataset_dir, target_set):
    sequence = data["sequence"]
    id_number = data["frame_id"].split('_')[-1]
    file_name = f"{sequence}_{id_number}"
    target_base = os.path.join(target_set, file_name)
    dataset_base = os.path.join(dataset_dir, target_base)

    images_dir = os.path.join(dataset_root, "images")
    frame_target_path = os.path.join(images_dir, f"{dataset_base}.jpg")

    labels_dir = os.path.join(dataset_root, "labels")
    annotation_target_path = os.path.join(labels_dir, f"{dataset_base}.txt")

    columns_dict = {
        "frame_target_path" : frame_target_path,
        "annotation_target_path" : annotation_target_path
    }
    
    new_columns = pd.Series(columns_dict)
    return new_columns
# -----------------------------------------------------------------------
def clean_dataset_directories(dataset_root, dataset_dir):
    images_path = os.path.join(dataset_root, "images")
    labels_path = os.path.join(dataset_root, "labels")
    dir_list = [images_path, labels_path]

    
    for dir_path in dir_list:
        directory = os.path.basename(dir_path)
        dataset_path = os.path.join(dir_path, dataset_dir)

        if os.path.exists(dataset_path):
            print(f"Remove existing directory in {directory}: {dataset_path}")
            shutil.rmtree(dataset_path)
# -----------------------------------------------------------------------
def preprocess_ROI(ROI_path):
    ROI_image = Image.open(ROI_path).convert("RGBA")
    new_pixels = []
    
    for channels in ROI_image.getdata():
        if channels[0] >= 240 and channels[1] >= 240 and channels[2] >= 240:
            new_pixels.append((255, 255, 255, 0))
        else:
            new_pixels.append(channels)

    ROI_image.putdata(new_pixels)
    return ROI_image
# -----------------------------------------------------------------------
def write_image(frame_source_path, frame_target_path, ROI_path = None, position = (0, 0)):
    image = Image.open(frame_source_path).convert('RGBA')

    if ROI_path is not None:
        overlay = preprocess_ROI(ROI_path)
        image.paste(overlay, position, overlay)
    
    image = image.convert('RGB')
    image.save(frame_target_path, 'JPEG')
# -----------------------------------------------------------------------
def calculate_coordinates(center_x, center_y, box_width, box_height, angle):
    input_elements = torch.tensor([center_x, center_y, box_width, box_height, angle])
    output_elements = ops.xywhr2xyxyxyxy(input_elements).reshape(8)
    return output_elements.tolist()
# -----------------------------------------------------------------------
def reformat_annotation(elements, frame_width, frame_height, object_class = 0):
    base = [object_class]

    center_x, center_y = elements[:2]
    angle = elements[-1]

    if angle > 0 and angle < 90:
        width, height = elements[2:4]
    else:
        height, width = elements[2:4]
    
    angle = math.radians(angle)
    new_elements = calculate_coordinates(center_x, center_y, width, height, angle)
                    
    for index in range(len(new_elements)):
        if index % 2 == 0:
            new_elements[index] /= frame_width
        else:
            new_elements[index] /= frame_height
    
    new_annotation = ' '.join(map(str, base + new_elements))
    return new_annotation
# -----------------------------------------------------------------------
def write_files(data):
     # Frame Details
    frame_source_path = data["frame_source_path"]
    frame_target_path = data["frame_target_path"]
    frame_width = data["frame_width"]
    frame_height = data["frame_height"]
    ROI_path = data["ROI_path"]

    # Annotation Details
    bb_annotation = [data["center_x"], data["center_y"], data["bb_width"], data["bb_height"], data["rotation"]]
    annotation_target_path = data["annotation_target_path"]

    # Directory Check
    images_dir = os.path.dirname(frame_target_path)
    labels_dir = os.path.dirname(annotation_target_path)

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # File Writing
    if os.path.exists(frame_target_path) and os.path.exists(annotation_target_path):
        with open(annotation_target_path, 'a') as line:
            line.write(f"\n{reformat_annotation(bb_annotation, frame_width, frame_height)}")
    else:
        write_image(frame_source_path, frame_target_path, ROI_path)
        tqdm.write(f"{frame_target_path} was successfuly created!")
        
        with open(annotation_target_path, 'w') as line:     
            line.write(reformat_annotation(bb_annotation, frame_width, frame_height))
        tqdm.write(f"{annotation_target_path} was successfuly created!")
        tqdm.write('-' * 100)