import os
import shutil
import json
import math
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics.utils import ops
from sklearn.model_selection import train_test_split
from utils import create_annotations_dataframe
from utils import create_frames_dataframe
from utils import assign_target_path
from utils import clean_dataset_directories
from utils import write_files

if __name__ == '__main__':
    # Parameters Initialization
    source_dir = "WEPDTOF"
    dataset_root = os.path.normpath("datasets")
    # dataset_dir = os.path.normpath("hbcd_train")
    dataset_dir = os.path.join("wdf_test", "fold_2")
    # target_set = "train"
    target_set = "train"
    # ROI_dir = None
    ROI_dir = "ROI_maps"
    # cross_set = None
    # cross_set = [
    #     "empty_store",
    #     "exhibition_setup",
    #     "convenience_store",
    #     "large_office",1
    #     "large_office_2",
    #     "warehouse",
    #     "exhibition",
    #     "call_center"
    # ]
    cross_set = [
        "tech_store",
        "jewelry_store",
        "jewelry_store_2",
        "street_grocery",
        "printing_store",
        "repair_store",
        "it_office",
        "kindergarten"
    ]
    split = 0
    clean = False
    # clean = True


    # Convert JSON Annotations to Pandas DataFrame
    source_path = os.path.join("datasources", source_dir)
    annotations_path = os.path.join(source_path, "annotations")
    frames_path = os.path.join(source_path, "frames")
    ROIs_path = None

    if ROI_dir is not None:
        ROIs_path = os.path.join(source_path, ROI_dir)

    annotations_records = []
    frames_records = []
    json_list = sorted(os.listdir(annotations_path))

    if cross_set is not None:
        json_list = [f"{sequence}.json" for sequence in cross_set]

    print(f"Retrieving data from : {source_path}...")
    print('-' * 100)
    for json_file in tqdm(json_list):
        json_path = os.path.join(annotations_path, json_file)

        sequence_name = os.path.splitext(json_file)[0]
        sequence_path = os.path.join(frames_path, sequence_name)

        with open(json_path, 'r') as read_json:
            json_data = json.load(read_json)

        records = json_data["images"]
        frames_records.append(
            {
                "records" : records,
                "sequence_name" : sequence_name,
                "sequence_path" : sequence_path,
                "ROIs_path" : ROIs_path
            }
        )

        records = json_data["annotations"]
        annotations_records.append(
            {
                "records" : records,
                "sequence_name" : sequence_name
            }
        )


    frames_data = create_frames_dataframe(frames_records)
    annotations_data = create_annotations_dataframe(annotations_records)

    # Frames Path Assignment
    frames_list = [frames_data]

    if split:
        split_set = "val"
        train_data, val_data = train_test_split(frames_list[0], test_size = split)
        val_data[["frame_target_path", "annotation_target_path"]] = val_data.apply(
            assign_target_path,
            dataset_root = dataset_root,
            dataset_dir = dataset_dir,
            target_set = split_set,
            axis = 1
            )
        frames_list = [train_data, val_data]

    main_data = frames_list[0]
    main_data[["frame_target_path", "annotation_target_path"]] = main_data.apply(
            assign_target_path,
            dataset_root = dataset_root,
            dataset_dir = dataset_dir,
            target_set = target_set,
            axis = 1
    )

    frames_data = pd.concat(frames_list)

    # Write Data to Target Paths
    complete_data = pd.merge(annotations_data, frames_data, how = "inner", on = ["frame_id", "sequence"]).sort_index()

    if clean:
        print("CLEAN MODE: starting from zero...")
        clean_dataset_directories(dataset_root, dataset_dir)

    tqdm.pandas()
    print("\nWriting Files...")
    print('-' * 100)
    complete_data.progress_apply(write_files, axis = 1)
    print("All files has been succesfully written!")