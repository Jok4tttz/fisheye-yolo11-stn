import warnings
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")

import os
import shutil
import torch
from ultralytics import YOLO

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if __name__ == '__main__':
    print(f"GPU Available\t: {torch.cuda.is_available()}")
    print(f"Device Count\t: {torch.cuda.device_count()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    model_path = os.path.normpath("models/raw/yolo11n-obb.yaml")
    data_path = os.path.normpath("datasets/targets/coco_2017.yaml")
    project = os.path.normpath("/mimlab/ja.archive01/FishEyeProject/runs/train/coco_pretrained")

    # model_path = os.path.normpath("models/trained/COCO/yolo11n_stn_100eph_001lr.pt")
    # data_path = os.path.normpath("datasets/targets/hbcd_train.yaml")
    # project = os.path.normpath("runs/train/hbcd_train/raw/")

    name = "yolo11n_test_100eph_001lr"
    run_path = os.path.join(project, name)
    

    if os.path.exists(run_path):
        print(f"Clearing existing directory: {run_path}")
        shutil.rmtree(run_path)

    model = YOLO(model_path)
    results = model.train(
        data = data_path,
        project = project,
        name = name,
        epochs = 100,
        classes = [0],
        batch = 256,
        device = 0,
        workers = 16,
        # cache = "ram",
        # lr0 = 0.1,
        val = False
    )

    # m = YOLO(model_path, task = "obb", verbose = True).model
    # m.info(verbose = True)