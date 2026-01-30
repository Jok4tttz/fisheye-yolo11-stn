import warnings
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")

import os
import shutil
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    print(f"GPU Available\t: {torch.cuda.is_available()}")
    print(f"Device Count\t: {torch.cuda.device_count()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    torch.use_deterministic_algorithms(False)
    model_path = os.path.normpath("models/trained/HBCD/yolo11n_stn_100eph.pt")
    data_path = os.path.normpath("datasets/targets/wdf_cv2f/fold_2.yaml")
    project = os.path.normpath("runs/test/wdf_cv2f/fold_2")
    name = "yolo11n_stn_100eph"
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
        batch = 8,
        workers = 16,
        val = True
    )