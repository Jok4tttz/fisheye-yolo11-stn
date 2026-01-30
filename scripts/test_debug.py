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

    model_path = os.path.normpath("models/yolo11n-obb-st.yaml")
    data_path = os.path.normpath("datasets/targets/wdf_cv2f/fold_1.yaml")
    project = os.path.normpath("runs/test")
    name = "just_for_test"
    run_path = os.path.join(project, name)
    
    if os.path.exists(run_path):
        print(f"Clearing existing directory: {run_path}")
        shutil.rmtree(run_path)

    # m = YOLO(model_path, task="obb", verbose=True).model  # prints build table
    # net = m.model  # Sequential
    # for i, layer in enumerate(net): # Removed 'model' since it's sequential
    #     if layer.__class__.__name__ == "STN":
    #         layer.register_forward_hook(lambda mod, inp, out: print(f"STN out: {tuple(out.shape)}"))
    #         net[i+1].register_forward_hook(lambda mod, inp, out: print(f"next({mod.__class__.__name__}) out: {tuple(out.shape)}")) # Removed 'model' since it's sequential
    # with torch.no_grad():
    #     m(torch.zeros(1, 3, 640, 640))

    model = YOLO(model_path, task = "obb")
    
    results = model.train(
        data = data_path,
        project = project,
        name = name,
        epochs = 3,
        batch = 8,
        workers = 16,
        val = True
    )