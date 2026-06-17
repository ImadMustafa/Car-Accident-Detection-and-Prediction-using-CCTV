from ultralytics import YOLO
from pathlib import Path

yolo_model = None

def load_yolomodel():
    global yolo_model
    if yolo_model is None:
        script_dir = Path(__file__).parent
        weights_path = script_dir / "weights" / "yolo26x.pt"
        yolo_model = YOLO(weights_path)
    return yolo_model