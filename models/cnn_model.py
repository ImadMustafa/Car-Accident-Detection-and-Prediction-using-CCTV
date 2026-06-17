import tensorflow as tf
from pathlib import Path

cnn_model = None

def load_cnnmodel():
    global cnn_model
    if cnn_model is None:
        script_dir = Path(__file__).parent
        weights_path = script_dir / "weights" / "car_accident_model.keras"
        cnn_model = tf.keras.models.load_model(weights_path)
    return cnn_model