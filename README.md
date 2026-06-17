# Car-Accident-Detection-and-Prediction-using-CCTV
This project uses CCTV footage to spot and predict car accidents using CNN trained model and YOLO26x model. Its currently focused on two car collisions, so single car crashes and car crashes with other vehicles are not reliable.

# Features
Car accident detection using a custom-trained CNN model.

Vehicle detection and tracking using YOLO.

Collision analysis using IoU, proximity, and trajectory calculations.

Accident prediction based on vehicle movement and collision-course estimation.

Web-based user interface for video upload, processing, and result visualization.

Three operating modes:

Hybrid Mode, CNN Mode, and YOLO Mode (YOLO mode is for testing only)

Requirements:

Python
TensorFlow / Keras
OpenCV
Ultralytics YOLO
CUDA-enabled GPU (recommended)
Installation

# Running the Project

To run the project, run the app.py python file and then open browser to this local host site "http://127.0.0.1:5000/"

After that, just upload a video in the web UI, choose a processing mode you prefer, and press process button to start the car accident detection and prediction in the video.

You can use this link to find car accident videos you can test with:
https://drive.google.com/drive/folders/1IczMgji3xmNLO2QV7dNZwIm0pWgvSj9P?usp=sharing

------------------------------------------------------------------
# Important notes:

If yolo26x.pt is not installed, it will automatically be installed in models/weights folder directory when the app runs.

In case its not automatically installed, you need to download yolo26x.pt and put it in models/weights folder so the project works.

You can download it online from Ultralytics, or you can use this link and download yolo26x.pt from here: 
https://drive.google.com/drive/folders/1EIGN62y1V174vCL8i94QckqPL2bEje7R?usp=sharing

Yolo uses Cuda GPU, so make sure the right libraries are installed to support GPU. it can still run with CPU but it will be slow.

# Recommended Usage
Hybrid Mode is recommended because it combines CNN accident classification with YOLO vehicle collision analysis, reducing false positives.

# Project Scope and Limitations

This project was developed and tested primarily for detecting collisions between two vehicles.

Current limitations include:

Focuses mainly on two-car collision scenarios.
Single-vehicle crashes are not reliably detected.
Vehicle-pedestrian accidents are not supported.
Motorcycle and truck accidents are not specifically supported.
Best results are achieved using CCTV footage captured from elevated positions with clear visibility.
Performance may decrease under poor lighting conditions or unfavorable camera angles.
CPU execution is supported but significantly slower than GPU execution.

Imad AlDeen Mustapha


Master of Science in Computer Science

Lebanese International University (LIU)
