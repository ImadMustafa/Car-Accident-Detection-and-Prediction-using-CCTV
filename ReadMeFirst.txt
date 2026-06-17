This is the CCTV Car Accident Detection and Prediction Project.

To run the project, run the app.py python file and then open browser to this local host site "http://127.0.0.1:5000/"

After that, just upload a video in the web UI, choose a processing mode you prefer, and press process button to start the car accident detection and prediction in the video.

You can use this link to find car accident videos you can test with:
https://drive.google.com/drive/folders/1IczMgji3xmNLO2QV7dNZwIm0pWgvSj9P?usp=sharing

------------------------------------------------------------------
Important notes:

If yolo26x.pt is not installed, it will automatically be installed in models/weights folder directory when the app runs.

In case its not automatically installed, you need to download yolo26x.pt and put it in models/weights folder so the project works.

You can download it online from Ultralytics, or you can use this link and download yolo26x.pt from here: 
https://drive.google.com/drive/folders/1EIGN62y1V174vCL8i94QckqPL2bEje7R?usp=sharing

Yolo uses Cuda GPU, so make sure the right libraries are installed to support GPU. it can still run with CPU but it will be slow.