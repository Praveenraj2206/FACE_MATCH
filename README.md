Face Match Implementation (OpenCV + DeepFace)
This project performs real-time face verification using OpenCV and DeepFace. The script loads multiple reference images, 
computes an averaged embedding, and compares webcam frames to determine if a match is found.

FEATURES
• Real-time webcam face verification
• Uses DeepFace with VGG-Face model
• RetinaFace backend for accurate detection
• Embedding smoothing using sliding window
• Debug distance logs
• Supports multiple reference images

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

PROJECT SETUP


1) Clone the GitHub Repository
https://github.com/Praveenraj2206/FACE_MATCH_IMPLEMENTATION.git


2) Open the folder in VS Code.

#CREATE & ACTIVATE VIRTUAL ENVIRONMENT
python -m venv venv
.\venv\Scripts\Activate.ps1

#MOVE TO PROJECT FOLDER
cd OPENCVPROJECT

#OPTIONAL → Upgrade pip
python -m pip install --upgrade pip

#INSTALL REQUIRED LIBRARIES
pip install opencv-python
pip install numpy
pip install deepface
pip install tensorflow==2.20.0
pip install tf-keras==2.20.0
pip install retina-face
pip install pandas
pip install matplotlib
pip install tqdm

#ADD YOUR REFERENCE IMAGES
Add your face images (me1.jpg, me2.jpg, me3.jpg) into the project folder.
You can add more images if needed.

3) RUN THE PROJECT
python FaceMatch.py

4) OUTPUT
[INFO] Loading reference images...
[INFO] Reference embedding computed.
[INFO] Starting camera... Press 'q' to quit.
[DEBUG] Distance: 0.3342  → MATCH FOUND
[DEBUG] Distance: 1.2438  → NO MATCH

5) STOP THE PROGRAM
Press 'q' in the webcam window
OR
Press Ctrl + C in the terminal
