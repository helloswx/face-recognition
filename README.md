# OpenCV based face recognition project.
**Step 1:** Detect faces.

**Step 2:** Compute 128-d face embeddings to quantify a face.

**Step 3:** Train a Support Vector Machine (SVM) on top of the embeddings.

**Step 4:** Recognize faces in images and video streams.

## Terminal execution commands

Generate image training set:
```
python ...\face_recognition\build-face-dataset\build_face_dataset.py --cascade ...\face_recognition\build-face-dataset\haarcascade_frontalface_default.xml --output ...\face_recognition\opencv-face-recognition\dataset\...
```
Compute 128-d face embeddings to quantify a face：
```
python ...\face_recognition\opencv-face-recognition\extract_embeddings.py --dataset ...\face_recognition\opencv-face-recognition\dataset --embeddings ...\face_recognition\opencv-face-recognition\output\embeddings.pickle --detector ...\face_recognition\opencv-face-recognition\face_detection_model  --embedding-model ...\face_recognition\opencv-face-recognition\openface_nn4.small2.v1.t7
```

Train a Support Vector Machine (SVM) on top of the embeddings：
```
python ...\face_recognition\opencv-face-recognition\train_model.py --embeddings ...\face_recognition\opencv-face-recognition\output\embeddings.pickle --recognizer ...\face_recognition\opencv-face-recognition\output\recognizer.pickle --le ...\face_recognition\opencv-face-recognition\output\le.pickle
```

Recognize faces in images:
```
python ...\face_recognition\opencv-face-recognition\recognize.py  --detector ...\face_recognition\opencv-face-recognition\face_detection_model --embedding-model ...\face_recognition\opencv-face-recognition\openface_nn4.small2.v1.t7 --recognizer ...\face_recognition\opencv-face-recognition\output\recognizer.pickle  --le ...\face_recognition\opencv-face-recognition\output\le.pickle --image ...\face_recognition\opencv-face-recognition\images\...
```

Recognize faces in video streams：
```
python ...\face_recognition\opencv-face-recognition\recognize_video.py  --detector ...\face_recognition\opencv-face-recognition\face_detection_model --embedding-model ...\face_recognition\opencv-face-recognition\openface_nn4.small2.v1.t7 --recognizer ...\face_recognition\opencv-face-recognition\output\recognizer.pickle  --le ...\face_recognition\opencv-face-recognition\output\le.pickle
```
