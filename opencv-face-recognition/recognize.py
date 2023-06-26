# python C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\recognize.py  --detector C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\face_detection_model --embedding-model C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\openface_nn4.small2.v1.t7  --recognizer  C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\output\recognizer.pickle  --le  C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\output\le.pickle --image C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\images\swx.jpg

import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
# 命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")#用来识别的图像
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")##OpenCV的深度学习人脸检测方法的路径
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")#从人脸ROI中提取的128-D特征向量输入到识别器
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")#识别器的路径
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")#标签编码器的路径，例如swx
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")#筛选弱脸检测的可选阈值
args = vars(ap.parse_args())

# 将三个模型从磁盘加载到内存中
#经过预先训练的Caffe DL模型，用于检测人脸在图像中的位置
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 预先训练的 Torch DL模型，用于计算128维面部嵌入
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# SVM人脸识别模型
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())
# 将图像加载到内存中并构建一个blob
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# 通过detector定位图像中的人脸
detector.setInput(imageBlob)
detections = detector.forward()

# 遍历
for i in range(0, detections.shape[2]):
	# 提取每个检测的confidence
	confidence = detections[0, 0, i, 2]

	# 将置信度与命令行args字典中包含的最小概率检测阈值进行比较，
	# 确保计算出的概率大于最小概率
	if confidence > args["confidence"]:
		#提取人脸ROI
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		#确保其空间维度足够大
		if fW < 20 or fH < 20:
			continue

		# 为面部ROI构建一个blob，然后传递这个blob
		# 通过人脸嵌入模型获得人脸的128维特征向量
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# 通过SVM模型，确定是谁
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# 采取最高概率指数，并查询编码器以找到名称，提取概率
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)