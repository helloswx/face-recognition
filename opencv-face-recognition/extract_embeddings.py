# python C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\extract_embeddings.py --dataset C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\dataset --embeddings C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\output\embeddings.pickle --detector C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\face_detection_model  --embedding-model C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\openface_nn4.small2.v1.t7
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# 构造参数解析器和解析的参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")#dataset的路径
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")#嵌入文件的路径
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")#caffe深度学习人脸检测器
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")#提取128-D面部嵌入向量
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")#可选阈值过滤
args = vars(ap.parse_args())
# 从磁盘装载序列化的人脸检测器
print("[INFO] loading face detector...")
#使用基于Caffe的DL人脸检测器来定位图像中的人脸。
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
#通过深度学习特征提取提取面部嵌入
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# 从磁盘装载序列化的脸嵌入模型
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
# 获取路径输入dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
# 初始化
knownEmbeddings = []
knownNames = []
# 初始化总数
total = 0
# 遍历图像路径
for (i, imagePath) in enumerate(imagePaths):
	# 从图像路径中提取人的名字
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# 加载图片,调整它的宽度为600像素,获取图像尺寸
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# 构造blob
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# 基于OpenCV的深度学习人脸检测器来识别图像中的人脸
	detector.setInput(imageBlob)
	detections = detector.forward()

	# 至少有一个检测
	if len(detections) > 0:
		# 假设图像中只有一张脸，因此我们以最高的置信度提取检测结果，
		# 并检查以确保置信度符合用于滤除弱检测的最小概率阈值
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
			# 计算(x, y)坐标的边界框的脸
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# 提取面部ROI并抓取/检查尺寸
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# 确保面部ROI足够大
			if fW < 20 or fH < 20:
				continue

			# 从面部ROI构造blob
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# 生成描述面部的128-D向量
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# 将数据转储到磁盘
#将数据序列化到 pickle文件中。
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()