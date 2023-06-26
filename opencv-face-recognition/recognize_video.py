# python C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\recognize_video.py  --detector C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\face_detection_model  --embedding-model C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\openface_nn4.small2.v1.t7   --recognizer  C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\output\recognizer.pickle  --le  C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\output\le.pickle

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
# 命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")#OpenCV的深度学习人脸检测方法
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")#OpenCV的深度学习人脸嵌入模型
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")#SVM训练识别人脸的模型路径
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")#标签编码器路径
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")#过滤弱检测的最小概率
args = vars(ap.parse_args())

# 从磁盘加载序列化人脸检测器
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 从磁盘加载序列化的人脸嵌入模型
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# 与标签编码器一起加载实际的人脸识别模型
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# 初始化视频流，然后让摄像机传感器预热
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 启动FPS吞吐量估计器
fps = FPS().start()

# 从视频文件流循环帧
while True:
	# 从线程视频流中获取帧
	frame = vs.read()

	# 调整帧的大小，使其宽度为600像素（同时
	# 保持长宽比），然后获取图像尺寸
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# 从图像构造一个blob
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# 基于OpenCV深度学习的人脸检测在输入图像中的定位
	detector.setInput(imageBlob)
	detections = detector.forward()

	# 遍历检测
	for i in range(0, detections.shape[2]):
		# 提取与预测相关的置信度（即概率）
		confidence = detections[0, 0, i, 2]

		# 滤除弱检测
		if confidence > args["confidence"]:
			# 计算面边界框的（x，y）-坐标
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# 提取面部ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# 确保工作面宽度和高度足够大
			if fW < 20 or fH < 20:
				continue

			# 为ROI构造一个blob，
			# 将blob通过我们的人脸嵌入模型得到人脸的128-d特征向量
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# 进行分类以识别人脸
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# 绘制面的边界框以及相关的概率
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# 更新FPS计数器
	fps.update()

	# 显示输出帧
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 如果按了“q”键，就脱离循环
	if key == ord("q"):
		break

# 停止计时器并显示FPS信息
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# 清理
cv2.destroyAllWindows()
vs.stop()