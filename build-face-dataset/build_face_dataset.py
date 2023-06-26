
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
# 命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,help = "path to where the face cascade resides")#级联
ap.add_argument("-o", "--output", required=True,help="path to output directory")#输出
args = vars(ap.parse_args())

# 从磁盘加载OpenCV的Haar级联进行人脸检测
detector = cv2.CascadeClassifier(args["cascade"])

# 初始化视频流，允许摄像机传感器预热，
# 并初始化到目前为止写入磁盘的示例面总数
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

#  视频流的帧上循环
while True:
	# 从线程视频流中获取帧，
	# 克隆它（以防我们想将其写入磁盘），
	# 然后调整帧的大小，以便更快地应用面部检测
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	# 检测灰度框中的面
	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

	# 在面部检测上画一个圈，然后把它们画在画框上
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# 显示输出帧
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# 按k键将帧写入磁盘
	if key == ord("k"):
		p = os.path.sep.join([args["output"], "{}.png".format(
			str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1

	# 按q键停止
	elif key == ord("q"):
		break
# 清理
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()