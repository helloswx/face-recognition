# python C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\train_model.py --embeddings  C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\output\embeddings.pickle --recognizer  C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\output\recognizer.pickle --le  C:\Users\swx\PycharmProjects\jiqixuexiswx\opencv-face-recognition\output\le.pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
# 命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")#序列化路径
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")#基于SVM的人脸识别输出模型
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")#标签编码器输出的文件路径
args = vars(ap.parse_args())

# 使用先前生成并序列化的嵌入
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# 初始化，编码标签
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

#初始化SVM，训练模型
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# 把人脸识别模型和标签编码器输出到磁盘
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()