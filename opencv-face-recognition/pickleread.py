import pickle

#文件读取
g = open('D://PycharmProjects//face_recognition//opencv-face-recognition//output//recognizer.pickle', 'rb')
e=pickle.load(g)
print(e)
