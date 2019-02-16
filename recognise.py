import tensorflow as tf
import numpy as np
import facenet
from detect import Detector
from align import detect_face
import cv2
import argparse
import os
import threading
from os import listdir
from os.path import isfile, join

#disabling tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# parser = argparse.ArgumentParser()
# parser.add_argument("--img1", type = str, required=True)
# parser.add_argument("--img2", type = str, required=True)
# args = parser.parse_args()
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160
images_placeholder = 0
embeddings = 0
phase_train_placeholder = 0
embedding_size = 0
sess=0
pnet=0
rnet=0
onet=0
class Recognition:
    
    def initialise(self,filename):
        # some constants kept as default from facenet
        global pnet,rnet,onet,threshold,factor,minsize,margin,sess,images_placeholder,phase_train_placeholder,embedding_size,embeddings
        sess = tf.Session()
        
        # read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
        pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

        # read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
        facenet.load_model("20170512-110547/20170512-110547.pb")

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        obj=Recognition
        obj.printResults(self,filename)

    def getFace(self,img):
        obj=Recognition()
        faces = []
        img_size = np.asarray(img.shape)[0:2]
        global pnet,rnet,onet,threshold,factor,minsize,margin,sess,images_placeholder,phase_train_placeholder,embedding_size,embeddings
        # print(minsize,"minsize")
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if not len(bounding_boxes) == 0:
            for face in bounding_boxes:
                if face[4] > 0.50:
                    det = np.squeeze(face[0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                    prewhitened = facenet.prewhiten(resized)
                    faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':obj.getEmbedding(prewhitened)})
        return faces
    def getEmbedding(self,resized):
        global pnet,rnet,onet,threshold,factor,minsize,margin,sess,images_placeholder,phase_train_placeholder,embedding_size,embeddings
        reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
        feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
        embedding = sess.run(embeddings, feed_dict=feed_dict)
        # print("returning embeddings")
        return embedding

    def compare2face(self,img1,img2):
        obj=Recognition()
        global pnet,rnet,onet,threshold,factor,minsize,margin,sess,images_placeholder,phase_train_placeholder,embedding_size,embeddings
        
        face1 = obj.getFace(img1)
        face2 = obj.getFace(img2)
        if face1 and face2:
            # calculate Euclidean distance
            dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
            return dist
        return -1

    def printResults(self,filename):
        obj=Recognition()
        # print("inside print")
        onlyfiles = [f for f in listdir("images") if isfile(join("images", f))]
        img1 = cv2.imread("image.jpg")
        min=1
        f1="Not identified.jpg"
        for f in onlyfiles:
            # print(f)
            img2 = cv2.imread("images/"+str(f))
            distance = obj.compare2face(img1, img2)
            threshold = 1.0    # set yourself to meet your requirement
            # print("distance = "+str(distance) ,"to ",str(f[:-4]))
            # print("Result = " + ("same person" if distance <= threshold else "not same person"))
            if distance<=threshold and distance != -1 and distance< min:
                min=distance
                f1=f
        print("Face recognised",str(f1[:-4]))
        cv2.rectangle(img1,(0,img1.shape[0]),(img1.shape[1],0),(0,255,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img1,str(f1[:-4]),(30,img1.shape[0]-20), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.imwrite("result.jpg",img1)
                
                # return [1,str(f[:-4])]
    
        # return [0,""]

obj2=Detector()
# obj2.detection()
obj1=Recognition()
obj1.initialise("")
while True:
    t1 = threading.Thread(target=obj2.detection())
    t2 = threading.Thread(target=obj1.printResults(""))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
# while True:
#     obj1=Recognition()
    
#     t1.start() 
#     t1.join()
#     result,name=obj1.printResults("")
#     if result==1:
#         print("Face Recognised ",name)
#     # obj2.detection()
    