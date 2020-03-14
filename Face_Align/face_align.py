'''
Idea is to rotate the face about the center point between the eyes to align the face
https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
Link above explains code for FaceAligner
'''
import numpy as np
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
from PIL import Image

'''
Process:
1. Given an image, get_frontal_face_detector() detects face and returns the bounding box of face.
2. For each detected face, apply face align using the landmark features from dlib's predictor and FaceAlign module from imutils.
'''
def faceAlign(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat') # for face landmark detection 
    fa = FaceAligner(predictor,desiredFaceWidth=244,desiredFaceHeight=244)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # generate gray scaled version
    rects = detector(gray,2) # may be potentially replaced with MTCNN
    aligned = [] # array of aligned faces 
    for rect in rects:
        (x,y,w,h) = rect_to_bb(rect)
        face_aligned = fa.align(image,gray,rect)
        aligned.append(face_aligned)
    return aligned

''' testing '''
image = cv2.imread('./face_align_test1.jpg')
aligned = faceAlign(image)
for x in aligned:
    plt.imshow(x)
    plt.show()
