import os
import cv2
import numpy as np

from face_detector import FaceDetectorDlib


def test_detect():
  face_detector = FaceDetectorDlib()

  image = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "samples/3.jpg"
  )

  print("image path is: " + image)
  test_image = cv2.imread(image, cv2.IMREAD_COLOR)
  # test_image = np.array(test_image)
  faces = face_detector.detect(test_image)

  for face in faces:
    cv2.rectangle(test_image,(int(face.x),int(face.y)),(int(face.x + face.w), int(face.y + face.h)), (0,255,0),3)

  window_name = "image"
  cv2.namedWindow(window_name, cv2.WND_PROP_AUTOSIZE)
  cv2.startWindowThread()

  cv2.imshow('image', test_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.waitKey(1)
  print("done showing face annotated image!")

  for face in faces:
    print(face.face_landmark)

  print("done")


test_detect()
