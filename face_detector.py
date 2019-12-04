import abc
import dlib
import os
import numpy as np

class FaceDetector(object):
  def __init__(self):
    pass

  def __str__(self):
    return self.name()

  @abc.abstractmethod
  def name(self):
    return 'detector'
  
  @abc.abstractmethod
  def dectect(self, npimg):
    pass

class BoundingBox:
  __slots__ = ['x', 'y', 'w', 'h', 'score', 'face_name', 'face_score', 'face_feature', 'face_landmark', 'face_roi']

  def __init__(self, x=0, y=0, w=0, h=0, score=0.0):
    self.x = x
    self.y = y
    self.w = w
    self.h = h
    self.score = score

    self.face_name = ''
    self.face_score = 0.0
    self.face_feature = None
    self.face_landmark = None
    self.face_roi = None
  
  def __repr__(self):
    return f'{self.x} {self.y} {self.w} {self.h} score={self.score} name={self.face_name}'

class FaceDetectorDlib(FaceDetector):
  
  NAME = 'detector_dlib'

  def __init__(self):
    super(FaceDetectorDlib, self).__init__()
    self.detector = dlib.get_frontal_face_detector()
    predictor_path = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), 
      './dlib/shape_predictor_68_face_landmarks.dat'
    )

    self.predictor = dlib.shape_predictor(predictor_path)
    # self.upsample_scale = 1
  
  def name(self):
    return FaceDetectorDlib.NAME

  def detect(self, npimg):
    dets, scores, idx = self.detector.run(npimg, 1, -1)
    faces = []

    for det, score in zip(dets, scores):
      if score < 0.8:
        continue

      x = max(det.left(), 0)
      y = max(det.top(), 0)
      w = min(det.right() - det.left(), npimg.shape[1] - x)
      h = min(det.bottom() - det.top(), npimg.shape[0] -y)

      if w <= 1 or h <= 1:
        continue
      
      bbox = BoundingBox(x, y, w, h, scores)

      bbox.face_landmark = self.detect_landmark(npimg, det)

      faces.append(bbox)

    faces = sorted(faces, key=lambda x: x.score, reverse=True)

    return faces
  
  def detect_landmark(self, npimg, det):
    shape = self.predictor(npimg, det)
    coords = np.zeros((68, 2), dtype=np.int)

    for i in range(0, 68):
      coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

