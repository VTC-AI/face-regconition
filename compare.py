import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input


w, h = 224, 224

def preprocess_image(image):
  img = load_img(image, target_size=(w, h))
  img = img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = preprocess_input(img)
  return img

def cosin_distance(vector1, vector2):
  a = np.matmul(np.transpose(vector1), vector2)
  b = np.sum(np.multiply(vector1, vector1))
  c = np.sum(np.multiply(vector2, vector2))
  return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def verify_face(list_vector, img_vector_input):
  min = 1
  for i in range(len(list_vector)):
    if cosin_distance(list_vector[i], img_vector_input) <= min:
      min = cosin_distance(list_vector[i], img_vector_input)
      i_min = i
  
  return min, i_min
