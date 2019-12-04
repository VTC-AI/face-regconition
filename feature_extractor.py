from __future__ import absolute_import, print_function
import PIL
import torch
import glob as bg
import numpy as np
from PIL import Image

# params
batch_size = 10
mean = (131.0912, 103.8827, 91.4953)

def load_data(path='', shape=None):
  short_size = 224.0
  crop_size = shape
  img = PIL.Image.open(path)
  im_shape = np.array(img.size) # (w, h)
  img = img.convert('RGB')

  ratio = float(short_size) / np.min(im_shape)
  img = img.resize(
    size=(int(np.ceil(im_shape[0] * ratio)), int(np.ceil(im_shape[1] * ratio))),
    resample=PIL.Image.BILINEAR
  )

  x = np.array(img)
  newshape = x.shape[:2]
  h_start = (newshape[0] - crop_size[0]) // 2
  w_start = (newshape[1] - crop_size[1]) // 2
  x = x[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
  x = x - mean
  return x

def chunks(l, n):
  for i in range(0, len(l), n):
    yield l[i:i+n]

def image_encoding(model, facepaths):
  print('==> compute image feature encoding')
  num_faces = len(facepaths)
  face_feats = np.empty((num_faces, 128))
  imgpaths = facepaths
  imgchunks = list(chunks(imgpaths, batch_size))

  for c, imgs in enumerate(imgchunks):
    im_array = np.array([load_data(path=i, shape=(224, 224, 3)) for i in imgs])
    f = model(torch.Tensor(im_array.transpose(0, 3, 1, 2)))[1].detach().cpu().numpy()[:, :, 0, 0]
    start = c * batch_size
    end = min((c + 1) * batch_size, num_faces)

    face_feats[start:end] = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True))
    if c % 50 == 0:
      print(f'-> finish encoding {c * batch_size}/{num_faces}')
  return face_feats
