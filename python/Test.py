# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cv2
#from scipy.misc import imread
import glob

import caffe

def plot_sin():
  x = np.arange(0, 3 * np.pi, 0.1)
  y_sin = np.sin(x)
  y_cos = np.cos(x)

  plt.plot(x, y_sin)
  plt.plot(x, y_cos)
  plt.show()

img_path = '/media/snhryt/Data/Research_Master/caffe/data/mnist/10k_images/00001.png'

def load_and_show_img():
  img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  cv2.imshow('image', img)
  cv2.waitKey(0)
  #plt.imshow(img)
  #plt.show()

def main():
  caffe_root = '/media/snhryt/Data/Research_Master/caffe'
  model_def = caffe_root + '/examples/mnist/lenet.prototxt'
  model_weights = caffe_root + '/examples/mnist/lenet_iter_50000.caffemodel'

  net = caffe.Net(model_def, model_weights, caffe.TEST)
  caffe.set_mode_gpu()

  mu = np.load(caffe_root + '/examples/mnist/mnist_mean.npy')
  #mu = mu.mean(1).mean(1)
  #print 'mean-substracted values: ', zip('BGR', mu)

  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_mean('data', mu)
  transformer.set_raw_scale('data', 255)
  #transformer.set_channel_swap('data', (2, 1, 0))
  #net.blobs['data'].reshape(64, 1, 28, 28)

  for i in range(1, 10001):
    num_padded = '{0:05d}'.format(i)
    img_filename = str(num_padded) + '.png'
    img_filepath = caffe_root + '/data/mnist/10k_images/' + img_filename
    #print img_filepath

    #img = caffe.io.load_image(caffe_root + '/data/mnist/10k_images/00002.png', color = False)
    img = caffe.io.load_image(img_filepath, color = False)

    #print 'img.ndim = ', img.ndim, ', img.shape = ', img.shape
    transformed_img = transformer.preprocess('data', img)
    #print 'transformed_img.ndim = ', transformed_img.ndim, ', transformed_img.shape = ', transformed_img.shape
    net.blobs['data'].data[...] = transformed_img

    output = net.forward()
    output_prob = output['prob'][0]
    output_prob *= 100
    #print 'predicted class: ', output_prob.argmax()

    labels_file = caffe_root + '/data/mnist/mnist_words.txt'
    labels = np.loadtxt(labels_file, str, delimiter = '\t')
    #print 'output label: ', labels[output_prob.argmax()]

    top_inds = output_prob.argsort()[::-1][:5]
    if output_prob[top_inds[0]] < 90.0:
      print img_filename, ': parobabilities and labels'
      print(zip(output_prob[top_inds], labels[top_inds]))
  """
  pycaffe_dir = os.path.dirname(__file__)

  net = caffe.Classifier(
      '/media/snhryt/Data/Research_Master/caffe/examples/mnist/lenet.prototxt',
      '/media/snhryt/Data/Research_Master/caffe/examples/mnist/lenet_iter_50000.caffemodel')
  caffe.set_mode_gpu()
  caffe.set_phase_test()
  net.set_raw_scale('data', 255)
  scores = net.predict(
      [caffe.io.load_image(img_path, color = False, )],
      oversample = False )
  print(scores)

  net = caffe.Net(
      '/media/snhryt/Data/Research_Master/caffe/examples/mnist/lenet.prototxt',
      '/media/snhryt/Data/Research_Master/caffe/examples/mnist/lenet_iter_50000.caffemodel',
      caffe.TEST)
  caffe.set_mode_gpu()
  img = caffe.io.load_image(img_path)
  #preprocessed = preprocess(img)
  net.blobs['data'].data[...] = img
  net.forward()
  result = net.blobs['prob'].data[0, :]
  """

if __name__ == '__main__':
  #plot_sin()
  #load_and_show_img()
  main()
