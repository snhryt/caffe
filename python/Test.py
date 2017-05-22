# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import glob
import caffe

def plot_sin():
  x = np.arange(0, 3 * np.pi, 0.1)
  y_sin = np.sin(x)
  y_cos = np.cos(x)

  plt.plot(x, y_sin)
  plt.plot(x, y_cos)
  plt.show()

def load_and_show_img():
  img_path = '/media/snhryt/Data/Research_Master/caffe/data/mnist/10k_images/00001.png'
  img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  cv2.imshow('image', img)
  cv2.waitKey(0)

def main():
  caffe_root = '/media/snhryt/Data/Research_Master/caffe'
  model_def = caffe_root + '/examples/mnist/lenet.prototxt'
  model_weights = caffe_root + '/examples/mnist/lenet_iter_50000.caffemodel'
  mean_array_file = caffe_root + '/examples/mnist/mnist_mean.npy'
  labels_file = caffe_root + '/data/mnist/mnist_words.txt'

  net = caffe.Net(model_def, model_weights, caffe.TEST)
  caffe.set_mode_gpu()

  mu = np.load(mean_array_file)
  #mu = mu.mean(1).mean(1)
  #print 'mean-substracted values: ', zip('BGR', mu)

  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_mean('data', mu)
  transformer.set_raw_scale('data', 255)
  #transformer.set_channel_swap('data', (2, 1, 0))
  #net.blobs['data'].reshape(64, 1, 28, 28)

  labels = np.loadtxt(labels_file, str, delimiter = '\t')

  for i in range(1, 10001):
    img_filename = str('{0:05d}'.format(i)) + '.png'
    img_filepath = caffe_root + '/data/mnist/10k_images/' + img_filename

    img = caffe.io.load_image(img_filepath, color = False)
    #print 'img.ndim = ', img.ndim, ', img.shape = ', img.shape
    transformed_img = transformer.preprocess('data', img)
    #print 'transformed_img.ndim = ', transformed_img.ndim, ', transformed_img.shape = ', transformed_img.shape
    net.blobs['data'].data[...] = transformed_img

    output = net.forward()
    output_prob = output['prob'][0]
    output_prob *= 100
    top_inds = output_prob.argsort()[::-1][:5]

    if output_prob[top_inds[0]] < 99.0:
      print '<', img_filename, '>'
      for i in range(0, len(top_inds)):
        if output_prob[top_inds[i]] == 0.0:
          break
        else:
          print labels[top_inds[i]], '(', output_prob[top_inds[i]], '%)'
      print '\n'

if __name__ == '__main__':
  #plot_sin()
  #load_and_show_img()
  main()
