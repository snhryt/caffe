#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import argparse
import csv
from sklearn import manifold
from sklearn import datasets
import matplotlib.pyplot as plt
import glob
import caffe

def main(argv):
  parent_dirpath = "MyWork/2fonts_2class/"
  model_def_filepath = parent_dirpath + "caffenet_deploy.prototxt"
  model_weight_filepath = parent_dirpath + "caffenet_iter_30000.caffemodel"
  mean_array = np.load(parent_dirpath + "mean.npy")

  net = caffe.Classifier(
    model_def_filepath,
    model_weight_filepath,
    mean=mean_array,
    raw_scale=255
  )
  net.blobs['data'].reshape(1, 1, 100, 100)
  net.reshape()

  print "\n[Loading images]"
  counter = 0
  layer = "conv5"
  #num_output, width, height = 96, 23, 23
  #num_output, width, height = 256, 11, 11
  #num_output, width, height = 384, 5, 5
  #num_output, width, height = 384, 5, 5
  num_output, width, height = 256, 5, 5
  input_imgs, indices = [], []
  input_txt_filepath = parent_dirpath + "validation.txt"
  f = open(input_txt_filepath)
  line = f.readline()
  while line:
    input_img_filepath = line.rsplit(" ")[0]
    index = line.rsplit(" ")[1]
    index = index.split("\n")[0]
    input_imgs.append(caffe.io.load_image(input_img_filepath, color=False))
    indices.append(index)
    if counter == 0:
      feature = net.blobs[layer].data
      print feature.shape
      feature = feature.reshape(1, num_output * width * height)
    else:
      r_feature = net.blobs[layer].data
      r_feature = r_feature.reshape(1, num_output * width * height)
      feature = np.vstack((feature, r_feature))     
    if counter % 100 == 0 and counter != 0:
      print(".. %d images are loaded" % counter)
      if counter == 500:
        break
    counter += 1 
    line = f.readline()
  f.close

  # t-SNE
  print "\n[Calculate t-SNE]"
  model = manifold.TSNE(n_components=2, perplexity=20.0, learning_rate=1000.0, n_iter=1000, verbose=3)
  tsne = model.fit_transform(feature)
  
  plt.figure(1, figsize=(8, 8))
  plt.clf()
  for i in range(1, len(indices)):
    if indices[i] == "0":
      plt.plot([tsne[i - 1, 0]], tsne[i, 0], tsne[i - 1, 1], tsne[i, 1], color="tomato", marker="o")
    else:
      plt.plot([tsne[i - 1, 0]], tsne[i, 0], tsne[i - 1, 1], tsne[i, 1], color="royalblue", marker="o")
  plt.show()


if __name__ == "__main__":
  main(sys.argv)