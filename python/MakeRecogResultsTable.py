#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import argparse
import csv
import matplotlib.pyplot as plt
import glob
import caffe

def main(argv):
  parser = argparse.ArgumentParser()
  # 絶対いるやつ
  parser.add_argument(
      "input_path",
      help="テストデータ（入力画像）のファイルパス or " + 
           "入力画像群が含まれるディレクトリパス"
  )
  parser.add_argument(
      "model_def_filepath",
      help="学習モデルが定義された .prototxt 形式のファイルパス"
  )
  parser.add_argument(
      "model_weights_filepath",
      help="学習後の重みが記された .caffemodel 形式のファイルパス"
  )
  parser.add_argument(
      "labels_filepath",
      help="データのラベルが記された .txt 形式のファイルパス"
  )
  parser.add_argument(
      "output_filepath",
      help="出力csvファイルのファイルパス"
  )
  # オプション
  parser.add_argument(
      "mean_img_filepath",
      help=".npy 形式の平均画像のファイルパス"
  ) 
  parset.add_argument(
      "images_dim",
      default='100x100',
      help="画像のサイズ \'<height>x<width>\'"
  )
  parser.add_argument(
      "ext",
      help="画像の拡張子（. から始まるやつ）"
  )
  parser.add_argument(
      "max_num",
      default=10,
      help="識別結果で上位何個まで示すか"
  )
  args = parser.parse_args()

  net = caffe.Net(args.model_def_filepath,
      args.model_weights_filepath,
      caffe.TEST)
  caffe.set_mode_gpu()

  mu = []
  if args.mean_img_filepath:
    mu = np.load(args.mean_img_filepath)

  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2,0,1))
  transformer.set_raw_scale('data', 255)
  if len(mu) != 0:
    transformer.set_mean('data', mu)
  
  # 入力データの読みとり
  if os.path.isdir(args.input_path):
    print("Loading directory: %s" % args.input_path)
    inputs, filenames = [], []
    for im_f in glob.glob(args.input_path + '/*.' + args.ext):
      inputs.append(caffe.io.load_image(im_f), color=False)
      filenames.append(im_f.split('/')[-1])
  else:
    print("Loading image file: %s" % args.input_path)
    inputs = [caffe.io.load_image(args.input_path), color=False]
    filenames = [glob.glob(args.input_path + '/*.' + args.ext).split('/')[-1]]
  print("Classifying %d inputs." % len(inputs))

  labels = np.loadtxt(args.labels_filepath, str, delimiter=' ')

  # 認識結果の格納
  for i in (0, len(inputs)):
    transformed_img = transformer.preprocess('data', inputs[i])
    net.blobs['data'].data[...] = transformed_img
    output = net.forward()
    output_prob = output['prob'][0]
    output_prob *= 100
    
    top_inds = output_prob.argsort()[::-1][:args.max_num]
    for j in range(0, len(top_inds)):
      if output_prob[top_inds[j]] == 0.0:
        break
      else:
        
  # csvへの書き込み
  csv_writer = csv.writer(open(args.output_filepath, 'w'))
  csv_header = ["full filepath", "filename"]
  for i in range(0, args.max_num):
	  csv_header.append("rank" + str(i + 1))
	  csv_header.append("confidence[%]")
  csv_writer.writerow(csv_header)

  for i in range(0, len(inputs)):
    csv_line = []
    for j in range(0, args.max_num):
      if output_prob[top_inds[i]] == 0.0:
        break
      else:
        print labels[top_inds[i]], '(', output_prob[top_inds[i]], '%)'



if __name__ == '__main__':
  main(sys.argv)