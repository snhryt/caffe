#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
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
      help="テストデータ（入力画像）のファイルパス" + 
           "or 入力画像群が含まれるディレクトリパス"
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
      "--mean_img_filepath",
      help=".npy 形式の平均画像のファイルパス"
  ) 
  parser.add_argument(
      "--ext",
      default="png",
      help="画像の拡張子（\".\"は不要）"
  )
  parser.add_argument(
      "--max_num",
      default=10,
      help="識別結果で上位何個まで示すか"
  )
  args = parser.parse_args()

  net = caffe.Net(
      args.model_def_filepath,
      args.model_weights_filepath,
      caffe.TEST)
  caffe.set_mode_gpu()

  # transformer 
  transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
  transformer.set_transpose("data", (2,0,1))
  transformer.set_raw_scale("data", 255)
  if args.mean_img_filepath:
    mu = np.load(args.mean_img_filepath)
    transformer.set_mean("data", mu)
  
  # 入力データの読みとり
  input_imgs, filepaths = [], []
  if os.path.isdir(args.input_path):
    print("Loading directory: %s" % args.input_path)
    for im_f in glob.glob(args.input_path + "/*." + args.ext):
      input_imgs.append(caffe.io.load_image(im_f, color=False))
      filepaths.append(im_f)
  else:
    print("Loading image file: %s" % args.input_path)
    input_imgs = caffe.io.load_image(args.input_path,  color=False)
    filenpaths = args.input_pat
  print("Classifying %d input_imgs." % len(input_imgs))

  labels = np.loadtxt(args.labels_filepath, str, delimiter="\t")
        
  # csvへの書き込み
  output_file = open(args.output_filepath, "w")
  if os.path.exists(args.output_filepath.rsplit("/", 1)[0]) == False:
    os.makedirs(args.output_filepath.rsplit("/", 1)[0])
  csv_writer = csv.writer(output_file)
  csv_header = ["full filepath"]
  for i in range(0, args.max_num):
	  csv_header.append("rank " + str(i + 1))
	  csv_header.append("confidence[%]")
  csv_writer.writerow(csv_header)

  for i in range(0, len(input_imgs)):
    transformed_img = transformer.preprocess("data", input_imgs[i])
    net.blobs["data"].data[...] = transformed_img
    output = net.forward()
    output_probs = output["prob"][0]
    top_inds = output_probs.argsort()[::-1][:args.max_num]

    print("<%s>" % filepaths[i])
    csv_line = [filepaths[i]]
    for j in range(0, len(top_inds)):
      output_prob = output_probs[top_inds[j]] * 100
      output_prob = round(output_prob, ndigits=1)
      if output_prob == 0.0:
        break
      print("#%d | %s | %3.1f%%" % (j+1, labels[top_inds[j]], output_prob))
      csv_line.append(labels[top_inds[j]])
      csv_line.append(output_prob)
    csv_writer.writerow(csv_line)
  output_file.close()

if __name__ == "__main__":
  main(sys.argv)