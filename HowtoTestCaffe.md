# Caffeモデルをダウンロードして利用する（2017/5/18時点）
参考
- https://techblog.yahoo.co.jp/programming/caffe-intro/
- http://punyo-er-met.hateblo.jp/entry/2016/03/13/214712

## ネットワーク定義ファイルのダウンロード
[Model ZOO](https://github.com/BVLC/caffe/wiki/Model-Zoo)で[VGG16(?)](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md)のページを探して、`README.md`のGist IDおよびcaffemodelのURLを確認する。
```bash
$ cd ~/caffe
$ mkdir work
$ cd work
$ ../scripts/download_model_from_gist.sh 211839e770f7b538e2d8 . # README.md の gist_id を入力
$ cd 211839e770f7b538e2d8
$ wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel # README.md の caffemodel_url を入力
```

## 画像データセットの準備
物体認識のデータセットの1つである[Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)をダウンロード。加えて、シェルスクリプトから関連ファイルもダウンロード。
```bash
$ cd ~/caffe/data
$ wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
$ tar xf 101_ObjectCategories.tar.gz
$ ./ilsvrc12/get_ilsvrc_aux.sh
```

## 分類結果の可視化プログラムの作成
`~/caffe/python`内に`show_result.py`として以下のプログラムを作成。
```python
#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, numpy

categories = numpy.loadtxt(sys.argv[1], str, delimiter="\t")
scores = numpy.load(sys.argv[2])
top_k = 3
prediction = zip(scores[0].tolist(), categories)
prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
for rank, (score, name) in enumerate(prediction[:top_k], start=1):
    print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
```

## スクリプトの実行
```bash
$ cd ~/caffe
$ ./python/classify.py \
    --raw_scale 224 \
    --model_def ./work/211839e770f7b538e2d8/VGG_ILSVRC_16_layers_deploy.prototxt \
    --pretrained_model ./work/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel \
    --mean_file '' \
    ./data/101_ObjectCategories/airplanes/image_0001.jpg \
    result.npy
$ python ./python/show_result.py ./data/ilsvrc12/synset_words.txt result.npy
```
