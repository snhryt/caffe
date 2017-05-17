# Ubuntu16.04でCaffeをGPUモードで動かすまで（2017/5/17時点）
参考→http://yusuke-ujitoko.hatenablog.com/entry/2016/06/19/203443

## ホームディレクトリ名の英語化
これをやっておかないと色々不便。
```bash
$ env LANGUAGE=C LC_MESSAGES=C xdg-user-dirs-gtk-update
```

## General Dependencies
```bash
$ sudo apt install libprotobuf-dev \
     libleveldb-dev \
     libsnappy-dev \
     libopencv-dev \
     libhdf5-serial-dev \
     protobuf-compiler
$ sudo apt install --no-install-recommends libboost-all-dev
```

## OpenCVのインストール
参考→http://shibafu3.hatenablog.com/entry/2017/03/28/164125
```bash
$ sudo apt install build-essential
$ sudo apt install cmake \
     git \
     ffmpeg \
     libopencv-dev \
     libgtk-3-dev \
     python-numpy \
     python3-numpy \
     libdc1394-22 \
     libdc1394-22-dev \
     libjpeg-dev \
     libpng12-dev \
     libtiff5-dev \
     libjasper-dev \
     libavcodec-dev \
     libavformat-dev \
     libswscale-dev \
     libxine2-dev \
     libgstreamer1.0-dev \
     libgstreamer-plugins-base1.0-dev \
     libtbb-dev \
     qtbase5-dev \
     libfaac-dev \
     libmp3lame-dev \
     libopencore-amrnb-dev \
     libopencore-amrwb-dev \
     libtheora-dev \
     libvorbis-dev \
     libxvidcore-dev x264
$ cd ${OPENCV_DIR} # OpenCVを置くディレクトリ。自分の場合はホームディレクトリ
$ git clone https://github.com/opencv/opencv.git
$ git clone https://github.com/opencv/opencv_contrib.git
$ cd opencv/
$ mkdir build
$ cd build/
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
     -D CMAKE_INSTALLPREFIX=/usr/local \
     -D WITH_TBB=ON \
     -D WITH_V4L=ON \
     -D WITH_QT=ON \
     -D WITH_OPENGL=ON \
     ..
$ make # めっちゃ時間かかる
$ sudo make install
$ sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
$ sudo ldconfig
$ sudo apt update
$ sudo reboot # 一旦再起動
```

## NVIDIAのGPUドライバのインストール
http://www.nvidia.co.jp/Download/index.aspx?lang=jp
```bash
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
$ sudo apt install nvidia-xxx #調べたバージョンに変更
$ sudo apt install mesa-common-dev freeglut3-dev
$ sudo reboot # 一旦再起動
```

## CUDA Toolkitのインストール
https://developer.nvidia.com/cuda-toolkit
```bash
$ sudo dpkg -i cuda-repo-ubuntu1604-xxxxxxxx.deb
$ sudo apt update
$ sudo apt install cuda
```

## cuDNNのインストール
https://developer.nvidia.com/cudnn
<br>CUDAのバージョンと対応したcuDNNをダウンロードする。
```bash
$ cd ${CUDNN_DIR} # cuDNNをダウンロードしたディレクトリ。たぶんデフォルトでは ~/Downloads
$ tar -zxf cudnn-xxxxxxxx.tgz
$ sudo cp -r ~/cuda/ /usr/local/cudnn-xxx # ~/cuda というディレクトリにコピー。別にホームディレクトリじゃなくてもいい
$ sudo cp -a ~/cuda/lib64/* /usr/local/lib/
$ sudo cp -a ~/cuda/include/* /usr/local/include/
$ sudo ldconfig
$ sudo reboot
```

## Caffeのインストール
Caffeのダウンロードおよび依存環境のインストール。
```bash
$ sudo apt install libatlas-base-dev \
     libopenblas-base \
     python-dev \
     libgflags-dev \
     libgoogle-glog-dev \
     liblmdb-dev
$ cd ${CAFFE_HOME} # Caffeをクローンするディレクトリ
$ git clone https://github.com/BVLC/caffe.git
$ cd caffe
$ cp Makefile.config.example Makefile.config
```

パスを通す。
```bash
$ cd ${CAFFE_HOME}
$ export PATH=$PATH:/usr/local/cuda/bin # このまま入力すればいい
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH # このまま入力すればいい
$ export LD_LIBRARY_PATH=/usr/local/cudnn-xxx:$LD_LIBRARY_PATH # xxx はインストールしたcuDNNのバージョンに合わせて変更
# $ export LD_LIBRARY_PATH=/usr/lib/openblas-base:$LD_LIBRARY_PATH
```
## Makefile.configの編集
- `#USE_CUDNN := 1` のアンコメントアウト
- `#OPENCV_VERSION := 3` のアンコメントアウト
- `INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include`
  -> `INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/`

## Caffeのmakeとtest
```bash
$ cd /usr/lib/x86_64-linux-gnu/
$ sudo ln -s libhdf5_serial.so.10.1.0 libhdf5.so # make でエラーが出ないように
$ sudo ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so # make でエラーが出ないように
$ cd ${CAFFE_HOME}
$ make all -j8 # 並列化
$ make test -j8
$ make runtest
```

## Pythonとのラッピング
```bash
$ cd ${CAFFE_HOME}/python
$ sudo apt install python-pip
$ sudo pip install -r requirements.txt
$ cd ../
$ make pycaffe -j8
$ export PYTHONPATH=${CAFFE_HOME}/python/:$PYTHONPATH # ${CAFFE_HOME}はCaffeをクローンしたディレクトリに変更
```

## パスの記録
次回からもパスが通った状態で起動できるように`~/.profile`に以下を追記。
```bash
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cudnn-xxx:$LD_LIBRARY_PATH
export PYTHONPATH=${CAFFE_HOME}/python/:$PYTHONPATH # ${CAFFE_HOME}はCaffeをクローンしたディレクトリに変更
# export LD_LIBRARY_PATH=/usr/lib/openblas-base:$LD_LIBRARY_PATH
```
## 最後に
MNISTの学習チュートリアルの実行。
```bash
$ cd ${CAFFE_HOME}
$ ./data/mnist/get_mnist.sh
$ ./examples/mnist/create_mnist.sh
$ ./examples/mnist/train_lenet.sh
```
