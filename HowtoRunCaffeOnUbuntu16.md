# Ubuntu16.04でCaffeをGPUモードで動かすまで（2017/5/17時点）
参考→http://yusuke-ujitoko.hatenablog.com/entry/2016/06/19/203443

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
$ cd ~
$ mkdir github
$ cd github
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
```
```bash
$ sudo apt update
$ sudo apt install cuda
```

## cuDNNのインストール
https://developer.nvidia.com/cudnn
<br>CUDAのバージョンと対応させる
```bash
$ tar -zxf cudnn-xxxxxxxx.tgz
$ sudo cp -r cuda/ /usr/local/cudnn-xxxx
```
```bash
$ sudo cp -a cuda/lib64/* /usr/local/lib/
$ sudo cp -a cuda/include/* /usr/local/include/
# $ sudo cp lib64/libcudnn* /usr/local/cuda/lib64
# $ sudo cp include/cudnn.h /usr/local/cuda/include
$ sudo ldconfig
$ sudo reboot
```

## Caffeのインストール
Caffeのクローニングおよび依存環境のインストール。ここではCaffeはホームディレクトリにクローンしてある。
```bash
$ cd ~
$ sudo apt install libatlas-base-dev \
     libopenblas-base \
     python-dev \
     libgflags-dev \
     libgoogle-glog-dev \
     liblmdb-dev
$ cd ~
$ git clone https://github.com/BVLC/caffe.git
$ cd caffe
$ cp Makefile.config.example Makefile.config
```
Caffeの`make`でエラーが出ないように
```bash
$ cd /usr/lib/x86_64-linux-gnu/
$ sudo ln -s libhdf5_serial.so.10.1.0 libhdf5.so
$ sudo ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so
```
パスを通す
```bash
$ cd ~/caffe
$ export PATH=$PATH:/usr/local/cuda/bin
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=/usr/local/cudnn-8.0:$LD_LIBRARY_PATH
# $ export LD_LIBRARY_PATH=/usr/lib/openblas-base:$LD_LIBRARY_PATH
```
## Makefile.configの編集
- #USE_CUDNN := 1 のアンコメントアウト
- #OPENCV_VERSION := 3 のアンコメントアウト
- INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
  -> INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/

## Caffeのmakeとtest
```bash
$ make all -j8 # 並列化
$ make test -j8
$ make runtest
```

## Pythonとのラッピング
```bash
$ cd python
$ sudo apt install python-pip
$ sudo pip install -r requirements.txt
$ cd ../
$ make pycaffe -j8
$ export PYTHONPATH=~/caffe/python/:$PYTHONPATH # Caffeをクローンしたディレクトリに併せて変更
```

## パスの記録
次回からもパスが通った状態で起動できるように`~/.profile`に以下を追記
```bash
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cudnn-xxxx:$LD_LIBRARY_PATH
export PYTHONPATH=~/caffe/python/:$PYTHONPATH
# export LD_LIBRARY_PATH=/usr/lib/openblas-base:$LD_LIBRARY_PATH
```
## 最後に
MNISTの学習チュートリアルの実行
```bash
$ cd ~/caffe
$ ./data/mnist/get_mnist.sh
$ ./examples/mnist/create_mnist.sh
$ ./examples/mnist/train_lenet.sh
```
