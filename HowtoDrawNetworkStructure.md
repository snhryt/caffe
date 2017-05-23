# ネットワーク構造の可視化
参考→http://hirotaka-hachiya.hatenablog.com/entry/2015/05/09/172124
```bash
$ sudo apt install python-pip graphviz
$ sudo pip install graphviz pydot
$ cd {CAFFE_HOME}
$ python ./python/draw_net.py \
      ./examples/mnist/lenet.prototxt \
      ./lenet.png \
      --rankdir TB
```
rankdirオプションは4種類から選択可能
- BT: Bottom to Top
- TB: Top to Bottom
- RL: Right to Left
- LR: Left to Right
