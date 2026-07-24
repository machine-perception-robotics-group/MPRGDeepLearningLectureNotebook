<div align="center">

# MPRG Deep Learning Lecture Notebook

**画像認識・深層学習を基礎から学ぶための Jupyter Notebook 集**

[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

</div>

---

## 概要

本リポジトリは，中部大学 MPRG (Machine Perception & Robotics Group) が公開している画像認識・深層学習を実践的に学ぶための Jupyter Notebook コレクションです．

各ノートブックは以下の2通りの方法で実行できます．

- **Google Colaboratory で実行する**：各ノートブックの `Open in Colab` バッジをクリックすると，ブラウザ上ですぐに実行できます．
- **ローカル環境で実行する**：Docker を用いた開発環境を用意しています．詳細は [`containers/README.md`](containers/README.md) を参照してください．

## 免責事項

本リポジトリは，MPRGの教員および学生が有志で作成したものです．そのため，バグやミス，ライブラリ等のバージョンアップに伴う動作不具合などが含まれている場合があります．できるだけ修正対応するように努めていますが，修正等の対応が間に合わない場合があることをご了承ください．

旧MPRG Deep Learning Lecture Notebook（v1）から，情報の古くなったノートブックや未完成だったノートブックなどは削除しています．古いノートブックを参照したい場合は，[release/v1.0ブランチ](https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/tree/release/v1.0)を参照してください．

## 目次

- [MPRG Deep Learning Lecture Notebook](#mprg-deep-learning-lecture-notebook)
  - [概要](#概要)
  - [免責事項](#免責事項)
  - [目次](#目次)
  - [Python チュートリアル](#python-チュートリアル)
    - [Google Colaboratory・Pythonの使い方](#google-colaboratorypythonの使い方)
  - [初級編](#初級編)
    - [初級1：ゼロから理解するDeep Learning](#初級1ゼロから理解するdeep-learning)
    - [初級2：PyTorchで作るシンプルネットワーク](#初級2pytorchで作るシンプルネットワーク)
  - [中級編](#中級編)
    - [中級1：PyTorchで作るCNNモデル](#中級1pytorchで作るcnnモデル)
      - [画像分類のネットワーク構造](#画像分類のネットワーク構造)
      - [転移学習・ファインチューニング](#転移学習ファインチューニング)
      - [CNNの可視化・特徴表現の理解](#cnnの可視化特徴表現の理解)
      - [物体検出](#物体検出)
      - [セマンティックセグメンテーション](#セマンティックセグメンテーション)
      - [マルチタスク学習](#マルチタスク学習)
      - [モデルの効率化・知識の伝達](#モデルの効率化知識の伝達)
      - [ラベルなしデータを活用した学習](#ラベルなしデータを活用した学習)
    - [中級2：PyTorchで作る深層生成モデル](#中級2pytorchで作る深層生成モデル)
    - [中級3：PyTorchで作る再帰型ネットワーク](#中級3pytorchで作る再帰型ネットワーク)
    - [中級4：PyTorchで作る強化学習](#中級4pytorchで作る強化学習)
    - [中級5：PyTorchで作るグラフニューラルネットワーク](#中級5pytorchで作るグラフニューラルネットワーク)
  - [上級編](#上級編)
    - [上級1：PyTorchで作るTransformerモデル](#上級1pytorchで作るtransformerモデル)
    - [上級2：PyTorchで作るVision Transformerモデル](#上級2pytorchで作るvision-transformerモデル)
    - [上級3：PyTorchで作る視覚言語モデル](#上級3pytorchで作る視覚言語モデル)
    - [上級4：PyTorchで作る拡散モデル](#上級4pytorchで作る拡散モデル)
  - [その他](#その他)
    - [ハイパーパラメータの探索](#ハイパーパラメータの探索)
    - [機械学習の基礎](#機械学習の基礎)
    - [プログラム練習帳](#プログラム練習帳)

---

## Python チュートリアル

### Google Colaboratory・Pythonの使い方

ノートブックのプログラムを実行する際に使用するGoogle Colaboratoryや，Python, Numpy, PyTorchなどのプログラミングに関する基本的な使い方をまとめたノートブックです．

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [Google Colaboratoryの動作確認](00_tutorial/operation_check.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/00_tutorial/operation_check.ipynb) |
| 2 | [PythonプログミングとNumPy](00_tutorial/python_and_numpy.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/00_tutorial/python_and_numpy.ipynb) |
| 3 | [PyTorchの基本操作](00_tutorial/pytorch_basics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/00_tutorial/pytorch_basics.ipynb) |
| 4 | [PyTorchにおけるCPUとGPU（CUDA）の使い方](00_tutorial/pytorch_device.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/00_tutorial/pytorch_device.ipynb) |

## 初級編

### 初級1：ゼロから理解するDeep Learning

スクラッチで基本的なニューラルネットワークモデルや学習スクリプトを記述したノートブックです．
PyTorchなどの深層学習フレームワークを使用せず，Numpyを使用して実装しています．

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [最適化（GD, Momentum, AdaGrad, Adam)](01_dnn_scratch/optimization.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/01_dnn_scratch/optimization.ipynb) |
| 2 | [単純パーセプトロンによるAND回路の作成](01_dnn_scratch/perceptron_and.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/01_dnn_scratch/perceptron_and.ipynb) |
| 3 | [MLPによるXOR回路の作成](01_dnn_scratch/mlp_xor.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/01_dnn_scratch/mlp_xor.ipynb) |
| 4 | [MLPによる2クラス分類](01_dnn_scratch/mlp_bernoulli.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/01_dnn_scratch/mlp_bernoulli.ipynb) |
| 5 | [ミニバッチを用いたMLPの学習](01_dnn_scratch/mlp_bernoulli_minibatch.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/01_dnn_scratch/mlp_bernoulli_minibatch.ipynb) |
| 6 | [MLPによる多クラス分類（MNIST）](01_dnn_scratch/mlp_mnist.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/01_dnn_scratch/mlp_mnist.ipynb) |
| 7 | [正則化（Dropout）](01_dnn_scratch/dropout.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/01_dnn_scratch/dropout.ipynb) |
| 8 | [Batch Normalizationの導入](01_dnn_scratch/batchnorm.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/01_dnn_scratch/batchnorm.ipynb) |
| 9 | [im2colを用いた効率的な畳み込み処理](01_dnn_scratch/im2col.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/01_dnn_scratch/im2col.ipynb) |
| 10 | [CNNによる画像認識（MNIST, Numpy実装）](01_dnn_scratch/cnn_mnist.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/01_dnn_scratch/cnn_mnist.ipynb) |

### 初級2：PyTorchで作るシンプルネットワーク

PyTorchを使用して基本的な多層パーセプトロンや畳み込みニューラルネットワークなどのモデル構築・学習を記述したノートブックです．
また，モデルの学習に関連するデータ拡張やPyTorchのクラスなども実装しています．

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [MLPによる画像認識（MNIST, PyTorch実装）](02_dnn_simple_pytorch/mnist_mlp.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/mnist_mlp.ipynb) |
| 2 | [CNNによる画像認識（MNIST, PyTorch実装）](02_dnn_simple_pytorch/mnist_cnn.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/mnist_cnn.ipynb) |
| 3 | [CNNによる画像認識（CIFAR10, PyTorch実装）](02_dnn_simple_pytorch/cifar_cnn.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/cifar_cnn.ipynb) |
| 4 | [既存のデータセットの活用](02_dnn_simple_pytorch/existing_dataset.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/existing_dataset.ipynb) |
| 5 | [データ拡張（Data Augmentation）](02_dnn_simple_pytorch/augmentation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/augmentation.ipynb) |
| 6 | [データセットクラスの自作（Custom Dataset）](02_dnn_simple_pytorch/custom_dataset.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/custom_dataset.ipynb) |
| 7 | [nn.Sequentialと動的なネットワーク構築](02_dnn_simple_pytorch/dynamic_network.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/dynamic_network.ipynb) |
| 8 | [モデルの保存と読み込み](02_dnn_simple_pytorch/save_and_load.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/save_and_load.ipynb) |
| 9 | [Optimizerの比較とSchedulerによる学習率の調整](02_dnn_simple_pytorch/optimizer_scheduler.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/optimizer_scheduler.ipynb) |
| 10 | [再現性の確保](02_dnn_simple_pytorch/reproducibility.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/reproducibility.ipynb) |
| 11 | [DataLoaderのチューニング](02_dnn_simple_pytorch/dataloader_tuning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/dataloader_tuning.ipynb) |
| 12 | [混合精度学習（AMP）](02_dnn_simple_pytorch/amp.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/02_dnn_simple_pytorch/amp.ipynb) |

## 中級編

### 中級1：PyTorchで作るCNNモデル

CNNをベースとした様々なネットワーク構造・学習法を扱ったノートブック群です．画像分類・可視化・物体検出・セグメンテーション・マルチタスク学習・モデルの効率化・半教師あり/自己教師あり学習・転移学習といったテーマ別に分類しています．

#### 画像分類のネットワーク構造

CIFAR100を用いて，代表的な画像分類モデルをスクラッチ実装したノートブック群です．基本的な分類モデル，軽量なモデル，自動探索（NAS）に基づくモデルの順に構成しています．

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [AlexNet](11_cnn_pytorch/classification/alexnet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/alexnet.ipynb) |
| 2 | [VGG](11_cnn_pytorch/classification/vgg.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/vgg.ipynb) |
| 3 | [GoogLeNet](11_cnn_pytorch/classification/googlenet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/googlenet.ipynb) |
| 4 | [ResNet](11_cnn_pytorch/classification/resnet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/resnet.ipynb) |
| 5 | [WideResNet](11_cnn_pytorch/classification/wide_resnet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/wide_resnet.ipynb) |
| 6 | [ResNeXt](11_cnn_pytorch/classification/resnext.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/resnext.ipynb) |
| 7 | [DenseNet](11_cnn_pytorch/classification/densenet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/densenet.ipynb) |
| 8 | [SENet](11_cnn_pytorch/classification/senet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/senet.ipynb) |
| 9 | [MobileNet V1](11_cnn_pytorch/classification/mobilenet_v1.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/mobilenet_v1.ipynb) |
| 10 | [MobileNet V2](11_cnn_pytorch/classification/mobilenet_v2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/mobilenet_v2.ipynb) |
| 11 | [MobileNet V3](11_cnn_pytorch/classification/mobilenet_v3.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/mobilenet_v3.ipynb) |
| 12 | [SqueezeNet](11_cnn_pytorch/classification/squeezenet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/squeezenet.ipynb) |
| 13 | [EfficientNet](11_cnn_pytorch/classification/efficientnet_v1.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/efficientnet_v1.ipynb) |
| 14 | [EfficientNetV2](11_cnn_pytorch/classification/efficientnet_v2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/efficientnet_v2.ipynb) |
| 15 | [MnasNet](11_cnn_pytorch/classification/mnasnet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/mnasnet.ipynb) |
| 16 | [RegNet](11_cnn_pytorch/classification/regnet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/regnet.ipynb) |

#### 転移学習・ファインチューニング

torchvisionとTIMM (PyTorch Image Models) モジュールに実装されている代表的なネットワークモデルとその事前学習モデルの呼び出し方，またそれらを用いた転移学習やファインチューニングの方法についてまとめています．
| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [torchvisionによるモデル呼び出しと事前学習モデルの利用](11_cnn_pytorch/classification/torchvision_models.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/torchvision_models.ipynb) |
| 2 | [timmによるモデル呼び出しと事前学習モデルの利用](11_cnn_pytorch/classification/timm_models.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/timm_models.ipynb) |
| 3 | [事前学習モデルを用いた転移学習・ファインチューニング](11_cnn_pytorch/classification/transfer_learning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/classification/transfer_learning.ipynb) |

#### CNNの可視化・特徴表現の理解

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [CNNの可視化（CAM）](11_cnn_pytorch/02_cam.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/02_cam.ipynb) |
| 2 | [CNNの可視化（Grad-CAM）](11_cnn_pytorch/02_4_grad_cam.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/02_4_grad_cam.ipynb) |
| 3 | [Attention Branch Network（ABN）](11_cnn_pytorch/05_abn.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/05_abn.ipynb) |
| 4 | [誤差関数の変更による学習効果](11_cnn_pytorch/loss_func.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/loss_func.ipynb) |

#### 物体検出

PASCAL VOC 2007を用いて，two-stage・one-stage・anchor-free・keypointベースなど，代表的な物体検出モデルをスクラッチ実装したノートブック群です．

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [SSD](11_cnn_pytorch/detection/ssd.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/detection/ssd.ipynb) |
| 2 | [Faster R-CNN](11_cnn_pytorch/detection/faster_rcnn.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/detection/faster_rcnn.ipynb) |
| 3 | [FPN（Feature Pyramid Network）](11_cnn_pytorch/detection/fpn.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/detection/fpn.ipynb) |
| 4 | [RetinaNet（Focal Loss）](11_cnn_pytorch/detection/retinanet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/detection/retinanet.ipynb) |
| 5 | [CornerNet](11_cnn_pytorch/detection/cornernet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/detection/cornernet.ipynb) |
| 6 | [CenterNet](11_cnn_pytorch/detection/centernet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/detection/centernet.ipynb) |
| 7 | [EfficientDet](11_cnn_pytorch/detection/efficientdet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/detection/efficientdet.ipynb) |
| 8 | [FCOS（Anchor-Free）](11_cnn_pytorch/detection/fcos.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/detection/fcos.ipynb) |
| 9 | [YOLOv1](11_cnn_pytorch/detection/yolo_v1.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/detection/yolo_v1.ipynb) |
| 10 | [YOLOv3](11_cnn_pytorch/detection/yolo_v3.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/detection/yolo_v3.ipynb) |

#### セマンティックセグメンテーション

画像の画素単位でクラスを予測するセマンティックセグメンテーションのノートブックです．

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [SegNet](11_cnn_pytorch/08_segnet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/08_segnet.ipynb) |

#### マルチタスク学習

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [マルチタスク基礎（分類＋回帰）](11_cnn_pytorch/09_multitask_fundamental.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/09_multitask_fundamental.ipynb) |
| 2 | [マルチタスク応用（検出＋セグメンテーション）](11_cnn_pytorch/10_multitask_applied_mtdssd.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/10_multitask_applied_mtdssd.ipynb) |

#### モデルの効率化・知識の伝達

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [Knowledge Distillation](11_cnn_pytorch/10_knowledge_distillation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/10_knowledge_distillation.ipynb) |
| 2 | [Deep Mutual Learning](11_cnn_pytorch/11_deep_mutual_learning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/11_deep_mutual_learning.ipynb) |
| 3 | [枝刈り](11_cnn_pytorch/15_pruning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/15_pruning.ipynb) |

#### ラベルなしデータを活用した学習

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [半教師付き学習](11_cnn_pytorch/13_semi_supervised_learning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/13_semi_supervised_learning.ipynb) |
| 2 | [自己教師付き学習](11_cnn_pytorch/14_self_supervised_learning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/11_cnn_pytorch/14_self_supervised_learning.ipynb) |

### 中級2：PyTorchで作る深層生成モデル

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [Auto Encoderによる画像の復元とデノイジング](12_gan/autoencoder.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/autoencoder.ipynb) |
| 2 | [Variational Autoencoder (VAE)](12_gan/variational_autoencoder.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/variational_autoencoder.ipynb) |
| 3 | [繰り返し処理による異常検知](12_gan/anomaly_detection_vae.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/anomaly_detection_vae.ipynb) |
| 4 | [Generative Adversarial Networks (GAN)](12_gan/gan.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/gan.ipynb) |
| 5 | [Deep Convolutional GAN (DC-GAN)](12_gan/dcgan.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/dcgan.ipynb) |
| 6 | [Conditional GAN](12_gan/conditional_gan.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/conditional_gan.ipynb) |
| 7 | [Conditional DC-GAN](12_gan/conditional_dcgan.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/conditional_dcgan.ipynb) |
| 8 | [CycleGAN（スタイル変換）](12_gan/cycle_gan.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/cycle_gan.ipynb) |
| 9 | [BigGAN](12_gan/big_gan.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/big_gan.ipynb) |
| 10 | [StyleGAN](12_gan/style_gan.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/style_gan.ipynb) |

### 中級3：PyTorchで作る再帰型ネットワーク

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [Recurrent Neural Networkによる電力予測](13_rnn/01_03_RNN.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/13_rnn/01_03_RNN.ipynb) |
| 2 | [Recurrent Neural NetworkによるBitcoinの価格予測](13_rnn/RNN_bitcoin.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/13_rnn/RNN_bitcoin.ipynb) |
| 3 | [Encoder-Decoderによる計算機作成](13_rnn/04_Seq2Seq.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/13_rnn/04_Seq2Seq.ipynb) |
| 4 | [Attention Seq2seqによる計算機作成](13_rnn/05_Attention.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/13_rnn/05_Attention.ipynb) |
| 5 | [Attention Seq2seqによる日付変換](13_rnn/05_Attention_alpha.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/13_rnn/05_Attention_alpha.ipynb) |
| 7 | [Convolutional LSTMを用いた動画像予測](13_rnn/07_ConvLSTM.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/13_rnn/07_ConvLSTM.ipynb) |

### 中級4：PyTorchで作る強化学習

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [強化学習（Q学習とQ Network）によるCart Pole制御](14_rl/00_Q_Learning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/14_rl/00_Q_Learning.ipynb) |
| 2 | [DQN（クリッピング・リプレイ・ターゲットネットワーク）](14_rl/01_Deep_Q_Network.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/14_rl/01_Deep_Q_Network.ipynb) |
| 3 | [Policy gradient （DQNの改良）](14_rl/02_Policy_gradient.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/14_rl/02_Policy_gradient.ipynb) |
| 4 | [Actor-cltic](14_rl/03_Actor_Critic.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/14_rl/03_Actor_Critic.ipynb) |
| 5 | [Mask-Attention ](14_rl/04_Mask_Attention.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/14_rl/04_Mask_Attention.ipynb) |
| 6 | [DQNの応用例](14_rl/05_Deep_Q_Network_application.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/14_rl/05_Deep_Q_Network_application.ipynb) |

### 中級5：PyTorchで作るグラフニューラルネットワーク

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [グラフ表現](15_gcn/01_graph.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/15_gcn/01_graph.ipynb) |
| 2 | [GCNによるノード分類](15_gcn/02_node_classification_GCN.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/15_gcn/02_node_classification_GCN.ipynb) |
| 3 | [ST-GCNによる動作認識](15_gcn/03_action_recognition_ST_GCN.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/15_gcn/03_action_recognition_ST_GCN.ipynb) |
| 4 | [STA-GCNによる動作認識](15_gcn/04_action_recognition_STA-GCN.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/15_gcn/04_action_recognition_STA-GCN.ipynb) |
| 5 | [グラフ生成](15_gcn/05_graph_generation_DGMG.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/15_gcn/05_graph_generation_DGMG.ipynb) |


## 上級編

### 上級1：PyTorchで作るTransformerモデル

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 6 | [Transformerによる計算機作成](13_rnn/06_Transformer.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/13_rnn/06_Transformer.ipynb) |
| 6 | [BERT](13_rnn/06_Transformer.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/13_rnn/06_Transformer.ipynb) |


### 上級2：PyTorchで作るVision Transformerモデル

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [ViTの教師あり学習（フルスクラッチ・fine-tuning）](17_vit/01_vit.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/17_vit/01_vit.ipynb) |
| 2 | [MAE](17_vit/02_mae.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/17_vit/02_mae.ipynb) |

### 上級3：PyTorchで作る視覚言語モデル

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 3 | [CLIPによる画像のゼロショットクラス分類](17_vit/03_clip.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/17_vit/03_clip.ipynb) |
| 4 | [CLIPと言語モデルを組み合わせたMLLM](17_vit/04_mllm.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/17_vit/04_mllm.ipynb) |

### 上級4：PyTorchで作る拡散モデル

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 11 | [Diffusion Model](12_gan/denoising_Diffusion_Model.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/Diffusion_Model.ipynb) |
| 12 | [Denoising Diffusion Probabilistic Model](12_gan/denoising_diffusion_probabilistic_model.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/denoising_diffusion_probabilistic_model.ipynb) |
| 13 | [Latent Diffusion Model](12_gan/latent_diffusion_model.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/12_gan/latent_diffusion_model.ipynb) |

## その他

### ハイパーパラメータの探索

ネットワークの学習に用いるハイパーパラメータの探索方法についてまとめたノートブックです．

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [ハイパーパラメータの探索と検証データ](99_others/parameter_search.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/99_others/parameter_search.ipynb) |
| 2 | [Optunaによるハイパーパラメータ探索](99_others/optuna.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/99_others/optuna.ipynb) |

### 機械学習の基礎

深層学習以前の機械学習手法について使い方をまとめたノートブックです．
主にscikit-learnを使用して実装しています．

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [グラフを描画する](80_classical_ml/draw_graph.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/80_classical_ml/draw_graph.ipynb) |
| 2 | [ユークリッド距離を用いたクラス識別](80_classical_ml/clf_euc_dist.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/80_classical_ml/clf_euc_dist.ipynb) |
| 3 | [マハラノビス距離を用いたクラス識別](80_classical_ml/clf_mahal_dist.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/80_classical_ml/clf_mahal_dist.ipynb) |
| 4 | [k最近傍法による教師あり学習](80_classical_ml/kNN.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/80_classical_ml/kNN.ipynb) |
| 5 | [線形SVMによる教師あり学習](80_classical_ml/linear_svm.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/80_classical_ml/linear_svm.ipynb) |
| 6 | [非線形SVMによる教師あり学習](80_classical_ml/nonliear_svm.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/80_classical_ml/nonliear_svm.ipynb) |
| 7 | [AdaBoostによる教師あり学習](80_classical_ml/adaboost.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/80_classical_ml/adaboost.ipynb) |
| 8 | [RandomForestによる教師あり学習](80_classical_ml/random_forest.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/80_classical_ml/random_forest.ipynb) |
| 9 | [マテリアルズインフォマティクス](80_classical_ml/materials_info.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/80_classical_ml/materials_info.ipynb) |

### プログラム練習帳

| No. | ノートブック | Colab |
| :-: | :-- | :-: |
| 1 | [プログラム練習帳：画像認識](81_workbook/ProgramWorkBook_ImageClassfication.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/81_workbook/ProgramWorkBook_ImageClassfication.ipynb) |
| 2 | [プログラム練習帳：CSVファイル](81_workbook/ProgramWorkBook_CSVFile.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/develop-v2/81_workbook/ProgramWorkBook_CSVFile.ipynb) |

---

<div align="center">

© 2026 Machine Perception and Robotics Group, Chubu University. All rights reserved.

</div>
