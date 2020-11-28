# MPRG Deep Learning Lecture Notebook


## ゼロから理解するDeep Learning（初級編）
1.  [PythonプログミングとNumPy](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/01_python_and_numpy.ipynb)
2.  [最適化（GD, Momentum, AdaGrad, Adam)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/02_optimization.ipynb)
3.  [MLPによる2クラス分類](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/03_mlp_bernoulli.ipynb)
4.  [ミニバッチを用いたMLPの学習](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/04_mlp_bernoulli_minibatch.ipynb)
5.  [MLPによる多クラス分類（MNIST）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/05_mlp_mnist.ipynb)
6.  [正則化（Dropout）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/06_dropout.ipynb)
7.  [Batch Normalizationの導入](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/07_batchnorm.ipynb)
8.  [im2colを用いた効率的な畳み込み処理](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/08_im2col.ipynb)
9.  [CNNを用いた画像認識（MNIST）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/09_cnn_mnist.ipynb)
10. [GPUを用いたCNNによる画像認識](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/10_cnn_pytorch.ipynb)
10. [データ拡張（Data Augmentation）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/11_augmentation.ipynb)
11. [ハイパーパラメータの探索と検証データ](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/01_dnn_scratch/12_parameter_search.ipynb)


## PyTorchで作るモダンネットワーク（中級編1）
1.  [CNN（データーローダ、データ拡張）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/01_cnn_dataloader_augmentation.ipynb)
2.  [CNNの可視化（CAM）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/02_cam.ipynb)
2.5.  [CNNの可視化（Grad-CAM）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/02_5_grad_cam.ipynb)
3.  [ResNet（スキップ構造）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/03_resnet.ipynb)
4.  [SENet](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/04_senet.ipynb)
5.  [Attention Branch Network（ABN）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/05_abn.ipynb)
6.  EfficientNet
7.  Single Shot Object Detector（YOLOv1, SSD）
8.  [SegNet](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/08_segnet.ipynb)
9.  [マルチタスク基礎（分類＋回帰）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/09_multitask_fundamental.ipynb)
10. マルチタスク応用（検出＋セグメンテーション）
11. [Knowledge Distillation](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/10_knowledge_distillation.ipynb)
12. [Deep Mutual Learning](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/11_deep_mutual_learning.ipynb)
13. [半教師付き学習](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/13_semi_supervised_learning.ipynb)
14. 自己教師付き学習
15. 弱教師付き学習
16. Few-Shot学習


## PyTorchで作るGAN（中級編2）
1.  [Variational Autoencoder (VAE)，UMAPによる特徴ベクトルの次元削減](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/12_gan/01_Variational_autoencoder.ipynb)
2.  [Generative Adversarial Networks (GAN)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/12_gan/02_GAN.ipynb)
3.  [Deep Convolutional GAN (DC-GAN)](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/12_gan/03_Deep_Convolutional_GAN.ipynb)
4.  [Conditional DC-GAN](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/12_gan/04_conditional_DC_GAN.ipynb)
5.  [CycleGAN（スタイル変換）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/12_gan/05_CycleGAN.ipynb)
6.  [StyleGAN](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/12_gan/06_StyleGAN.ipynb), [BigGAN](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/12_gan/06_BigGAN.ipynb)


## PyTorchで作る再帰型ネットワーク（中級編3）
1.  [RNN (LSTM, GRU)による電力予測](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/13_rnn/01_03_RNN.ipynb)
2.  [Encoder-Decoderによる計算機作成](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/13_rnn/04_Seq2seq.ipynb)
3.  [Attention機構による日付変換](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/13_rnn/05_Attention.ipynb)
4.  [Convolutional LSTMを用いた動画像予測](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/13_rnn/06_ConvLSTM.ipynb)


## PyTorchで作る強化学習（中級編4）
1.  [DQN（クリッピング・リプレイ・ターゲットネットワーク）](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/14_rl/01_Deep_Q_Network.ipynb)
2.  Policy gradient （DQNの改良）
3.  [Actor-cltic](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/14_rl/03_Actor_Critic.ipynb)
4.  Mask-Attention


## PyTorchで作るGCN（中級編5）
1.  [グラフ表現](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/01_graph.ipynb)
2.  [GCNによるノード分類](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/02_node_classification_GCN.ipynb)
3.  [ST-GCNによる動作認識](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/03_action_recognition_ST_GCN.ipynb)
4.  [STA-GCNによる動作認識](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/04_action_recognition_STA_GCN.ipynb)
5.  [グラフ生成](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/05_graph_generation_DGMG.ipynb)
