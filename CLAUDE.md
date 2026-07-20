# MPRG Deep Learning Lecture Notebookのアップデート（v1 --> v2）

このプロジェクトでは、[MPRG Deep Learning Lecture Notebook](https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook)をアップデートし、バグ修正やリファクタリング、新たなノートブックの追加などを目的とします。


## やりたいこと1

ローカルでの開発環境を提供したいです。
具体的には、Dockerのイメージを提供したいです。

### PythonおよびPythonモジュールのバージョンについて

その際、これらの環境では、Google Colaboratoryが提供するPython環境とできるだけ同じモジュールバージョンを提供するようにイメージビルド用のファイルを定義する。
現時点でのGoogle Colaboratoryの環境に関する情報は、`containers/README.md`および`containers/pip_freeze_all.txt`に記載している。

**注意点**

- ColaboratoryとCUDA + PyTorchのバージョンの整合性が取れない場合があるが、PyTorch (`torch`) のバージョンを優先する。
- `containers/pip_freeze_all.txt`に記載されている全てのモジュールをインストールする必要はない。

### Claude codeをインストールした開発用Dockerイメージの作成



## やりたいこと2

TBA


## その他の注意点

- gitの操作はユーザーが行います。Claude codeでは実行しないでください。
