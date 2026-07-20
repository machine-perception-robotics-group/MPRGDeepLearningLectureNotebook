# Python Environment

Google Colaboratoryのノートブックの開発環境とできるだけ同じ環境を構築します。

Linux (Ubuntu) 上で動作させることを想定しています。その他のOS上で動作させる場合は、適宜修正してください。


## Docker

### サービス構成

`docker-compose.yml`では以下の2つのサービスを定義しています。どちらか一方を選んで起動してください（同時に起動することも可能です）。

| サービス名 | 内容 | Jupyter (ホスト側ポート) | TensorBoard (ホスト側ポート) |
| --- | --- | --- | --- |
| `lecturenotebook` | 標準環境（PyTorch + 画像処理系ライブラリ） | 10283 | 6006 |
| `lecturenotebook-claude` | `lecturenotebook`に加えてClaude Code CLI（`claude`コマンド）をインストール | 10284 | 6007 |

### 実行方法

`containers/docker`ディレクトリ内で以下を実行して起動してください。以下は`lecturenotebook`を使う場合の例です。`lecturenotebook-claude`を使う場合はサービス名を読み替えてください。

```bash
# UID/GID を合わせる（初回のみ）
echo "DOCKER_UID=$(id -u)" >> .env
echo "DOCKER_GID=$(id -g)" >> .env

# 初回、またはDockerfile変更後はビルド
docker compose build lecturenotebook

# 起動（バックグラウンド; -d を付けるとバックグラウンド起動になり、ターミナルがブロックされない）
docker compose up -d lecturenotebook

# コンテナに入る（bash）
docker compose exec lecturenotebook bash

# Jupyter / TensorBoard を使う場合
# ホスト側からは http://localhost:10283 でアクセスできる（docker-compose.yml のポートマッピングによる）
docker compose exec lecturenotebook jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

# 停止・削除
docker compose down lecturenotebook

# ログ確認
docker compose logs -f lecturenotebook
```

### Claude Code を使う場合（`lecturenotebook-claude`）

コンテナ内で開発しながらClaude Codeを使いたい場合は、`lecturenotebook-claude`サービスを使ってください。使い方は`lecturenotebook`と同様で、サービス名を`lecturenotebook-claude`に読み替えるだけです。

```bash
docker compose build lecturenotebook-claude
docker compose up -d lecturenotebook-claude
docker compose exec lecturenotebook-claude bash

# コンテナ内で Claude Code を起動
docker compose exec lecturenotebook-claude claude
```

初回起動時はブラウザでのログインが必要です。ログイン情報（`~/.claude`など）はコンテナに永続化していないため、コンテナを再作成すると再ログインが必要になります。


## Google Colaboratory上の開発環境（2026.07.20時点）

**OS**

```bash
$ lsb_release -a
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 22.04.5 LTS
Release:	22.04
Codename:	jammy
```

**CPU**

```bash
$ cat /proc/cpuinfo | grep 'model name' | uniq
$ cat /proc/cpuinfo | grep 'processor' | uniq
model name	: Intel(R) Xeon(R) CPU @ 2.00GHz
processor	: 0
processor	: 1
```

**Memory**

```bash
$ cat /proc/meminfo | grep 'MemTotal'
$ cat /proc/meminfo | grep 'MemAvailable'
MemTotal:       13286936 kB
MemAvailable:   12118592 kB
```

**GPU**

```bash
$ nvidia-smi
Mon Jul 20 06:53:50 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   46C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

**Python**

```python
import sys
print(sys.version)
3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]
```

**Python Modules**

```bash
$ pip freeze
(pip_freeze_all.txtに記載)
```
