#!/usr/bin/env bash
# SmolVLA（事前学習済み lerobot/smolvla_base）を「エサやり／片付け」の2タスクデータで微調整する。
# 前提: GPU推奨（CPUでは非常に遅い）、huggingface.co に接続できること。
# 環境変数で調整可: STEPS=20000 BATCH=32 bash train_smolvla.sh
# データ: ./data/so101_feed_put（LeRobot形式、feed/put の2タスク、言い回し各4種）
set -euo pipefail
cd "$(dirname "$0")"

STEPS=${STEPS:-15000}   # 2タスクなので1回目(10000)より長め
BATCH=${BATCH:-16}   # T4(16GB)想定。A100ならBATCH=64でもよい

# smolvla_base は camera1〜camera3 という名前でカメラを受け取る。
# 俯瞰カメラ→camera1、手首カメラ→camera2 に対応づける（camera3 は無しで可）。
# この対応は学習後の前処理パイプラインにも保存されるので、評価時は top/wrist のまま渡せばよい。
RENAME_MAP='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/so101_feed_put \
  --dataset.root=./data/so101_feed_put \
  --rename_map="$RENAME_MAP" \
  --batch_size=$BATCH \
  --steps=$STEPS \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir=./outputs/smolvla_feed_put \
  "$@"
