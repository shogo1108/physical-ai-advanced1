# 01 エサやり1タスク — ACT

カップに入ったエサ（12粒）を皿に注ぐタスクを、ACT（Action Chunking with Transformers）で学習しました。

## 手順

```bash
python gen_demos.py 1000 100 raw & python gen_demos.py 2000 100 raw   # 先生役でお手本を生成（成功したものだけ保存 → 149本）
python to_lerobot.py raw data/so101_feed      # LeRobot形式に変換（02で使用）
python train_act.py --raw raw --out ckpt --steps 3000
python eval_act.py --ckpt ckpt --seeds 50000 --n 15   # 学習に使っていないシードで評価
python eval_act.py --ckpt ckpt --seeds 60000 --n 15
```

## 設定

- 入力：カメラ画像2枚（上・手首、96×128）＋関節角度6個。指示文は使わない
- ACT：dim_model 256、チャンク長 20、画像のランダムシフトで水増し、ImageNet重みなし、CPUで3,000ステップ（約40分）

## 結果

- 学習に使っていない30シードで **成功率 33%（10/30）**（シード50000〜: 3/15、60000〜: 7/15）
- 失敗の多くは、カップを運べているが皿から外れた位置で注いでしまうもの

## ファイル

| ファイル | 役割 |
|---|---|
| `feed_env.py` | MuJoCo のエサやり環境（カップ・エサ粒・皿、成功判定） |
| `ik.py` / `teacher.py` | 逆運動学と、正確な物体位置を使う先生役 |
| `gen_demos.py` / `to_lerobot.py` | お手本の生成と LeRobot 形式への変換 |
| `train_act.py` / `eval_act.py` | ACT の学習と評価 |
