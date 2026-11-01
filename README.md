# 指示を聞き分けて エサやり と カップの片付け をするロボットアーム

SO-101（SO-ARM101）を MuJoCo でシミュレーションし、模倣学習で次の2つのタスクを学習させました。

| 指示（例） | 動作 | 成功条件 |
|---|---|---|
| 「feed the pet」（エサをあげて） | カップをつかみ、皿の上で傾けてエサを注ぐ | 12粒の半分以上が皿に入る |
| 「put the cup away」（カップを片付けて） | カップをつかみ、緑のマットに立てて置く | カップがマット上に立ち、エサが半分以上残っている |

場面（エサ入りカップ・青い皿・緑のマット）は同じで、どちらの動作をするかは指示文で決まります。

## 段階と結果

| フォルダ | 内容 | 結果 |
|---|---|---|
| `01_act_feed/` | エサやり1タスクを ACT（画像→動作の模倣学習）で学習 | 成功率 33%（10/30） |
| `02_smolvla_feed/` | 同じデータで SmolVLA（言語対応の学習済みモデル）を微調整 | 成功率 70%（21/30）。ただし言い回しを変えると 8/10 → 3/10 |
| `03_smolvla_feed_put/` | 同じ場面で2タスク・各4通りの言い回しを学習し、指示で動作を切り替える | 成功 55〜60%、取り違え 1/80。学習に使っていない言い回しでも成功率が落ちない |

詳しい結果は各フォルダの README を参照してください。

## 工夫した点

1. **データ収集**：同じ場面で2タスクのお手本を集め、カップを持ち上げるまでの動きをタスク間でそろえた。持ち上げた時点の画像は同じになるため、その先（注ぐ／置く）は指示文がないと決められない。
2. **言い回しの多様化**：1タスクにつき4通りの言い回しで学習させ、特定の1文を合図として覚えるのではなく意味で判断させることを狙った。
3. **評価**：同じシーン（同じシード）に対して指示文だけを変えて実行し、成功率に加えて「取り違え」（指示と違う動作）を数えた。

## しくみ

- **環境**：MuJoCo 3 ＋ [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) の `robotstudio_so101`（Apache-2.0、`mujoco_menagerie/` に同梱）。エサは12粒の小球。
- **お手本データ**：シミュレーションの正確な物体位置を使うスクリプトの先生役（逆運動学）で動かし、成功したエピソードだけを保存（10 Hz）。
- **方策への入力**：カメラ画像2枚（上・手首、96×128）、関節角度6個、指示文。出力は6関節の目標角度。
- **学習**：[LeRobot](https://github.com/huggingface/lerobot) 0.6系。ACT はCPUで学習、SmolVLA は Google Colab の GPU で `lerobot/smolvla_base` を微調整。

## 動かし方（03 の例）

```bash
pip install "lerobot[smolvla,dataset,training]" mujoco imageio imageio-ffmpeg
cd 03_smolvla_feed_put
python gen_demos_lang.py 3000 170 raw          # お手本生成（成功したものだけ保存）
python to_lerobot.py raw data/so101_feed_put   # LeRobot形式に変換
STEPS=15000 BATCH=16 bash train_smolvla.sh     # 学習（GPU推奨）
python eval_smolvla_lang.py --ckpt outputs/smolvla_feed_put/checkpoints/last/pretrained_model
```

画面のない環境では `MUJOCO_GL=egl`（GPU）または `MUJOCO_GL=osmesa` を設定してください。Colab用ノートブックはGoogleドライブに置いたzipから展開する前提で書かれているので、リポジトリを使う場合は展開セルを `git clone` に置き換えてください。

## ライセンス

`mujoco_menagerie/` 以下は MuJoCo Menagerie のファイルで、Apache License 2.0 に従います。
