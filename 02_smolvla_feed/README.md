# SO-101 エサやり — SmolVLA版

ACT版（`01_act_feed/`）から分けた別プロジェクトです。シミュレーション環境、先生役スクリプト、デモデータはACT版と同じものを使い、方策だけをSmolVLAに差し替えます。

## 中身

| ファイル | 役割 |
|---|---|
| `smolvla_colab.ipynb` | Colabで上から実行するだけのノートブック（学習→評価→言い換えテスト→保存） |
| `train_smolvla.sh` | `lerobot-train` で `lerobot/smolvla_base` を微調整 |
| `eval_smolvla.py` | シミュレーションでロールアウトして成功率を測る。`--task` で指示文を変えられる |
| `data/so101_feed/` | LeRobot形式のデータセット（149エピソード、18,029フレーム、10fps、タスク文 "feed the pet"） |
| `feed_env.py` / `teacher.py` / `ik.py` | MuJoCoのエサやり環境、IKの先生役（ACT版と同一） |
| `gen_demos.py` / `to_lerobot.py` | デモの追加生成と LeRobot 形式への変換 |
| `mujoco_menagerie/robotstudio_so101/` | SO-101のモデル（DeepMind MuJoCo Menagerie、Apache-2.0） |

## Colabでの手順

1. `feed_smolvla.zip` を Google ドライブの `マイドライブ/physical_ai/` に置く
2. `smolvla_colab.ipynb` を Colab で開き、ランタイムを GPU（T4以上）にする
3. 上から順に実行

## 比較の基準（ACT版の結果）

- 学習に使っていない30シードで **成功率 33%（10/30）**
- 成功判定: 12粒のうち半分以上が皿に入る
- ACTはCPU・ImageNet重みなし・3,000ステップの条件。失敗の多くは「運べているが皿から外れた位置で注ぐ」

## SmolVLA版の結果（2026-11-01、Colab GPU、10,000ステップ）

- 学習に使っていない30シード（50000〜50029）で **成功率 70%（21/30）**。成功時はほぼ全粒が皿に入る
- 失敗9件中7件は粒0個。原因は **つかむ段階でカップを倒す** ことで、倒した後もカップなしで「運ぶ→注ぐ」動作を続ける（立て直しのデモがないため）
- ACTの弱点だった「注ぐ位置のずれ」はほぼ解消

言い換えテスト（同じ10シード 50000〜50009）:

| 指示文 | 成功 |
|---|---|
| "feed the pet"（学習時の文） | 8/10 |
| "give the dog its food"（言い換え） | 3/10 |
| "pick up the red cup"（別の指示） | 4/10（それでもエサを注ぐ） |

→ タスクが1文だけのデータでは、タスク文は意味ではなく「いつもの合図」として扱われている。言葉の理解を試す続きは `feed_smolvla_lang/` で行う。

## 既知の注意点

- **画像解像度**: デモ画像は 96×128 です。SmolVLAは内部で 512×512 に拡大（パディング付き）するので、細部が粗いままです。成功率が伸びない場合は、`feed_env.py` の `IMG_H, IMG_W` を 240×320 程度に上げて `gen_demos.py` → `to_lerobot.py` でデータを作り直すのが最初の改善候補です。
- **カメラ名**: `smolvla_base` はカメラを `camera1`〜`camera3` という名前で受け取るため、`train_smolvla.sh` で 俯瞰→camera1、手首→camera2 に対応づけています。この対応は前処理パイプラインに保存されるので、評価スクリプトには元の名前のまま渡せます。
- **Colabでの依存関係**: `lerobot[smolvla,dataset,training]` が必要（`smolvla` だけだと学習時に `av` / `accelerate` が足りない）。

## 言語条件付けを試すなら

今のデータはタスクが1つだけなので、ACTとの差は出にくいです。「カップを片付けて（"put the cup away"）」など2つ目のタスクを `teacher.py` に追加し、タスク文を変えて記録すると、言葉で動作を切り替える比較ができます。
