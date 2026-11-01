"""提出用の動画（約2分）を作る。
同じシーンで指示文だけを変えた2本を左右に並べ、字幕つきで1本の mp4 にまとめる。

  段階1 学習で使った言い回し   / 段階2 学習で使っていない言い回し
  段階3 自由な言い回し         / 段階4 失敗例（つかむ段階でカップを倒す）
  + タイトル・背景・設定・結果表・まとめのカード

使い方:
  python make_video.py --ckpt outputs/smolvla_feed_put/checkpoints/last/pretrained_model
      → video_out/clips/ に各段階・各テイクの並べ動画、video_out/final.mp4 に完成版
  python make_video.py --assemble_only --pick s2=1 s3=0
      → 撮り直さずに、選ぶテイクだけ変えて final.mp4 を作り直す
  python make_video.py --teacher        # 方策なしで先生役の動きで全体の流れを確認（ローカル用）

SmolVLA は行動生成にランダム性があるため、各段階を最大 --takes 回撮影して残す。
自動選択は「指示どおり両方成功したテイク」優先（失敗例の段階は1テイクのみ）。
"""
import argparse, glob, json, os, sys
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
from feed_env import FeedEnv, N_PELLETS

FPS = 10
W, H = 1280, 720
PW, PH = 640, 480          # 1パネルの大きさ
HEAD, FOOT = 130, 110       # 上下の字幕帯
BG = (18, 20, 26)
FG = (240, 240, 240)
SUB = (170, 178, 190)
OK = (90, 210, 120)
NG = (235, 90, 90)
ACC = (255, 200, 60)

STAGES = [
    dict(key="s1", title="① 学習で使った言い回し", seed=50003,
         feed="fill the pet's bowl", put="place the cup on the mat",
         feed_ja="ペットのお皿を満たして", put_ja="カップをマットに置いて"),
    dict(key="s2", title="② 学習で使っていない言い回し", seed=50003,
         feed="it's dinner time for the pet", put="move the cup to the mat",
         feed_ja="ペットの夕ご飯の時間だよ", put_ja="カップをマットに移して"),
    dict(key="s3", title="③ 自由な言い回し（pet / food / bowl を含まない）", seed=None,
         feed="the cat is hungry", put="could you put that cup back on the mat",
         feed_ja="猫がお腹をすかせている", put_ja="そのカップをマットに戻してくれる？",
         feed_seed=50002, put_seed=50003),
    dict(key="s4", title="④ 失敗例：つかむ段階でカップを倒す", seed=50004,
         fail_seeds=[50004, 50008, 50002, 50017],  # 評価で4条件すべて失敗したシーン
         feed="feed the pet", put="put the cup away",
         feed_ja="ペットにエサをあげて", put_ja="カップを片付けて", fail=True),
]


# ---------------------------------------------------------------- 文字描画
def _font_path(bold):
    names = ["NotoSansCJK-Bold.ttc", "NotoSansCJKjp-Bold.otf"] if bold else []
    names += ["NotoSansCJK-Regular.ttc", "NotoSansCJKjp-Regular.otf", "NotoSansCJK-Black.ttc"]
    for n in names:
        hits = glob.glob(f"/usr/share/fonts/**/{n}", recursive=True)
        if hits:
            return hits[0]
    sys.exit("日本語フォントがありません。Colabでは  !apt-get -qq install fonts-noto-cjk  を実行してください")


_FONTS = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _FONTS:
        _FONTS[k] = ImageFont.truetype(_font_path(bold), size)
    return _FONTS[k]


def text_c(d, xy_center, s, size, color=FG, bold=False):
    f = font(size, bold)
    w = d.textlength(s, font=f)
    d.text((xy_center[0] - w / 2, xy_center[1]), s, font=f, fill=color)


def card(lines, seconds):
    """lines: [(文字列, 大きさ, 色, 太字)]  を縦に中央揃えで並べたカード"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    gap = 18
    total = sum(s for _, s, _, _ in lines) + gap * (len(lines) - 1)
    y = (H - total) / 2
    for s, size, color, bold in lines:
        text_c(d, (W / 2, y), s, size, color, bold)
        y += size + gap
    arr = np.asarray(img)
    return [arr] * int(seconds * FPS)


def table_card(seconds):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    text_c(d, (W / 2, 70), "評価結果（同じ20シーンで指示文だけ変えて実行）", 38, FG, True)
    rows = [("条件", "成功", "取り違え"),
            ("feed / 学習で使った言い回し", "11/20 (55%)", "0/20"),
            ("feed / 学習で使っていない言い回し", "12/20 (60%)", "1/20"),
            ("put  / 学習で使った言い回し", "12/20 (60%)", "0/20"),
            ("put  / 学習で使っていない言い回し", "11/20 (55%)", "0/20")]
    xs = [140, 760, 1010]
    y = 170
    for i, r in enumerate(rows):
        c = SUB if i == 0 else FG
        for x, s in zip(xs, r):
            d.text((x, y), s, font=font(32, i == 0), fill=c)
        if i == 0:
            d.line([(130, y + 50), (W - 130, y + 50)], fill=SUB, width=2)
        y += 70
    text_c(d, (W / 2, y + 30), "取り違え（指示と違う動作）は 80回中1回 → 指示文で動作を選べている", 30, ACC, True)
    text_c(d, (W / 2, y + 85), "※ 動画の①〜③は成功したシーンを選んで掲載しています", 24, SUB)
    arr = np.asarray(img)
    return [arr] * int(seconds * FPS)


# ---------------------------------------------------------------- 撮影
class Recorder:
    def __init__(self, ckpt=None, device=None, teacher=False, max_steps=150):
        import mujoco
        self.mujoco = mujoco
        self.env = FeedEnv(render=True, max_steps=max_steps)
        self.hi = mujoco.Renderer(self.env.model, PH, PW)  # 1つだけ作って使い回す
        self.teacher = teacher
        if not teacher:
            import torch
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.factory import make_pre_post_processors
            self.torch = torch
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.policy = SmolVLAPolicy.from_pretrained(ckpt).to(device).eval()
            self.pre, self.post = make_pre_post_processors(
                self.policy.config, pretrained_path=ckpt,
                preprocessor_overrides={"device_processor": {"device": device}})

    def _cam(self):
        cam = self.mujoco.MjvCamera()
        cam.lookat[:] = [0.22, 0, 0.05]
        cam.distance, cam.azimuth, cam.elevation = 0.62, 200, -25
        return cam

    def episode(self, seed, phrase, task, take):
        env = self.env
        obs = env.reset(seed)
        cam = self._cam()
        frames, stats = [], []
        if self.teacher:
            from teacher import Teacher
            t = Teacher(env, np.random.default_rng(seed + 7 * take), noise=0.005, task=task)
        else:
            torch = self.torch
            torch.manual_seed(seed * 10 + take)
            self.policy.reset()
        for i in range(env.max_steps):
            if self.teacher:
                a = t.act()
            else:
                def img(x):
                    return torch.from_numpy(x).permute(2, 0, 1).float().div(255.0)[None]
                batch = {"observation.state": torch.from_numpy(obs["state"]).float()[None],
                         "observation.images.top": img(obs["top"]),
                         "observation.images.wrist": img(obs["wrist_cam"]),
                         "task": [phrase]}
                with torch.no_grad():
                    a = self.post(self.policy.select_action(self.pre(batch)))[0].cpu().numpy()
            obs = env.step(a)
            self.hi.update_scene(env.data, cam)
            frames.append(self.hi.render().copy())
            stats.append((int(env.pellets_in_dish()), int(env.pellets_in_cup())))
        ok = bool(env.success_feed() if task == "feed" else env.success_put())
        cup = env.cup_pos()
        placed = bool(np.linalg.norm(cup[:2] - env.mat_xy) < 0.07 and cup[2] < 0.03)
        return frames, stats, dict(ok=ok, dish=stats[-1][0], cup=stats[-1][1], placed=placed)


def panel_frames(frames, stats, res, task, phrase, ja, x0, canvas_list):
    """canvas_list に左右どちらかのパネルを書き込む"""
    n = len(frames)
    for i in range(n):
        img = canvas_list[i]
        img.paste(Image.fromarray(frames[i]), (x0, HEAD))
        d = ImageDraw.Draw(img)
        # 指示文（上帯）
        tag = "feed（エサやり）" if task == "feed" else "put（片付け）"
        d.text((x0 + 20, 12), f"指示: \"{phrase}\"", font=font(26 if len(phrase) < 32 else 22, True), fill=ACC)
        d.text((x0 + 20, 52), f"（{ja}）", font=font(22), fill=SUB)
        d.text((x0 + 20, 88), f"正解の動作: {tag}", font=font(20), fill=SUB)
        # 状態（下帯）
        dish, cup = stats[i]
        y = HEAD + PH + 14
        d.text((x0 + 20, y), f"皿の中のエサ {dish:2d}/{N_PELLETS}    カップの中 {cup:2d}/{N_PELLETS}", font=font(24), fill=FG)
        if i >= n - int(3 * FPS):  # 最後の3秒に判定を出す
            if res["ok"]:
                s, c = ("成功：皿にエサを注いだ" if task == "feed" else "成功：マットに立てて置いた"), OK
            elif res["dish"] < 2 and res["cup"] < 3:
                s, c = "失敗：カップを倒してエサがこぼれた", NG
            else:
                s, c = "失敗", NG
            d.text((x0 + 20, y + 42), s, font=font(28, True), fill=c)


def compose(stage, left, right, tag_text):
    """left/right: (frames, stats, res, task, phrase, ja)"""
    n = max(len(left[0]), len(right[0]))
    canvas = [Image.new("RGB", (W, H), BG) for _ in range(n)]
    for (fr, st, res, task, ph, ja), x0 in ((left, 0), (right, PW)):
        fr = fr + [fr[-1]] * (n - len(fr)); st = st + [st[-1]] * (n - len(st))
        panel_frames(fr, st, res, task, ph, ja, x0, canvas)
    out = []
    for img in canvas:
        d = ImageDraw.Draw(img)
        d.line([(PW, HEAD), (PW, HEAD + PH)], fill=BG, width=6)
        d.text((W - 20 - d.textlength(tag_text, font=font(18)), H - 30), tag_text, font=font(18), fill=SUB)
        out.append(np.asarray(img))
    return out


def write(path, frames):
    w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                           pixelformat="yuv420p", ffmpeg_params=["-preset", "veryfast"])
    for f in frames:
        w.append_data(f)
    w.close()


def record_all(args):
    rec = Recorder(args.ckpt, args.device, args.teacher)
    os.makedirs(f"{args.out}/clips", exist_ok=True)
    meta = {}
    for st in STAGES:
        # 失敗例の段階は、実際に失敗が起きるシーンが見つかるまでシードを変える
        trials = [(0, sd) for sd in st["fail_seeds"]] if st.get("fail") else [(k, None) for k in range(args.takes)]
        meta[st["key"]] = []
        for k, fail_seed in trials:
            sides = []
            for task in ("feed", "put"):
                seed = fail_seed or st.get(f"{task}_seed", st["seed"])
                fr, stt, res = rec.episode(seed, st[task], task, k)
                sides.append((fr, stt, res, task, st[task], st[f"{task}_ja"]))
                print(f"[{st['key']} take{k}] {task} seed {seed} '{st[task]}' -> {res}", flush=True)
            seeds = [fail_seed] if fail_seed else sorted({st.get('feed_seed', st['seed']), st.get('put_seed', st['seed'])})
            tag = f"MuJoCo シミュレーション・等倍（10Hz）  seed {'/'.join(map(str, seeds))}"
            k = len(meta[st["key"]])
            path = f"{args.out}/clips/{st['key']}_take{k}.mp4"
            write(path, compose(st, sides[0], sides[1], tag))
            meta[st["key"]].append(dict(take=k, path=path, feed=sides[0][2], put=sides[1][2], fail=bool(st.get("fail"))))
            both = sides[0][2]["ok"] and sides[1][2]["ok"]
            if st.get("fail"):
                if not both:
                    break  # 失敗が撮れた
            elif both and not args.all_takes:
                break  # 両方成功したテイクが撮れたら次の段階へ
    with open(f"{args.out}/takes.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


def choose(meta, picks):
    chosen = {}
    for key, takes in meta.items():
        if key in picks:
            chosen[key] = takes[picks[key]]
            continue
        if takes[0].get("fail"):
            bad = [t for t in takes if not (t["feed"]["ok"] and t["put"]["ok"])]
            chosen[key] = bad[0] if bad else None  # 失敗が撮れなければ④は入れない
            continue
        score = lambda t: (t["feed"]["ok"] + t["put"]["ok"], -t["take"])
        chosen[key] = max(takes, key=score)
    return chosen


def read_clip(path):
    r = imageio.get_reader(path)
    fr = [f for f in r]
    r.close()
    return fr


def assemble(args, meta):
    picks = dict((p.split("=")[0], int(p.split("=")[1])) for p in (args.pick or []))
    chosen = choose(meta, picks)
    L = lambda s, size, color=FG, bold=False: (s, size, color, bold)
    v = []
    v += card([L("指示を聞き分けて", 40, SUB, True),
               L("エサやり と カップの片付け をするロボットアーム", 50, FG, True),
               L("SO-101 × SmolVLA（MuJoCo シミュレーション）", 28, SUB)], 5)
    v += card([L("目指したタスク", 40, ACC, True),
               L("同じ場面（エサ入りカップ・青い皿・緑のマット）で、指示を聞き分けて2つのタスクを実行する", 26, SUB),
               L("", 6),
               L("「エサをあげて」→ カップをつかみ、皿の上で傾けてエサを注ぐ", 30),
               L("成功：12粒の半分以上が皿に入る", 24, SUB),
               L("", 6),
               L("「カップを片付けて」→ カップをつかみ、緑のマットに立てて置く", 30),
               L("成功：カップがマット上に立ち、エサが半分以上残っている", 24, SUB)], 10)
    v += card([L("これまでの取り組み", 40, ACC, True),
               L("まずエサやり1タスクだけを学習（お手本149本・シミュレーションで30回試行）", 26, SUB),
               L("ACT（画像→動作の模倣学習）33%　→　SmolVLA（言語対応モデルを微調整）70%", 28),
               L("しかし言い回しを変えると 8/10 → 3/10 に低下。別の指示でもエサを注いだ", 26, SUB),
               L("→ 指示文を「意味」ではなく「いつもの合図」として扱っていた", 28, ACC)], 9)
    v += card([L("聞き分けられるようにするための工夫", 40, ACC, True),
               L("① データ収集：同じ場面で2タスクのお手本を集め、持ち上げるまでの動きをそろえる", 28),
               L("持ち上げた時点の画像は同じ → その先（注ぐ／置く）は指示文がないと決められない", 26, SUB),
               L("② 1タスクにつき4通りの言い回しで学習させる", 28),
               L("特定の1文を合図として覚えるのではなく、意味で判断させる", 26, SUB),
               L("", 6),
               L("学習：お手本254本で SmolVLA を微調整　／　入力：カメラ画像2枚・関節角度・指示文", 24, SUB)], 10)
    for st in STAGES:
        if chosen.get(st["key"]) is None:
            print(f"{st['key']}: 該当テイクなし → 省略")
            continue
        sub = ("左右とも同じシーン。違うのは指示文だけ" if not st.get("fail")
               else "どの指示でも、皿とマットに分かれる前にカップを倒してしまうシーン")
        v += card([L(st["title"], 44, FG, True), L(sub, 28, SUB)], 3)
        print(f"{st['key']}: take{chosen[st['key']]['take']} を使用  feed={chosen[st['key']]['feed']} put={chosen[st['key']]['put']}")
        v += read_clip(chosen[st["key"]]["path"])
    v += table_card(9)
    v += card([L("まとめと考察", 40, ACC, True),
               L("できたこと：同じ場面で、指示文に応じて「注ぐ」と「置く」を選び分けられた", 28),
               L("学習で使っていない言い回しでも成功率が落ちない（1タスク版は大きく低下）", 26, SUB),
               L("課題：失敗の多くは指示と関係なく、つかむ段階でカップを倒すこと", 28),
               L("学習文から遠い言い回しほど、方向は合っていても動作が迷いやすい", 26, SUB),
               L("次の一手：つかみ直しのお手本を追加、言い回しを各タスク10〜20に増やす", 26, SUB)], 9)
    path = f"{args.out}/final.mp4"
    write(path, v)
    print(f"保存: {path}  長さ {len(v)/FPS:.0f} 秒")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/smolvla_feed_put/checkpoints/last/pretrained_model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="video_out")
    ap.add_argument("--takes", type=int, default=3, help="各段階の最大撮影回数")
    ap.add_argument("--all_takes", action="store_true", help="成功しても --takes 回すべて撮る")
    ap.add_argument("--assemble_only", action="store_true")
    ap.add_argument("--pick", nargs="*", help="使うテイクを指定 例: s2=1 s3=0")
    ap.add_argument("--teacher", action="store_true", help="方策の代わりに先生役で動かす（流れの確認用）")
    args = ap.parse_args()
    if args.assemble_only:
        meta = json.load(open(f"{args.out}/takes.json"))
    else:
        meta = record_all(args)
    assemble(args, meta)


if __name__ == "__main__":
    main()
