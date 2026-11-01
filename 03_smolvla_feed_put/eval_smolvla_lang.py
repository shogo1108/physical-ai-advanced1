"""言葉で動作を切り替えられるかを評価する。
同じシーン（同じシード）に対して指示文だけを変え、
  - 指示どおりのタスクを成功したか
  - 指示と違う動作（取り違え）をしていないか
を数える。方策に渡すのは画像2枚・関節角度・指示文だけ。

使い方:
  python eval_smolvla_lang.py --ckpt outputs/smolvla_feed_put/checkpoints/last/pretrained_model          # 標準スイート
  python eval_smolvla_lang.py --ckpt ... --phrase "please put the cup away" --task put --n 10            # 任意の言い回し
  python eval_smolvla_lang.py --ckpt ... --video_seed 50003                                             # 同じシーンで2つの指示の動画
"""
import argparse, json, os, time
import numpy as np
import torch
from feed_env import FeedEnv, TRAIN_PHRASES, HELDOUT_PHRASES, N_PELLETS
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


def to_img(x):
    return torch.from_numpy(x).permute(2, 0, 1).float().div(255.0)[None]


class Runner:
    def __init__(self, ckpt, device, max_steps):
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(ckpt).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": device}})
        self.env = FeedEnv(render=True, max_steps=max_steps)
        self.hi = None

    def episode(self, seed, phrase, record=False):
        env = self.env
        obs = env.reset(seed)
        self.policy.reset()
        frames = []
        if record and self.hi is None:
            import mujoco
            self.hi = mujoco.Renderer(env.model, 240, 320)  # 1つだけ作って使い回す
        for t in range(env.max_steps):
            batch = {
                "observation.state": torch.from_numpy(obs["state"]).float()[None],
                "observation.images.top": to_img(obs["top"]),
                "observation.images.wrist": to_img(obs["wrist_cam"]),
                "task": [phrase],
            }
            with torch.no_grad():
                action = self.post(self.policy.select_action(self.pre(batch)))
            obs = env.step(action[0].cpu().numpy())
            if record and t % 2 == 0:
                import mujoco
                cam = mujoco.MjvCamera(); cam.lookat[:] = [0.22, 0, 0.05]
                cam.distance, cam.azimuth, cam.elevation = 0.6, 200, -25
                self.hi.update_scene(env.data, cam); frames.append(self.hi.render())
        cup = env.cup_pos()
        result = {
            "seed": seed, "phrase": phrase,
            "feed_ok": bool(env.success_feed()), "put_ok": bool(env.success_put()),
            "pellets_in_dish": int(env.pellets_in_dish()), "pellets_in_cup": int(env.pellets_in_cup()),
            # どちらの動作を「選んだ」か（成否とは別）
            "did_pour": bool(env.pellets_in_dish() >= 2),
            # マット付近で、手から離れて台の上にある（持ったまま近くを通っただけは数えない）
            "did_place": bool(np.linalg.norm(cup[:2] - env.mat_xy) < 0.07 and cup[2] < 0.03),
        }
        return result, frames


def run_condition(runner, task, phrases, seeds, label, log):
    ok = wrong = 0
    rows = []
    for i, seed in enumerate(seeds):
        phrase = phrases[i % len(phrases)]
        t0 = time.time()
        r, _ = runner.episode(seed, phrase)
        success = r["feed_ok"] if task == "feed" else r["put_ok"]
        mistaken = r["did_place"] if task == "feed" else r["did_pour"]
        ok += success; wrong += mistaken
        r.update(task=task, condition=label, success=bool(success), mistaken=bool(mistaken))
        rows.append(r)
        print(f"[{label}] seed {seed} '{phrase}': success={success} mistaken={mistaken} "
              f"dish={r['pellets_in_dish']}/{N_PELLETS} cup={r['pellets_in_cup']} ({time.time()-t0:.0f}s)", flush=True)
    n = len(seeds)
    summary = f"{label:28s} success {ok}/{n} ({ok/n:.0%})   取り違え {wrong}/{n}"
    print("==> " + summary, flush=True)
    log.append(summary)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=50000)
    ap.add_argument("--n", type=int, default=20, help="1条件あたりのエピソード数")
    ap.add_argument("--phrase", default=None, help="任意の言い回しを1つだけ試す")
    ap.add_argument("--task", default=None, choices=["feed", "put"], help="--phrase の正解タスク")
    ap.add_argument("--video_seed", type=int, default=None, help="同じシーンで feed/put 両方の動画を保存")
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runner = Runner(args.ckpt, args.device, args.max_steps)
    seeds = list(range(args.seeds, args.seeds + args.n))

    if args.video_seed is not None:
        import imageio
        for task in ("feed", "put"):
            phrase = TRAIN_PHRASES[task][0]
            r, frames = runner.episode(args.video_seed, phrase, record=True)
            path = os.path.join(args.out, f"seed{args.video_seed}_{task}.mp4")
            imageio.mimsave(path, frames, fps=5)
            print(f"'{phrase}' -> {r}  saved {path}")
        return

    log, rows = [], []
    if args.phrase:
        assert args.task, "--phrase には --task（正解のタスク）も指定してください"
        rows += run_condition(runner, args.task, [args.phrase], seeds, f"{args.task}: '{args.phrase}'", log)
    else:
        # 同じシード列に、学習時の言い回し／未学習の言い回しを順に与える
        for task in ("feed", "put"):
            rows += run_condition(runner, task, TRAIN_PHRASES[task], seeds, f"{task} / 学習時の言い回し", log)
            rows += run_condition(runner, task, HELDOUT_PHRASES[task], seeds, f"{task} / 未学習の言い回し", log)
    print("\n===== まとめ =====")
    for line in log:
        print(line)
    with open(os.path.join(args.out, f"results_{int(time.time())}.json"), "w") as f:
        json.dump({"summary": log, "episodes": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
