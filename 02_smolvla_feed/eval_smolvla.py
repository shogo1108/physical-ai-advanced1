"""学習したSmolVLA方策をエサやりシミュレーションでロールアウトし、成功率を測る。
方策に渡すのは「カメラ画像2枚＋関節角度＋タスク文」だけ（カップや皿の座標は渡さない）。

使い方: python eval_smolvla.py --ckpt outputs/smolvla_feed/checkpoints/last/pretrained_model --n 30
        python eval_smolvla.py --ckpt ... --task "feed the pet" --video videos/rollout.mp4
"""
import argparse, os, time
import numpy as np
import torch
from feed_env import FeedEnv, N_PELLETS, TASK_TEXT
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


def to_img(x):
    """(H,W,3) uint8 -> (1,3,H,W) float [0,1]（LeRobotの画像規約）"""
    return torch.from_numpy(x).permute(2, 0, 1).float().div(255.0)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default=TASK_TEXT)
    ap.add_argument("--seeds", type=int, default=50000)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=140)
    ap.add_argument("--video", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    policy = SmolVLAPolicy.from_pretrained(args.ckpt).to(args.device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    env = FeedEnv(render=True, max_steps=args.max_steps)
    hi = None
    if args.video:
        import imageio, mujoco
        hi = mujoco.Renderer(env.model, 240, 320)  # 1つだけ作って使い回す（OSMesa対策）

    ok = 0
    for i, seed in enumerate(range(args.seeds, args.seeds + args.n)):
        t0 = time.time()
        obs = env.reset(seed)
        policy.reset()
        frames = [] if hi is not None and i < 4 else None
        for t in range(env.max_steps):
            batch = {
                "observation.state": torch.from_numpy(obs["state"]).float()[None],
                "observation.images.top": to_img(obs["top"]),
                "observation.images.wrist": to_img(obs["wrist_cam"]),
                "task": [args.task],
            }
            with torch.no_grad():
                action = post(policy.select_action(pre(batch)))
            obs = env.step(action[0].cpu().numpy())
            if frames is not None and t % 2 == 0:
                cam = mujoco.MjvCamera(); cam.lookat[:] = [0.2, 0, 0.05]
                cam.distance, cam.azimuth, cam.elevation = 0.6, 200, -25
                hi.update_scene(env.data, cam); frames.append(hi.render())
        s, k = env.success(), env.pellets_in_dish()
        ok += s
        print(f"seed {seed}: success={s} pellets={k}/{N_PELLETS} ({time.time()-t0:.0f}s) running {ok}/{i+1}", flush=True)
        if frames:
            base, ext = os.path.splitext(args.video)
            os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
            imageio.mimsave(f"{base}_{seed}{ext}", frames, fps=5)
    print(f"SUCCESS RATE ({args.task!r}): {ok}/{args.n} = {ok/args.n:.0%}")


if __name__ == "__main__":
    main()
