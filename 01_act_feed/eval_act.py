"""学習したACT方策をシミュレーション内でロールアウトし、成功率を測る。
方策が見るのは「カメラ画像2枚＋関節角度」だけ（カップや皿の座標は渡さない）。

使い方: python eval_act.py --ckpt ckpt --seeds 50000 --n 30 [--video out.mp4]
"""
import argparse, json, os, time
import numpy as np
import torch
from feed_env import FeedEnv, N_PELLETS
from train_act import make_config, img_norm
from lerobot.policies.act.modeling_act import ACTPolicy


def load_policy(ckpt, h, w, chunk, n_action_steps, ensemble):
    cfg = make_config(h, w, chunk=chunk, n_action_steps=n_action_steps)
    if ensemble:
        cfg.temporal_ensemble_coeff = 0.01
        cfg.n_action_steps = 1
    policy = ACTPolicy(cfg)
    policy.load_state_dict(torch.load(f"{ckpt}/last.pt", weights_only=False)["model"])
    policy.eval()
    stats = {k: np.array(v) for k, v in json.load(open(f"{ckpt}/stats.json")).items()}
    return policy, stats


def run_episode(env, policy, stats, seed, frames=None, hi_renderer=None):
    import mujoco
    obs = env.reset(seed)
    policy.reset()
    for t in range(env.max_steps):
        batch = {
            "observation.state": torch.tensor((obs["state"] - stats["state_mean"]) / stats["state_std"],
                                              dtype=torch.float32)[None],
            "observation.images.top": img_norm(obs["top"][None]),
            "observation.images.wrist": img_norm(obs["wrist_cam"][None]),
        }
        with torch.no_grad():
            a = policy.select_action(batch)[0].numpy()
        a = a * stats["action_std"] + stats["action_mean"]
        obs = env.step(a)
        if frames is not None and t % 2 == 0:
            cam = mujoco.MjvCamera(); cam.lookat[:] = [0.2, 0, 0.05]; cam.distance = 0.6
            cam.azimuth = 200; cam.elevation = -25
            hi_renderer.update_scene(env.data, cam)
            big = hi_renderer.render()
            small = np.concatenate([np.kron(obs["top"], np.ones((1, 1, 1), np.uint8)),
                                    obs["wrist_cam"]], 0)  # 方策が見ている画像
            pad = np.zeros((big.shape[0], small.shape[1], 3), np.uint8); pad[:small.shape[0]] = small
            frames.append(np.concatenate([big, pad], 1))
    return env.success(), env.pellets_in_dish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt"); ap.add_argument("--seeds", type=int, default=50000)
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--n_action_steps", type=int, default=10); ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--video", default=None); ap.add_argument("--max_steps", type=int, default=140)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    env = FeedEnv(render=True, max_steps=args.max_steps)
    from feed_env import IMG_H, IMG_W
    policy, stats = load_policy(args.ckpt, IMG_H, IMG_W, args.chunk, args.n_action_steps, args.ensemble)
    ok = 0; results = []
    import imageio, mujoco
    # 高解像度の録画用レンダラーは1つだけ作って使い回す
    # （OSMesaでは途中で破棄すると観測用レンダラーの画像が壊れるため）
    hi = mujoco.Renderer(env.model, 240, 320) if args.video else None
    for i, seed in enumerate(range(args.seeds, args.seeds + args.n)):
        t0 = time.time()
        frames = [] if args.video and i < 4 else None
        s, k = run_episode(env, policy, stats, seed, frames, hi)
        ok += s; results.append({"seed": seed, "success": bool(s), "pellets": int(k)})
        print(f"seed {seed}: success={s} pellets={k}/{N_PELLETS}  ({time.time()-t0:.0f}s)  running {ok}/{i+1}", flush=True)
        if frames:
            base, ext = os.path.splitext(args.video)
            imageio.mimsave(f"{base}_{seed}{ext}", frames, fps=5)
    print(f"SUCCESS RATE: {ok}/{args.n} = {ok/args.n:.0%}")
    json.dump(results, open(os.path.join(args.ckpt, f"eval_{args.seeds}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
