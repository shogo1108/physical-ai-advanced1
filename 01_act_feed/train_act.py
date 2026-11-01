"""LeRobotのACTPolicyを、CPUでも回る軽量設定で学習する。
（GPUがあれば lerobot-train でも同じデータセットを学習できる。README参照）

使い方: python train_act.py --raw raw --out ckpt --steps 3000
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

IMG_KEYS = {"top": "observation.images.top", "wrist": "observation.images.wrist"}


def make_config(h, w, chunk=20, n_action_steps=10):
    return ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=chunk, n_action_steps=n_action_steps,
        pretrained_backbone_weights=None,      # ImageNet重みはこの環境から取得できないため最初から学習
        dim_model=256, dim_feedforward=1024, n_heads=8,
        n_encoder_layers=2, n_decoder_layers=1, n_vae_encoder_layers=2,
        optimizer_lr=1e-4, optimizer_lr_backbone=1e-4, device="cpu",
    )


def img_norm(x_uint8):
    """(B,H,W,3) uint8 -> (B,3,H,W) float, おおよそ平均0・分散1"""
    x = torch.as_tensor(x_uint8).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.45) / 0.25


def random_shift(x, pad=4):
    """画像を数ピクセルずらすデータ拡張（DrQ風）"""
    b, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    ix = torch.randint(0, 2 * pad + 1, (b,)); iy = torch.randint(0, 2 * pad + 1, (b,))
    return torch.stack([x[i, :, iy[i]:iy[i] + h, ix[i]:ix[i] + w] for i in range(b)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw"); ap.add_argument("--out", default="ckpt")
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=20); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_minutes", type=float, default=1e9)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(f"{args.raw}/ep_*.npz"))
    eps = [dict(np.load(f)) for f in files]
    S = np.concatenate([e["state"] for e in eps]); A = np.concatenate([e["action"] for e in eps])
    stats = {"state_mean": S.mean(0).tolist(), "state_std": (S.std(0) + 1e-3).tolist(),
             "action_mean": A.mean(0).tolist(), "action_std": (A.std(0) + 1e-3).tolist()}
    json.dump(stats, open(f"{args.out}/stats.json", "w"))
    sm, ss = np.array(stats["state_mean"]), np.array(stats["state_std"])
    am, as_ = np.array(stats["action_mean"]), np.array(stats["action_std"])
    index = [(ei, t) for ei, e in enumerate(eps) for t in range(len(e["action"]))]
    h, w = eps[0]["top"].shape[1:3]
    print(f"episodes={len(eps)} frames={len(index)} image={h}x{w}", flush=True)

    cfg = make_config(h, w, chunk=args.chunk)
    policy = ACTPolicy(cfg)
    start = 0
    if args.resume and os.path.exists(f"{args.out}/last.pt"):
        ck = torch.load(f"{args.out}/last.pt", weights_only=False)
        policy.load_state_dict(ck["model"]); start = ck["step"]
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
    if args.resume and os.path.exists(f"{args.out}/last.pt"):
        opt.load_state_dict(ck["opt"])
    print("params:", sum(p.numel() for p in policy.parameters()) // 1000, "k", flush=True)

    rng = np.random.default_rng(start)
    t0 = time.time(); hist = []
    for step in range(start, args.steps):
        pick = [index[i] for i in rng.integers(0, len(index), args.batch)]
        top, wr, st, act, pad = [], [], [], [], []
        for ei, t in pick:
            e = eps[ei]; T = len(e["action"])
            idx = np.arange(t, t + args.chunk)
            pad.append(idx >= T); idx = np.minimum(idx, T - 1)
            top.append(e["top"][t]); wr.append(e["wrist"][t]); st.append(e["state"][t]); act.append(e["action"][idx])
        batch = {
            "observation.state": torch.tensor((np.array(st) - sm) / ss, dtype=torch.float32),
            "observation.images.top": random_shift(img_norm(np.array(top))),
            "observation.images.wrist": random_shift(img_norm(np.array(wr))),
            "action": torch.tensor((np.array(act) - am) / as_, dtype=torch.float32),
            "action_is_pad": torch.tensor(np.array(pad)),
        }
        loss, info = policy.forward(batch)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()
        hist.append((info.get("l1_loss", 0), info.get("kld_loss", 0)))
        if (step + 1) % 50 == 0:
            l1, kl = np.mean(hist, 0); hist = []
            el = time.time() - t0
            print(f"step {step+1} l1 {l1:.4f} kld {kl:.4f}  {el/60:.1f}min", flush=True)
        done = (step + 1) == args.steps or (time.time() - t0) / 60 > args.max_minutes
        if (step + 1) % 250 == 0 or done:
            torch.save({"model": policy.state_dict(), "opt": opt.state_dict(), "step": step + 1}, f"{args.out}/last.pt")
            policy.save_pretrained(f"{args.out}/pretrained_model")
        if done:
            break


if __name__ == "__main__":
    main()
