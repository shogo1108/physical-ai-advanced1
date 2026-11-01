"""先生役でデモを生成し、成功したエピソードだけ保存する。
使い方: python gen_demos.py <開始シード> <本数> <出力フォルダ>"""
import os, sys, time
import numpy as np
from feed_env import FeedEnv
from teacher import Teacher

start, count, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
env = FeedEnv(render=True)
kept = 0
for seed in range(start, start + count):
    t0 = time.time()
    obs = env.reset(seed)
    teacher = Teacher(env, np.random.default_rng(seed + 10_000), noise=0.005)
    top, wrist, state, action = [], [], [], []
    for _ in range(len(teacher)):
        a = teacher.act()
        top.append(obs["top"]); wrist.append(obs["wrist_cam"]); state.append(obs["state"]); action.append(a)
        obs = env.step(a)
    ok = env.success()
    if ok:
        kept += 1
        np.savez_compressed(os.path.join(out, f"ep_{seed:05d}.npz"), top=np.array(top), wrist=np.array(wrist),
                            state=np.array(state, np.float32), action=np.array(action, np.float32))
    print(f"seed {seed} success={ok} pellets={env.pellets_in_dish()} kept={kept} ({time.time()-t0:.0f}s)", flush=True)
