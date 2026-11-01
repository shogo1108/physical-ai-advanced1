"""2タスク（feed / put）のデモを先生役で生成し、成功したものだけ保存する。
各エピソードには学習用の言い回し（TRAIN_PHRASES）からランダムに1つを割り当てる。
シードの偶奇でタスクを交互にする（偶数=feed、奇数=put）。

使い方: python gen_demos_lang.py <開始シード> <本数> <出力フォルダ>
"""
import os, sys, time
import numpy as np
from feed_env import FeedEnv, TRAIN_PHRASES
from teacher import Teacher

start, count, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
env = FeedEnv(render=True)
kept = {"feed": 0, "put": 0}
for seed in range(start, start + count):
    t0 = time.time()
    task = "feed" if seed % 2 == 0 else "put"
    rng = np.random.default_rng(seed + 10_000)
    phrase = TRAIN_PHRASES[task][rng.integers(len(TRAIN_PHRASES[task]))]
    obs = env.reset(seed)
    teacher = Teacher(env, rng, noise=0.005, task=task)
    top, wrist, state, action = [], [], [], []
    for _ in range(len(teacher)):
        a = teacher.act()
        top.append(obs["top"]); wrist.append(obs["wrist_cam"]); state.append(obs["state"]); action.append(a)
        obs = env.step(a)
    ok = env.success(task)
    if ok:
        kept[task] += 1
        np.savez_compressed(os.path.join(out, f"ep_{seed:05d}.npz"), top=np.array(top), wrist=np.array(wrist),
                            state=np.array(state, np.float32), action=np.array(action, np.float32),
                            task=task, phrase=phrase)
    print(f"seed {seed} {task:4s} success={ok} kept={kept} ({time.time()-t0:.0f}s)", flush=True)
