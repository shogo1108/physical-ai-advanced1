"""raw/*.npz を LeRobot 形式に変換する（エピソードごとのタスク文つき）。
使い方: python to_lerobot.py raw data/so101_feed_put"""
import glob, sys, shutil, collections
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from feed_env import FPS, IMG_H, IMG_W, JOINTS

src, root = sys.argv[1], sys.argv[2]
shutil.rmtree(root, ignore_errors=True)
names = [f"{j}.pos" for j in JOINTS]
features = {
    "observation.images.top": {"dtype": "image", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "image", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
    "action": {"dtype": "float32", "shape": (6,), "names": names},
}
ds = LeRobotDataset.create(repo_id="local/so101_feed_put", fps=FPS, features=features, root=root,
                           robot_type="so101_follower", use_videos=False, image_writer_threads=4)
count = collections.Counter()
for f in sorted(glob.glob(f"{src}/ep_*.npz")):
    d = np.load(f)
    phrase = str(d["phrase"])
    count[(str(d["task"]), phrase)] += 1
    for i in range(len(d["action"])):
        ds.add_frame({
            "observation.images.top": d["top"][i],
            "observation.images.wrist": d["wrist"][i],
            "observation.state": d["state"][i],
            "action": d["action"][i],
            "task": phrase,  # 言い回しそのものをタスク文として記録
        })
    ds.save_episode()
if hasattr(ds, "finalize"):
    ds.finalize()
print("episodes:", sum(count.values()), "frames:", ds.num_frames)
for (task, phrase), n in sorted(count.items()):
    print(f"  {task:4s} {n:3d}  {phrase}")
