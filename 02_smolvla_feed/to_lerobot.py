"""raw/*.npz を LeRobot 形式のデータセットに変換する。
使い方: python to_lerobot.py raw data/so101_feed"""
import glob, sys, shutil
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from feed_env import FPS, IMG_H, IMG_W, JOINTS, TASK_TEXT

src, root = sys.argv[1], sys.argv[2]
shutil.rmtree(root, ignore_errors=True)
names = [f"{j}.pos" for j in JOINTS]
features = {
    "observation.images.top": {"dtype": "image", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "image", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
    "action": {"dtype": "float32", "shape": (6,), "names": names},
}
ds = LeRobotDataset.create(repo_id="local/so101_feed", fps=FPS, features=features, root=root,
                           robot_type="so101_follower", use_videos=False, image_writer_threads=4)
files = sorted(glob.glob(f"{src}/ep_*.npz"))
for f in files:
    d = np.load(f)
    for i in range(len(d["action"])):
        ds.add_frame({
            "observation.images.top": d["top"][i],
            "observation.images.wrist": d["wrist"][i],
            "observation.state": d["state"][i],
            "action": d["action"][i],
            "task": TASK_TEXT,  # SmolVLA移行用のタスク文
        })
    ds.save_episode()
ds.finalize() if hasattr(ds, "finalize") else None
print("episodes:", len(files), "frames:", ds.num_frames)
