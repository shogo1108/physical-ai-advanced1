"""SO-101 エサやり／片付け環境（MuJoCo）— 言語で動作を切り替える版

シーンは常に同じ（エサ粒入りカップ・青いエサ皿・緑のマット）。
  feed（エサやり）: カップをつかみ、皿の上で傾けて注ぐ
  put （片付け）  : カップをつかみ、緑のマットの上に立てて置く（粒はこぼさない）
どちらを行うかは画像からは決まらず、タスク文（言葉）だけで決まる。
観測: 俯瞰カメラ画像 + 手首カメラ画像 + 6関節角度
行動: 6関節の目標角度（位置制御）
"""
import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
# ロボットモデル（MuJoCo Menagerie の robotstudio_so101）。同じフォルダ → リポジトリ直下 の順に探す
SO101_XML = next((p for p in (os.path.join(d, "mujoco_menagerie", "robotstudio_so101", "so101.xml")
                              for d in (HERE, os.path.dirname(HERE))) if os.path.exists(p)),
                 os.path.join(HERE, "mujoco_menagerie", "robotstudio_so101", "so101.xml"))

FPS = 10
PHYS_DT = 0.005
SUBSTEPS = int(round(1.0 / FPS / PHYS_DT))  # 10
IMG_H, IMG_W = 96, 128
N_PELLETS = 12
PELLET_R = 0.0055
CUP_R, CUP_H, CUP_WALL = 0.021, 0.08, 0.003
DISH_R = 0.075
MAT_HALF = 0.035  # 緑のマット（正方形）の半幅

# 学習に使う言い回し（エピソードごとにランダムに1つ割り当てる）
TRAIN_PHRASES = {
    "feed": ["feed the pet", "give the pet its food", "pour the food into the bowl", "fill the pet's bowl"],
    "put": ["put the cup away", "put the cup on the green mat", "tidy up the cup", "place the cup on the mat"],
}
# 評価専用の言い回し（学習には一切使わない）
HELDOUT_PHRASES = {
    "feed": ["give the dog its food", "it's dinner time for the pet"],
    "put": ["clean up the cup", "move the cup to the mat"],
}
TASK_TEXT = TRAIN_PHRASES["feed"][0]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOME = np.array([0.0, -1.57, 1.57, 1.0, 0.0, 0.0])


def _ring(name, r, h, z0, n, thick, rgba, extra=""):
    """円筒の壁を箱の輪で近似する（凹形状の当たり判定対策）"""
    out = []
    seg = 2 * np.pi * r / n * 0.55
    for i in range(n):
        a = 2 * np.pi * i / n
        x, y = r * np.cos(a), r * np.sin(a)
        out.append(
            f'<geom name="{name}{i}" type="box" size="{thick/2:.4f} {seg:.4f} {h/2:.4f}" '
            f'pos="{x:.4f} {y:.4f} {z0 + h/2:.4f}" euler="0 0 {a:.4f}" rgba="{rgba}" {extra}/>'
        )
    return "\n".join(out)


def build_xml():
    cup_walls = _ring("cupwall", CUP_R - CUP_WALL / 2, CUP_H, 0.0, 16, CUP_WALL,
                      "0.85 0.35 0.25 1", 'mass="0.002" condim="4" friction="1.2 0.01 0.001"')
    dish_rim = _ring("dishrim", DISH_R, 0.022, 0.004, 28, 0.004, "0.3 0.5 0.85 1", "")
    pellets = "\n".join(
        f'<body name="pellet{i}" pos="0 0 -1"><freejoint/>'
        f'<geom name="pellet{i}" type="sphere" size="{PELLET_R}" mass="0.001" '
        f'rgba="0.55 0.33 0.12 1" condim="6" friction="0.7 0.01 0.002" solref="0.01 1"/></body>'
        for i in range(N_PELLETS)
    )
    return f"""
<mujoco model="so101_feed">
  <include file="so101.xml"/>
  <option timestep="{PHYS_DT}"/>
  <visual>
    <headlight diffuse="0.35 0.35 0.35" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <global offwidth="640" offheight="480"/>
  </visual>
  <asset>
    <texture type="2d" name="wood" builtin="checker" rgb1="0.55 0.45 0.35" rgb2="0.5 0.4 0.3" width="100" height="100"/>
    <material name="table" texture="wood" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0.2 0 1.5" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="table" friction="1 0.01 0.001"/>
    <camera name="top" pos="0.17 0 0.62" xyaxes="0 -1 0 1 0 0" fovy="55"/>
    <body name="cup" pos="0.2 -0.15 0">
      <freejoint name="cup"/>
      <geom name="cupbottom" type="cylinder" size="{CUP_R} 0.002" pos="0 0 0.002" mass="0.01" rgba="0.85 0.35 0.25 1" condim="4"/>
      {cup_walls}
    </body>
    <body name="mat" pos="0.34 0 0" mocap="true">
      <geom name="mat" type="box" size="{MAT_HALF} {MAT_HALF} 0.0015" pos="0 0 0.0015" rgba="0.25 0.65 0.3 1" friction="1 0.01 0.001"/>
    </body>
    <body name="dish" pos="0.2 0.15 0" mocap="true">
      <geom name="dishbase" type="cylinder" size="{DISH_R} 0.002" pos="0 0 0.002" rgba="0.3 0.5 0.85 1"/>
      {dish_rim}
    </body>
    {pellets}
  </worldbody>
</mujoco>"""


class FeedEnv:
    def __init__(self, render=True, max_steps=260):
        # 並列実行で同じファイルを取り合わないよう、プロセスごとに別名で書いて読み込む
        path = os.path.join(os.path.dirname(SO101_XML), f"_scene_{os.getpid()}.xml")
        with open(path, "w") as f:
            f.write(build_xml())
        try:
            self.model = mujoco.MjModel.from_xml_path(path)
        finally:
            os.remove(path)
        self.data = mujoco.MjData(self.model)
        self.render_on = render
        self.model.vis.quality.offsamples = 0
        self.renderer = mujoco.Renderer(self.model, IMG_H, IMG_W) if render else None
        if render:  # ソフトウェア描画を軽くする（影・反射なし）
            fl = self.renderer._scene.flags
            fl[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            fl[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            fl[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.max_steps = max_steps
        m = self.model
        self.qadr = np.array([m.joint(j).qposadr[0] for j in JOINTS])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in JOINTS])
        self.cup_qadr = m.joint("cup").qposadr[0]
        self.pellet_qadr = [m.joint(m.body(f"pellet{i}").jntadr[0]).qposadr[0] for i in range(N_PELLETS)]
        self.dish_mocap = m.body("dish").mocapid[0]
        self.mat_mocap = m.body("mat").mocapid[0]
        self.cup_body = m.body("cup").id
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.cup_geoms = [i for i in range(m.ngeom) if m.geom(i).name.startswith(("cupwall", "cupbottom"))]
        self.base_cup_rgba = m.geom_rgba[self.cup_geoms[0]].copy()
        self.lo = m.actuator_ctrlrange[:, 0].copy()
        self.hi = m.actuator_ctrlrange[:, 1].copy()

    # ---------- リセット（ランダム化） ----------
    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.rng = rng
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        # カップと皿の位置（ロボットの右前と左前）
        r_c, a_c = rng.uniform(0.22, 0.26), rng.uniform(-0.6, -0.3)
        r_d, a_d = rng.uniform(0.21, 0.26), rng.uniform(0.3, 0.6)
        cup_xy = np.array([0.0388 + r_c * np.cos(a_c), r_c * np.sin(a_c)])
        dish_xy = np.array([0.0388 + r_d * np.cos(a_d), r_d * np.sin(a_d)])
        if rng.random() < 0.5:  # 左右入れ替え
            cup_xy[1], dish_xy[1] = -cup_xy[1], -dish_xy[1]
        self.cup_xy0, self.dish_xy = cup_xy, dish_xy
        d.qpos[self.cup_qadr:self.cup_qadr + 3] = [cup_xy[0], cup_xy[1], 0.0]
        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[self.cup_qadr + 3:self.cup_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.mocap_pos[self.dish_mocap] = [dish_xy[0], dish_xy[1], 0.0]
        # 緑のマット：ロボット正面の奥
        r_m, a_m = rng.uniform(0.29, 0.31), rng.uniform(-0.1, 0.1)
        self.mat_xy = np.array([0.0388 + r_m * np.cos(a_m), r_m * np.sin(a_m)])
        d.mocap_pos[self.mat_mocap] = [self.mat_xy[0], self.mat_xy[1], 0.0]
        # エサ粒をカップの中に積む
        inner = CUP_R - CUP_WALL - PELLET_R - 0.001
        for i, adr in enumerate(self.pellet_qadr):
            layer, k = divmod(i, 4)
            a = k * np.pi / 2 + layer * 0.6
            rr = inner * 0.65
            d.qpos[adr:adr + 3] = [cup_xy[0] + rr * np.cos(a), cup_xy[1] + rr * np.sin(a),
                                   0.01 + PELLET_R + layer * 2.1 * PELLET_R]
            d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        # 見た目のランダム化（色・照明）
        tint = self.base_cup_rgba[:3] * rng.uniform(0.8, 1.15, 3)
        for g in self.cup_geoms:
            m.geom_rgba[g, :3] = np.clip(tint, 0, 1)
        m.light_diffuse[0] = rng.uniform(0.35, 0.6, 1).repeat(3)
        # 腕を初期姿勢へ（少しノイズ）
        q0 = HOME + rng.normal(0, 0.03, 6) * np.array([1, 1, 1, 1, 1, 0])
        d.qpos[self.qadr] = q0
        d.ctrl[:] = q0
        mujoco.mj_forward(m, d)
        for _ in range(100):  # 粒を落ち着かせる
            mujoco.mj_step(m, d)
        self.t = 0
        return self.obs()

    # ---------- 観測 ----------
    def qpos(self):
        return self.data.qpos[self.qadr].copy()

    def images(self):
        out = {}
        for cam in ("top", "wrist_cam"):
            self.renderer.update_scene(self.data, camera=cam)
            out[cam] = self.renderer.render().copy()
        return out

    def obs(self):
        o = {"state": self.qpos()}
        if self.render_on:
            o.update(self.images())
        return o

    # ---------- 1ステップ ----------
    def step(self, action):
        # 目標角を前回値から直線補間しながら送る（10Hz指令でも動きをなめらかに）
        target = np.clip(action, self.lo, self.hi)
        start = self.data.ctrl.copy()
        for k in range(SUBSTEPS):
            self.data.ctrl[:] = start + (target - start) * (k + 1) / SUBSTEPS
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        return self.obs()

    # ---------- 成功判定（特権情報。学習する方策には渡さない） ----------
    def pellets_in_dish(self):
        n = 0
        for adr in self.pellet_qadr:
            p = self.data.qpos[adr:adr + 3]
            if np.linalg.norm(p[:2] - self.dish_xy) < DISH_R and p[2] < 0.03:
                n += 1
        return n

    def pellets_in_cup(self):
        c = self.data.xpos[self.cup_body]
        n = 0
        for adr in self.pellet_qadr:
            p = self.data.qpos[adr:adr + 3]
            if np.linalg.norm(p[:2] - c[:2]) < CUP_R and p[2] > c[2]:
                n += 1
        return n

    def cup_upright(self, max_tilt_deg=25):
        R = self.data.xmat[self.cup_body].reshape(3, 3)
        return R[2, 2] > np.cos(np.radians(max_tilt_deg))

    def cup_on_mat(self):
        c = self.cup_pos()
        return bool(np.all(np.abs(c[:2] - self.mat_xy) < MAT_HALF + 0.005) and c[2] < 0.02)

    def success_feed(self, ratio=0.5):
        return self.pellets_in_dish() >= ratio * N_PELLETS

    def success_put(self, ratio=0.5):
        return bool(self.cup_on_mat() and self.cup_upright() and self.pellets_in_cup() >= ratio * N_PELLETS)

    def success(self, task="feed"):
        return self.success_feed() if task == "feed" else self.success_put()

    def cup_pos(self):
        return self.data.qpos[self.cup_qadr:self.cup_qadr + 3].copy()
