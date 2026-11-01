"""スクリプトの先生役：特権情報（カップ・皿・マットの正確な位置）を使う。
  task="feed": つかむ → 持ち上げる → 皿の上へ → 手首を回して注ぐ → 戻す
  task="put" : つかむ → 持ち上げる → マットの上へ → 下ろして離す → 手を引く
「持ち上げる」までは両タスクで完全に同じ動き。"""
import numpy as np
import ik
from feed_env import FPS

PITCH = np.radians(30)       # 斜め上から横づかみ
GRIP_OPEN, GRIP_CLOSE = 1.2, -0.17
ROLL_POUR = 2.4              # 注ぐときの手首ロール量 [rad]
GRASP_Z = 0.035              # カップをつかむ高さ
CARRY_Z = 0.16               # 運ぶ高さ
POUR_Z = 0.15               # 注ぐ高さ
PLACE_Z = GRASP_Z            # マットに置くときの高さ
PLACE_FLEX = -0.2           # 置く前に手首を起こしてカップを垂直に戻す量 [rad]
OFF_A, OFF_T = 0.025, -0.02   # カップ中心に対するジョー先端の位置補正（接近方向／開閉方向）


def frame(xy, pitch=PITCH):
    a = np.r_[xy - ik.PAN_XY, 0.0]; a /= np.linalg.norm(a)
    t = np.array([-a[1], a[0], 0.0])
    ap = np.cos(pitch) * a + np.sin(pitch) * np.array([0, 0, -1.0])
    y = np.cross(t, ap)
    return np.column_stack([ap, y, t]), ap, t


def roll_R(R, ang):
    """接近軸まわりにang回転"""
    a = R[:, 0]
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    Rot = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    return Rot @ R


def site_target(center_xy, z, roll=0.0, back=0.0):
    R, a, t = frame(center_xy)
    pos = np.r_[center_xy, z] + (OFF_A - back) * a + OFF_T * t
    return pos, roll_R(R, roll)


def _polar(xy):
    v = xy - ik.PAN_XY
    return np.linalg.norm(v), np.arctan2(v[1], v[0])


class Teacher:
    """1制御ステップごとに目標関節角を返す。noiseで軌道を少し揺らす。"""

    def __init__(self, env, rng, noise=0.0, task="feed"):
        self.env, self.rng, self.noise, self.task = env, rng, noise, task
        c = env.cup_pos()[:2]
        dsh = env.dish_xy
        # 皿の中心より少しずらす（傾けたカップの口が皿の中心に来るように）
        _, a_d, t_d = frame(dsh)
        pour_xy = dsh + 0.055 * t_d[:2] + 0.013 * a_d[:2] + rng.normal(0, 0.006, 2) * (noise > 0)
        jit = lambda s: rng.normal(0, s, 3) * (noise > 0)
        # (秒, カップ中心xy, 高さ, ロール, グリッパー)
        common = [  # つかんで持ち上げるまで（両タスク共通）
            (1.5, c, GRASP_Z + 0.03, 0.0, GRIP_OPEN, 0.07),
            (1.2, c, GRASP_Z, 0.0, GRIP_OPEN, 0.0),
            (0.8, c, GRASP_Z, 0.0, GRIP_CLOSE, 0.0),
            (1.2, c, CARRY_Z, 0.0, GRIP_CLOSE, 0.0),
        ]
        if task == "put":
            m = env.mat_xy + rng.normal(0, 0.004, 2) * (noise > 0)
            self.keys = common + [
                (2.0, m, CARRY_Z, 0.0, GRIP_CLOSE, 0.0),
                (1.2, m, PLACE_Z, 0.0, GRIP_CLOSE, 0.0),
                (0.6, m, PLACE_Z, 0.0, GRIP_OPEN, 0.0),         # 離す
                (1.0, m, PLACE_Z + 0.06, 0.0, GRIP_OPEN, 0.07),  # 手前上に引く
                (1.0, m, PLACE_Z + 0.06, 0.0, GRIP_OPEN, 0.07),  # 静止
            ]
            self.jit = [jit(0.004) for _ in self.keys]
            self.plan()
            # 横づかみでカップが手の中で傾くので、運ぶ途中から手首を起こして垂直に戻す
            n_common = int(sum(int(k[0] * FPS) for k in common))
            n_move = int(2.0 * FPS)
            ramp = np.clip((np.arange(len(self.traj)) - n_common) / n_move, 0, 1)
            self.traj[:, 3] += PLACE_FLEX * ramp
            return
        self.keys = common + [
            (2.0, pour_xy, CARRY_Z, 0.0, GRIP_CLOSE, 0.0),
            (0.8, pour_xy, POUR_Z, 0.0, GRIP_CLOSE, 0.0),
            (1.6, pour_xy, POUR_Z, -ROLL_POUR, GRIP_CLOSE, 0.0),
            (0.3, pour_xy, POUR_Z, -ROLL_POUR + 0.4, GRIP_CLOSE, 0.0),  # 小刻みに振って出し切る
            (0.3, pour_xy, POUR_Z, -ROLL_POUR, GRIP_CLOSE, 0.0),
            (0.3, pour_xy, POUR_Z, -ROLL_POUR + 0.4, GRIP_CLOSE, 0.0),
            (0.3, pour_xy, POUR_Z, -ROLL_POUR, GRIP_CLOSE, 0.0),
            (0.6, pour_xy, POUR_Z, -ROLL_POUR, GRIP_CLOSE, 0.0),
            (1.2, pour_xy, POUR_Z, 0.0, GRIP_CLOSE, 0.0),
            (0.8, pour_xy, POUR_Z, 0.0, GRIP_CLOSE, 0.0),  # 静止
        ]
        self.jit = [jit(0.004) for _ in self.keys]
        self.plan()

    def plan(self):
        env = self.env
        fps = FPS
        q = env.qpos()[:5]
        start = (env.qpos(), None)
        traj = []
        # 最初の位置（現在の姿勢）
        prev = None
        for k, (dur, xy, z, roll, g, back) in enumerate(self.keys):
            n = max(1, int(dur * fps))
            for i in range(1, n + 1):
                s = i / n
                s = 3 * s**2 - 2 * s**3  # なめらかな補間
                if prev is None:
                    # ホームから最初のキーへは関節空間で補間
                    pos, R = site_target(xy, z, roll, back)
                    if not hasattr(self, "_q1"):
                        self._q1, *_ = ik.solve_ik(env.model, env.data, env.site, pos + self.jit[k], R,
                                                   self._seed_q(xy), env.qadr, env.dadr, iters=400)
                    qa = (1 - s) * env.qpos()[:5] + s * self._q1
                    traj.append(np.r_[qa, (1 - s) * env.qpos()[5] + s * g])
                    continue
                pxy, pz, proll, pg, pb = prev
                # 極座標（パン軸まわり）で補間して根元に近づきすぎないようにする
                pr, pa = _polar(pxy); r_, a_ = _polar(xy)
                cr, ca = (1 - s) * pr + s * r_, (1 - s) * pa + s * a_
                cxy = ik.PAN_XY + cr * np.array([np.cos(ca), np.sin(ca)])
                cz = (1 - s) * pz + s * z
                croll = (1 - s) * proll + s * roll
                cg = (1 - s) * pg + s * g
                cb = (1 - s) * pb + s * back
                # 位置・向きはロールなしでIKを解き、注ぐ回転は手首ロール関節に直接足す
                pos, R = site_target(cxy, cz, 0.0, cb)
                q, ep, er = ik.solve_ik(env.model, env.data, env.site, pos + self.jit[k], R, q,
                                        env.qadr, env.dadr, iters=100)
                self.max_err = max(getattr(self, "max_err", 0), ep)
                qr = q.copy(); qr[4] += croll
                traj.append(np.r_[qr, cg])
            if prev is None:
                q = self._q1.copy()
            prev = (xy, z, roll, g, back)
        self.traj = np.array(traj)
        self.i = 0

    def _seed_q(self, xy):
        # 横づかみの姿勢に近い初期値（ロール≈+1.6）
        ang = np.arctan2(xy[1] - ik.PAN_XY[1], xy[0] - ik.PAN_XY[0])
        return np.array([-ang, -0.4, 1.3, -0.4, 1.62])

    def __len__(self):
        return len(self.traj)

    def act(self):
        a = self.traj[min(self.i, len(self.traj) - 1)].copy()
        self.i += 1
        if self.noise > 0:
            a[:5] += self.rng.normal(0, self.noise, 5)
        return a
