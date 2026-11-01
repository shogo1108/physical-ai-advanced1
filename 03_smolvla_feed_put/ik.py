"""SO-101の数値IK（減衰最小二乗法）。デモ生成の先生役が使う。"""
import numpy as np
import mujoco

PAN_XY = np.array([0.0388, 0.0])


def side_grasp_R(target_xy, roll_sign=1.0):
    """カップを横からつかむ向き: 接近方向=水平・放射方向、ジョーの開閉方向=水平・接線方向"""
    a = np.r_[target_xy - PAN_XY, 0.0]
    a /= np.linalg.norm(a)
    t = roll_sign * np.array([-a[1], a[0], 0.0])
    y = np.cross(t, a)
    return np.column_stack([a, y, t])


def solve_ik(model, data_template, site, pos, R, q_init, qadr, dadr, iters=200, w_rot=0.3):
    d = mujoco.MjData(model)
    d.qpos[:] = data_template.qpos
    q = q_init.copy()
    jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
    lo = model.jnt_range[[model.dof_jntid[i] for i in dadr[:5]], 0]
    hi = model.jnt_range[[model.dof_jntid[i] for i in dadr[:5]], 1]
    for _ in range(iters):
        d.qpos[qadr[:5]] = q
        mujoco.mj_kinematics(model, d); mujoco.mj_comPos(model, d)
        cur_p = d.site_xpos[site]; cur_R = d.site_xmat[site].reshape(3, 3)
        e_p = pos - cur_p
        e_R = 0.5 * (np.cross(cur_R[:, 0], R[:, 0]) + np.cross(cur_R[:, 1], R[:, 1]) + np.cross(cur_R[:, 2], R[:, 2]))
        err = np.r_[e_p, w_rot * e_R]
        if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_R) < 1e-3:
            break
        mujoco.mj_jacSite(model, d, jp, jr, site)
        J = np.vstack([jp[:, dadr[:5]], w_rot * jr[:, dadr[:5]]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
    return q, np.linalg.norm(e_p), np.linalg.norm(e_R)
