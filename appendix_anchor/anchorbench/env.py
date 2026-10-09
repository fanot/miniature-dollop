"""AnchorBench: a minimal, fully controllable language-conditioned tabletop grasping task.

The point of this environment is not realism but *control over the data distribution*.
It reproduces the structure of a narrow real-robot fine-tuning dataset (each object is almost
always demonstrated in "its" place on the table), so that we can measure precisely when a
policy binds the instruction to the object and when it binds it to a memorised location.

World: 2D workspace [0,1]^2 seen top-down by a 64x64 RGB camera.
Objects: 4 colours x 3 shapes = 12 identities. Each object has a "handle" (grey stub) with a
random orientation; a grasp only counts if the gripper closes at the handle tip, so the policy
has to approach from the handle side without touching the body (a local, geometry-dependent
motor skill that cannot be solved by "go to the point").
Action: (dx, dy) in [-1,1]^2 scaled by STEP, plus a binary "close gripper" that ends the episode.
"""
from __future__ import annotations

import numpy as np

IMG = 64
R = 0.06                 # object body radius
HANDLE = 0.075           # handle length outside the body
GRASP_OFF = R + 0.035    # grasp point distance from object centre (along handle)
PRE_OFF = R + 0.14       # pre-grasp point distance from object centre
STEP = 0.04              # max end-effector displacement per step
TMAX = 50                # episode horizon
GRIP_TOL = 0.025         # success tolerance at the grasp point
COLLIDE = 0.85 * R       # entering this radius of any body = collision (episode fails)
MIN_SEP = 0.20           # min distance between object centres

COLORS = np.array([[0.92, 0.22, 0.22],   # red
                   [0.22, 0.78, 0.28],   # green
                   [0.28, 0.48, 0.97],   # blue
                   [0.95, 0.83, 0.16]],  # yellow
                  dtype=np.float32)
COLOR_NAMES = ["red", "green", "blue", "yellow"]
SHAPE_NAMES = ["square", "circle", "triangle"]
IDENTS = [(c, s) for c in range(4) for s in range(3)]           # 12 identities
HELDOUT = {(0, 2), (1, 1), (2, 0)}                               # red triangle, green circle, blue square
TRAIN_IDS = [i for i, cs in enumerate(IDENTS) if cs not in HELDOUT]
HELD_IDS = [i for i, cs in enumerate(IDENTS) if cs in HELDOUT]

BG_DEFAULT = np.array([0.16, 0.16, 0.19], np.float32)
BG_SHIFT = np.array([0.40, 0.35, 0.30], np.float32)            # "lighting / tablecloth" shift
HANDLE_RGB = np.array([0.72, 0.72, 0.72], np.float32)
EE_RGB = np.array([1.0, 1.0, 1.0], np.float32)

# Each identity has its own "usual place" on the table in the robot demonstrations.
_xs = [0.17, 0.39, 0.61, 0.83]
_ys = [0.36, 0.60, 0.84]
_grid = np.array([(x, y) for y in _ys for x in _xs], np.float32)
ANCHORS = _grid[np.random.RandomState(0).permutation(12)]       # ANCHORS[identity] -> (x, y)

OBJ_Y = (0.32, 0.88)
OBJ_X = (0.15, 0.85)


def ident_name(i: int) -> str:
    c, s = IDENTS[i]
    return f"{COLOR_NAMES[c]} {SHAPE_NAMES[s]}"


def instruction(i: int) -> str:
    return f"pick the {ident_name(i)}"


def unit(theta):
    return np.stack([np.cos(theta), np.sin(theta)], -1)


# --------------------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------------------
class Scene:
    """One episode's initial state. Slot 0 holds the instructed object (unless removed)."""

    def __init__(self, pos, ids, theta, ee0, target_id, present=None, bg=None):
        self.pos = np.asarray(pos, np.float32)          # (K,2)
        self.ids = np.asarray(ids, np.int64)            # (K,)
        self.theta = np.asarray(theta, np.float32)      # (K,)
        self.ee0 = np.asarray(ee0, np.float32)          # (2,)
        self.target_id = int(target_id)                 # identity named in the instruction
        self.present = np.ones(len(ids), bool) if present is None else np.asarray(present, bool)
        self.bg = BG_DEFAULT.copy() if bg is None else np.asarray(bg, np.float32)

    @property
    def K(self):
        return len(self.ids)


def grasp_point(c, th):
    return c + GRASP_OFF * unit(th)


def pregrasp_point(c, th):
    return c + PRE_OFF * unit(th)


def expert_action(ee, c, th):
    """State-feedback expert: go to the pre-grasp point, then slide in along the handle, close."""
    u = unit(th)
    g = c + GRASP_OFF * u
    p = c + PRE_OFF * u
    d = ee - c
    along = float(d @ u)
    lat = float(np.linalg.norm(d - along * u))
    in_corr = lat < 0.02 and (R + 0.02) < along < PRE_OFF + 0.02
    tgt = g if in_corr else p
    v = tgt - ee
    dist = float(np.linalg.norm(v))
    a = v / max(dist, 1e-6) * min(1.0, dist / STEP)
    grip = bool(in_corr and np.linalg.norm(ee - g) < 0.012)
    return a.astype(np.float32), grip


def _collides(ee, scene):
    m = scene.present
    if not m.any():
        return False
    return bool((np.linalg.norm(scene.pos[m] - ee, axis=-1) < COLLIDE).any())


def expert_solves(scene: Scene) -> bool:
    ee = scene.ee0.copy()
    c, th = scene.pos[0], scene.theta[0]
    for _ in range(TMAX):
        a, grip = expert_action(ee, c, th)
        if grip:
            return bool(np.linalg.norm(ee - grasp_point(c, th)) < GRIP_TOL)
        ee = np.clip(ee + a * STEP, 0.0, 1.0)
        if _collides(ee, scene):
            return False
    return False


def _sample_positions(rng, ids, layout, sigma):
    K = len(ids)
    for _ in range(200):
        if layout == "anchored":
            pos = ANCHORS[ids] + rng.normal(0, sigma, (K, 2))
            pos[:, 0] = np.clip(pos[:, 0], *OBJ_X)
            pos[:, 1] = np.clip(pos[:, 1], *OBJ_Y)
        else:
            pos = np.stack([rng.uniform(*OBJ_X, K), rng.uniform(*OBJ_Y, K)], -1)
        dmat = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(K) * 9
        if dmat.min() >= MIN_SEP:
            return pos.astype(np.float32)
    return None


def sample_scene(rng: np.random.RandomState, split: str = "id", n_obj: int = 3,
                 anchor_sigma: float = 0.04, p_uniform: float = 0.0, max_tries: int = 500) -> Scene:
    """Splits:
    id        - training distribution: every object near its usual place, train identities as targets
    pos_ood   - same identities, uniform positions (the usual places carry no information)
    combo_ood - held-out colour x shape combinations as targets, uniform positions
    swap      - id layout, but the target and a distractor exchange places (layout vs language conflict)
    empty     - id layout with the instructed object removed (memorisation probe, no success possible)
    bg        - id layout with a different table colour (appearance shift)
    """
    for _ in range(max_tries):
        if split == "combo_ood":
            tgt = rng.choice(HELD_IDS)
        else:
            tgt = rng.choice(TRAIN_IDS)
        others = [i for i in range(12) if i != tgt]
        ids = np.array([tgt] + list(rng.choice(others, n_obj - 1, replace=False)))
        layout = "uniform" if split in ("pos_ood", "combo_ood") else "anchored"
        if split == "id" and p_uniform > 0 and rng.rand() < p_uniform:
            layout = "uniform"
        pos = _sample_positions(rng, ids, layout, anchor_sigma)
        if pos is None:
            continue
        if split == "swap":
            pos[[0, 1]] = pos[[1, 0]]
        theta = rng.uniform(0, 2 * np.pi, n_obj).astype(np.float32)
        ee0 = np.array([rng.uniform(0.25, 0.75), rng.uniform(0.04, 0.12)], np.float32)
        bg = BG_SHIFT if split == "bg" else BG_DEFAULT
        sc = Scene(pos, ids, theta, ee0, tgt, bg=bg)
        if not expert_solves(sc):
            continue
        if split == "empty":
            sc.present[0] = False
        return sc
    raise RuntimeError("could not sample a feasible scene")


def sample_pointing_scene(rng, n_obj=None):
    """'Web-style' pointing sample: any identity, anywhere, no actions. Mimics the VLM pointing
    data (RefSpatial / PixMo-Points / RoboPoint) that Green-VLA mixes in at stage L1."""
    n_obj = n_obj or rng.randint(2, 5)
    ids = rng.choice(12, n_obj, replace=False)
    pos = None
    while pos is None:
        pos = _sample_positions(rng, ids, "uniform", 0.0)
    theta = rng.uniform(0, 2 * np.pi, n_obj).astype(np.float32)
    ee = np.array([rng.uniform(0.05, 0.95), rng.uniform(0.03, 0.95)], np.float32)
    k = rng.randint(n_obj)
    return Scene(pos, ids, theta, ee, ids[k]), k


# --------------------------------------------------------------------------------------
# Rendering (vectorised over a batch of scenes with the same number of slots)
# --------------------------------------------------------------------------------------
_pix = (np.arange(IMG, dtype=np.float32) + 0.5) / IMG
PX, PY = np.meshgrid(_pix, 1.0 - _pix)        # PX[row, col] = x ; PY[row, col] = y


def render_batch(pos, ids, theta, present, ee, bg):
    """pos (N,K,2) ids (N,K) theta (N,K) present (N,K) ee (N,2) bg (N,3) -> uint8 (N,64,64,3)."""
    N, K = ids.shape
    img = np.broadcast_to(bg[:, None, None, :], (N, IMG, IMG, 3)).copy()
    X, Y = PX[None], PY[None]
    for k in range(K):
        cx = pos[:, k, 0][:, None, None]
        cy = pos[:, k, 1][:, None, None]
        dx, dy = X - cx, Y - cy
        shape = np.array([IDENTS[i][1] for i in ids[:, k]])[:, None, None]
        col = COLORS[[IDENTS[i][0] for i in ids[:, k]]]
        sq = (np.abs(dx) < 0.82 * R) & (np.abs(dy) < 0.82 * R)
        ci = dx ** 2 + dy ** 2 < R ** 2
        tr = (dy > -0.78 * R) & (dy < 1.0 * R) & (np.abs(dx) < (1.0 * R - dy) * 0.62)
        body = np.where(shape == 0, sq, np.where(shape == 1, ci, tr))
        u = unit(theta[:, k])
        ux, uy = u[:, 0][:, None, None], u[:, 1][:, None, None]
        along = dx * ux + dy * uy
        lat = np.abs(-dx * uy + dy * ux)
        handle = (along > 0.7 * R) & (along < R + HANDLE) & (lat < 0.011)
        m_present = present[:, k][:, None, None]
        img[handle & m_present] = HANDLE_RGB
        bmask = body & m_present
        img[bmask] = np.repeat(col[:, None, None, :], IMG, 1).repeat(IMG, 2)[bmask]
    # end-effector: white ring
    ex = ee[:, 0][:, None, None]
    ey = ee[:, 1][:, None, None]
    rr = np.sqrt((X - ex) ** 2 + (Y - ey) ** 2)
    img[np.abs(rr - 0.022) < 0.008] = EE_RGB
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def render_scenes(scenes, ee=None):
    pos = np.stack([s.pos for s in scenes])
    ids = np.stack([s.ids for s in scenes])
    th = np.stack([s.theta for s in scenes])
    pr = np.stack([s.present for s in scenes])
    bg = np.stack([s.bg for s in scenes])
    ee = np.stack([s.ee0 for s in scenes]) if ee is None else ee
    return render_batch(pos, ids, th, pr, ee, bg)


def to_pixel(xy):
    """workspace (x,y) -> (row, col) floats."""
    xy = np.asarray(xy)
    return np.stack([(1.0 - xy[..., 1]) * IMG - 0.5, xy[..., 0] * IMG - 0.5], -1)
