import numpy as np

from dog_perception.autolabel import auto_label
from dog_perception.detection import CLASSES, OracleDetector, OracleNoise
from dog_perception.detection.evaluate import evaluate_detections
from dog_perception.geometry import inv_T
from dog_perception.preprocess import det_frame_from_body
from dog_perception.sim import GaitModel, RobotTrajectory, SimObject


def test_auto_label_beats_raw_teacher():
    rng = np.random.default_rng(0)
    objs = [SimObject.random(i, "pedestrian", rng, 21.0, (0, 0), (15, 10)) for i in range(8)]
    robot = RobotTrajectory(GaitModel(speed=0.3))
    det = OracleDetector(OracleNoise(false_positive_rate=0.5, pos_std=0.15, yaw_flip_prob=0.1), seed=2)
    frames, gts = [], []
    for k in range(200):
        t = (k + 1) / 10
        T_wd = det_frame_from_body(robot.T(t), 0.45)
        gt = []
        for o in objs:
            c, y, v = o.state([t])
            gt.append(dict(id=o.id, label=o.label, center=c[0], size=o.size, yaw=float(y[0]), velocity=v[0], num_points=12))
        frames.append((t, det(gt, inv_T(T_wd), t).transform(T_wd, "world")))
        gts.append(dict(boxes=np.array([[*g["center"], *g["size"], g["yaw"]] for g in gt]),
                        labels=np.array([CLASSES.index(g["label"]) for g in gt]),
                        velocities=np.array([g["velocity"] for g in gt])))
    labels = auto_label(frames)
    raw = [dict(boxes=d.boxes, labels=d.labels, scores=d.scores, velocities=d.velocities) for _, d in frames]
    al = []
    for t, _ in frames:
        L = labels[round(t, 6)]
        al.append(dict(boxes=np.array([l["box"] for l in L]).reshape(-1, 7), labels=np.array([l["label"] for l in L], int),
                       scores=np.array([l["score"] for l in L]), velocities=np.array([l["velocity"] for l in L]).reshape(-1, 2)))
    r_raw = evaluate_detections(raw, gts, CLASSES)["pedestrian"]
    r_al = evaluate_detections(al, gts, CLASSES)["pedestrian"]
    assert r_al["AP@0.5"] > r_raw["AP@0.5"] + 0.2
    assert r_al["ATE"] < 0.7 * r_raw["ATE"] and r_al["AVE"] < 0.7 * r_raw["AVE"]
