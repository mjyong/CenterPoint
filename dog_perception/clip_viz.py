"""Draw one frame of a clip (ground truth vs detection / tracking / prediction)
into a Foxglove MCAP. Dataset independent: callers pass boxes in the map frame.

Topics (each can be toggled in the 3D panel):

    /tf                       map -> ego -> sensors (follow "ego" in the 3D panel)
    /lidar                    current sweep, map frame, intensity
    /gt/boxes                 GT wireframes: green = tracked, red = missed (FN),
                              grey = not evaluated (no points / out of range /
                              other class), thin green = interpolated (non-keyframe)
    /gt/paths                 GT past 2 s (solid) and future 3 s (dashed)
    /det/boxes                reported detections, filled by class; FP outlined red
    /det/low_score            detections kept only for the tracker (score < report threshold)
    /det/filtered             detections the geometric filters dropped, with the reason
    /track/boxes              confirmed tracks, wireframe colored by track id,
                              velocity arrow; label "T<id> class score speed"
    /track/history            past positions of every track
    /pred/paths               predicted modes (width / alpha by probability),
                              1-sigma ellipse at the horizon, FDE segment to the GT end point
    /events                   ID switches, missed objects, false tracks (keyframes)
    /ego                      ego box, speed, path
    /blind                    the lidar's blind sector and the edge bands next to it (ego frame)
    /cam/<CH>/image|calibration|annotations   keyframe images with projected GT / tracks
    /metrics/frame            per-keyframe numbers for the Plot panel
    /log                      ID switches etc. for the Log panel

Clicking an entity shows its metadata (scores, ids, errors, attributes).
"""
import numpy as np

from . import foxglove as fx
from .detection.boxes import CLASSES

# Class colors stay blue / orange / gold. Track ids use a separate palette so a
# track wireframe is not the same color as the detection of that class.
CLASS_RGB = {0: fx.hex_rgba("#3987e5"), 1: fx.hex_rgba("#d95926"), 2: fx.hex_rgba("#c98500")}
TRACK_RGB = [fx.hex_rgba(h) for h in ("#14b8a6", "#db2777", "#7c3aed", "#22d3ee", "#f472b6", "#a78bfa")]
# GT status uses the reserved status colors (good / critical), always with a text label
GT_RGB = {"tracked": fx.hex_rgba("#0ca30c"), "missed": fx.hex_rgba("#d03b3b"), "ignored": fx.hex_rgba("#898781"),
          "interp": fx.hex_rgba("#0ca30c", 0.5)}
WHITE, RED, GREY = (1, 1, 1, 1), fx.hex_rgba("#d03b3b"), fx.hex_rgba("#898781")
EVENT_PX = 24          # /events labels, screen pixels; one size for ID SWITCH, MISSED, FALSE TRACK
SHORT = {"vehicle": "veh", "pedestrian": "ped", "cyclist": "cyc"}


def track_rgb(tid, a=1.0):
    r = TRACK_RGB[int(tid) % len(TRACK_RGB)]
    return r[:3] + (a,)


def _label(label):
    return SHORT.get(CLASSES[label], "?") if 0 <= label < len(CLASSES) else "other"


class ClipVisualizer:
    def __init__(self, writer, frame_period=0.1, keyframe_period=0.5):
        self.w = writer
        self.life = 1.5 * frame_period
        self.key_life = keyframe_period
        self.ego_path = []

    # ---------------------------------------------------------------- frames
    def transforms(self, t, items):
        self.w.write("/tf", "foxglove.FrameTransforms", t, fx.frame_transforms(t, items))

    def lidar(self, t, xyz_map, intensity):
        self.w.write("/lidar", "foxglove.PointCloud", t, fx.point_cloud(t, "map", xyz_map, intensity=intensity))

    def ego(self, t, T_map_ego, speed, size=(4.084, 1.730, 1.562), offset=1.0):
        """``offset``: box center ahead of the ego origin along ego x (nuScenes: the rear axle, 1 m
        behind the center); the origin is on the ground."""
        p = T_map_ego[:3, 3]
        yaw = float(np.arctan2(T_map_ego[1, 0], T_map_ego[0, 0]))
        self.ego_path.append(p.copy())
        self.ego_path = self.ego_path[-60:]
        c = p + T_map_ego[:3, :3] @ np.array([offset, 0.0, size[2] / 2])
        ents = [fx.entity("ego", t, lifetime=self.life, metadata={"speed_mps": "%.2f" % speed},
                          cubes=[fx.cube(c, size, yaw, (0.85, 0.85, 0.85, 0.35))],
                          texts=[fx.text(c + [0, 0, 1.6], "ego %.1f m/s" % speed, WHITE, 13)],
                          lines=[fx.line(np.array(self.ego_path), (0.85, 0.85, 0.85, 0.6), 0.1)]
                          if len(self.ego_path) > 1 else [])]
        self.w.write("/ego", "foxglove.SceneUpdate", t, fx.scene_update(ents))

    def ground_truth(self, t, items, keyframe):
        """items: dicts with box (7,), id, label, status, vel (2,) or NaN, past (K,3),
        future (K,3), meta (dict)."""
        life = self.key_life if keyframe else self.life
        boxes, paths = [], []
        for it in items:
            rgb = GT_RGB[it["status"]]
            b = it["box"]
            thick = 0.09 if it["status"] == "missed" else (0.03 if it["status"] == "interp" else 0.05)
            prims = dict(lines=[fx.box_wire(b, rgb, thick)], texts=[], arrows=[])
            if it["status"] != "interp":
                v = it.get("vel")
                spd = float(np.hypot(*v)) if v is not None and np.all(np.isfinite(v)) else float("nan")
                tag = "GT %s #%d" % (_label(it["label"]), it["id"]) + ("" if np.isnan(spd) else " %.1fm/s" % spd)
                if it["status"] == "missed":
                    tag = "FN " + tag
                prims["texts"].append(fx.text(b[:3] - [0, 0, b[5] / 2 + 0.2], tag, rgb, 11))
                if not np.isnan(spd) and spd > 0.3:
                    prims["arrows"].append(fx.arrow(b[:3] + [0, 0, b[5] / 2], [v[0], v[1], 0], rgb, 0.08))
            boxes.append(fx.entity("gt-%d" % it["id"], t, lifetime=life, metadata=it.get("meta"), **prims))
            ln = []
            if it.get("past") is not None and len(it["past"]) > 1:
                ln.append(fx.line(it["past"], GT_RGB["tracked"][:3] + (0.7,), 0.05))
            if it.get("future") is not None and len(it["future"]) > 1:
                d = fx.dashed(it["future"], 0.5, 0.35)
                if len(d):
                    ln.append(fx.line(d, WHITE[:3] + (0.85,), 0.06, "list"))
            if ln:
                paths.append(fx.entity("gtpath-%d" % it["id"], t, lifetime=life, lines=ln))
        self.w.write("/gt/boxes", "foxglove.SceneUpdate", t, fx.scene_update(boxes))
        self.w.write("/gt/paths", "foxglove.SceneUpdate", t, fx.scene_update(paths))

    def detections(self, t, dets_map, reported_mask, tp=None, matched_gt=None):
        """dets_map: Detections in the map frame; tp / matched_gt only on keyframes."""
        rep, low = [], []
        for i, (b, s, l, v) in enumerate(zip(dets_map.boxes, dets_map.scores, dets_map.labels, dets_map.velocities)):
            if int(l) != 0:              # vehicle only
                continue
            rgb = CLASS_RGB.get(int(l), GREY)
            meta = {"class": CLASSES[l], "score": "%.3f" % s, "speed_mps": "%.2f" % np.hypot(*v),
                    "vel": "(%.2f, %.2f)" % tuple(v), "size_lwh": "%.2f %.2f %.2f" % tuple(b[3:6])}
            if not reported_mask[i]:
                low.append(fx.entity("low-%d" % i, t, lifetime=self.life, metadata=meta,
                                     lines=[fx.box_wire(b, rgb[:3] + (0.35,), 0.02)]))
                continue
            prims = dict(cubes=[fx.cube(b[:3], b[3:6], b[6], rgb[:3] + (0.28,))], lines=[], texts=[], arrows=[])
            label = "%s %.2f" % (_label(l), s)
            if tp is not None:
                meta["eval"] = "TP" if tp[i] else "FP"
                if tp[i] and matched_gt is not None:
                    meta["gt_id"] = matched_gt.get(i, "")
                if not tp[i]:
                    prims["lines"].append(fx.box_wire(b, RED, 0.07))
                    label = "FP " + label
            prims["texts"].append(fx.text(b[:3] + [0, 0, b[5] / 2 + 0.25], label, rgb, 10))
            prims["arrows"].append(fx.arrow(b[:3], [v[0], v[1], 0], rgb[:3] + (0.8,), 0.06))
            rep.append(fx.entity("det-%d" % i, t, lifetime=self.life, metadata=meta, **prims))
        self.w.write("/det/boxes", "foxglove.SceneUpdate", t, fx.scene_update(rep))
        self.w.write("/det/low_score", "foxglove.SceneUpdate", t, fx.scene_update(low))

    def filtered(self, t, dets_map, reasons):
        """Detections the geometric filters dropped (``filters.filter_reasons``), reason as the label."""
        ents = []
        for i, (b, s, l) in enumerate(zip(dets_map.boxes, dets_map.scores, dets_map.labels)):
            if int(l) != 0:
                continue
            ents.append(fx.entity("filt-%d" % i, t, lifetime=self.life,
                                  metadata={"class": CLASSES[l], "score": "%.3f" % s, "dropped": reasons[i]},
                                  lines=[fx.box_wire(b, GREY[:3] + (0.6,), 0.02)],
                                  texts=[fx.text(b[:3] + [0, 0, b[5] / 2 + 0.25], "%s %.2f" % (reasons[i], s),
                                                 GREY, 9)]))
        self.w.write("/det/filtered", "foxglove.SceneUpdate", t, fx.scene_update(ents))

    def sector(self, t, wedge, edge=20.0, radius=40.0, frame_id="ego"):
        """The blind ``wedge`` (lo, hi) [deg, 0 = ahead along +y of ``frame_id``, positive towards +x]
        and the ``edge`` bands beside it, as outlines on the ground."""
        def fan(lo, hi):
            a = np.radians(np.linspace(lo, hi, max(2, int(abs(hi - lo) / 3))))
            return np.vstack([[0, 0, 0.05], np.c_[radius * np.sin(a), radius * np.cos(a), np.full(len(a), 0.05)],
                              [0, 0, 0.05]])
        lo, hi = wedge
        ents = [fx.entity("blind", t, frame_id=frame_id, lifetime=self.life,
                          lines=[fx.line(fan(lo, hi), GREY[:3] + (0.9,), 0.08),
                                 fx.line(fan(lo - edge, lo), fx.hex_rgba("#c98500", 0.7), 0.04),
                                 fx.line(fan(hi, hi + edge), fx.hex_rgba("#c98500", 0.7), 0.04)],
                          texts=[fx.text([0.5 * radius * np.sin(np.radians((lo + hi) / 2)),
                                          0.5 * radius * np.cos(np.radians((lo + hi) / 2)), 0.5], "blind", GREY, 11)])]
        self.w.write("/blind", "foxglove.SceneUpdate", t, fx.scene_update(ents))

    def tracks(self, t, tracks, info=None):
        """tracks: TrackState list (map frame). info: track id -> extra metadata
        (matched GT id, distance, FP) on keyframes."""
        boxes, hist = [], []
        for s in tracks:
            if int(s.label) != 0:
                continue
            rgb = track_rgb(s.track_id, 0.45 if s.coasting else 1.0)
            b = np.r_[s.position, s.size, s.yaw]
            spd = float(np.hypot(*s.velocity))
            meta = {"track_id": s.track_id, "class": s.name, "score": "%.3f" % s.score, "speed_mps": "%.2f" % spd,
                    "hits": s.hits, "age_s": "%.1f" % s.age, "coasting": s.coasting,
                    "mode_probs": " ".join("%.2f" % p for p in s.mode_probs),
                    "pos_std_m": "%.2f" % np.sqrt(max(np.trace(s.covariance[:2, :2]) / 2, 0))}
            extra = (info or {}).get(s.track_id, {})
            meta.update(extra)
            motion = getattr(s, "motion", "static")
            meta["motion"] = motion
            label = "T%d %s %.2f %.1fm/s" % (s.track_id, _label(s.label), s.score, spd) + (" coast" if s.coasting else "")
            if motion != "static":
                label += " " + motion
            if extra.get("eval") == "FP":
                label = "FP " + label
            prims = dict(lines=[fx.box_wire(b, rgb, 0.08)],
                         texts=[fx.text(b[:3] + [0, 0, b[5] / 2 + 0.7], label, rgb, 12)],
                         arrows=[fx.arrow(b[:3] + [0, 0, b[5] / 2], [s.velocity[0], s.velocity[1], 0], rgb, 0.1)])
            boxes.append(fx.entity("trk-%d" % s.track_id, t, lifetime=self.life, metadata=meta, **prims))
            h = np.asarray(s.history)
            if len(h) > 1:
                pts = np.column_stack([h[:, 1], h[:, 2], np.full(len(h), s.position[2] - s.size[2] / 2 + 0.1)])
                hist.append(fx.entity("hist-%d" % s.track_id, t, lifetime=self.life,
                                      lines=[fx.line(pts, rgb[:3] + (0.8,), 0.07)],
                                      spheres=[fx.sphere(p, 0.15, rgb[:3] + (0.8,)) for p in pts[h[:, 5] > 0.5][::3]]))
        self.w.write("/track/boxes", "foxglove.SceneUpdate", t, fx.scene_update(boxes))
        self.w.write("/track/history", "foxglove.SceneUpdate", t, fx.scene_update(hist))

    def predictions(self, t, preds, z_of, fde=None):
        """preds: Prediction list (map frame); z_of: track id -> z for drawing;
        fde: track id -> (gt end xy, fde_best, minfde) on keyframes."""
        ents = []
        for p in preds:
            z = z_of.get(p.track_id, 0.0)
            start = np.asarray(z_of.get(("xy", p.track_id), p.modes[0, 0]))
            prims = dict(lines=[], spheres=[], texts=[])
            order = np.argsort(-p.probs)
            for rank, k in enumerate(order):
                pr = float(p.probs[k])
                rgb = track_rgb(p.track_id, 0.35 + 0.65 * pr)
                pts = np.vstack([np.r_[start, z], np.column_stack([p.modes[k], np.full(len(p.times), z)])])
                prims["lines"].append(fx.line(pts, rgb, 0.04 + 0.16 * pr))
                prims["spheres"].append(fx.sphere(pts[-1], 0.25 + 0.3 * pr, rgb))
                if rank < 3:
                    prims["texts"].append(fx.text(pts[-1] + [0, 0, 0.4], "%.0f%%" % (100 * pr), rgb, 10))
                if rank == 0 and p.covs is not None:
                    prims["lines"].append(fx.line(fx.ellipse(p.modes[k, -1], p.covs[k, -1], 1.0, z), rgb, 0.04,
                                                  "loop"))
            meta = {"track_id": p.track_id, "source": p.source, "modes": len(p.probs),
                    "probs": " ".join("%.2f" % q for q in p.probs[order])}
            if fde is not None and p.track_id in fde:
                gt_end, f1, fmin = fde[p.track_id]
                best = p.modes[order[0], -1]
                prims["lines"].append(fx.line([np.r_[best, z], np.r_[gt_end, z]], RED, 0.05))
                prims["texts"].append(fx.text(np.r_[(best + gt_end) / 2, z + 0.3], "FDE %.1fm" % f1, RED, 10))
                meta.update(FDE_top_m="%.2f" % f1, minFDE_m="%.2f" % fmin)
            ents.append(fx.entity("pred-%d" % p.track_id, t, lifetime=self.life, metadata=meta, **prims))
        self.w.write("/pred/paths", "foxglove.SceneUpdate", t, fx.scene_update(ents))

    def events(self, t, ev, gt_pos, trk_pos):
        """ev: MotAccumulator events; *_pos: id -> xyz in the map frame."""
        ents, logs = [], []
        for g, old, new in ev["idsw"]:
            p = gt_pos.get(g)
            if p is None:
                continue
            ents.append(fx.entity("idsw-%d" % g, t, lifetime=self.key_life,
                                  texts=[fx.text(p + [0, 0, 3.0], "ID SWITCH T%d->T%d" % (old, new), WHITE, EVENT_PX)],
                                  lines=[fx.line([p, p + [0, 0, 2.7]], RED, 0.12)]))
            logs.append("ID switch: GT #%d moved from track T%d to T%d" % (g, old, new))
        for g in ev["fn"]:
            p = gt_pos.get(g)
            if p is not None:
                ents.append(fx.entity("fn-%d" % g, t, lifetime=self.key_life,
                                      texts=[fx.text(p + [0, 0, 2.2], "MISSED", RED, EVENT_PX)]))
        for tid in ev["fp"]:
            p = trk_pos.get(tid)
            if p is not None:
                ents.append(fx.entity("fp-%d" % tid, t, lifetime=self.key_life,
                                      texts=[fx.text(p + [0, 0, 2.2], "FALSE TRACK", RED, EVENT_PX)]))
        self.w.write("/events", "foxglove.SceneUpdate", t, fx.scene_update(ents))
        for m in logs:
            self.w.write("/log", "foxglove.Log", t, fx.log(t, m, level=3))

    def camera(self, t, channel, cam, jpeg, boxes_map):
        """boxes_map: [(box (7,), rgba, label)] drawn into the image."""
        frame_id = channel
        self.w.write("/cam/%s/image" % channel, "foxglove.CompressedImage", t, fx.compressed_image(t, frame_id, jpeg))
        self.w.write("/cam/%s/calibration" % channel, "foxglove.CameraCalibration", t,
                     fx.camera_calibration(t, frame_id, cam["K"], cam["width"], cam["height"]))
        T_cam_map = np.linalg.inv(cam["T_global_cam"])
        lines, texts = [], []
        for b, rgba, lab in boxes_map:
            c = fx.box_corners(b) @ T_cam_map[:3, :3].T + T_cam_map[:3, 3]
            if (c[:, 2] > 0.5).sum() < 8:
                continue                      # partly behind the camera: skip rather than draw it wrong
            uv = c @ cam["K"].T
            uv = uv[:, :2] / uv[:, 2:3]
            if uv[:, 0].max() < 0 or uv[:, 0].min() > cam["width"] or uv[:, 1].max() < 0 or uv[:, 1].min() > cam["height"]:
                continue
            lines.append((np.array([uv[i] for e in fx.BOX_EDGES[:12] for i in e]), rgba, 2.0))
            texts.append(((uv[:, 0].min(), uv[:, 1].min() - 4), lab, rgba, 22))
        self.w.write("/cam/%s/annotations" % channel, "foxglove.ImageAnnotations", t,
                     fx.image_annotations(t, lines, texts))

    def metrics(self, t, values):
        self.w.write("/metrics/frame", "dog.FrameMetrics", t, {k: (None if isinstance(v, float) and np.isnan(v)
                                                                    else v) for k, v in values.items()})


def bev_figure(snap, path, extent=40.0, forward="x", ego_outline=None, sector=None):
    """Static bird's-eye view of one keyframe for the report (light surface), heading up.

    ``snap["T_global_ego"]``: the frame the view is centered on; ``forward``: its axis that points
    ahead ("x" for the nuScenes ego, "y" for the kygi det frame). ``ego_outline``: (K, 2) polygon of the
    robot in the view (right, forward) [m], default the nuScenes car. ``sector``: (lo, hi) azimuth
    [deg, 0 = ahead, positive to the right] drawn as a shaded wedge (the lidar's blind sector)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon

    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    CLS = {0: "#2a78d6", 1: "#eb6834", 2: "#eda100"}
    STATUS = {"tracked": "#0ca30c", "missed": "#d03b3b", "ignored": "#c3c2b7"}
    T = snap["T_global_ego"]
    R, o = T[:2, :2], T[:2, 3]

    def P(xy):                                   # map -> ego frame rotated so the heading points up
        e = (np.asarray(xy)[..., :2] - o) @ R
        return np.stack([-e[..., 1], e[..., 0]], axis=-1) if forward == "x" else e

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=SURF)
    ax.set_facecolor(SURF)
    if sector is not None:
        a = np.radians(np.linspace(sector[0], sector[1], 40))
        r = 1.5 * extent
        ax.add_patch(Polygon(np.vstack([[0, 0], np.c_[r * np.sin(a), r * np.cos(a)]]), closed=True, fc=GRID,
                             ec="none", alpha=0.6, zorder=0))
    pts = P(snap["lidar"])
    m = (np.abs(pts) < extent).all(1)
    ax.scatter(pts[m, 0], pts[m, 1], s=0.3, c="#c3c2b7", linewidths=0, rasterized=True, zorder=1)
    for it in snap["gt"]:
        c = P(fx.box_corners(it["box"])[:4])
        if not (np.abs(c) < extent).all():
            continue
        st = it["status"]
        ax.add_patch(Polygon(c, closed=True, fill=False, ec=STATUS.get(st, MUTED), lw=1.6 if st != "ignored" else 0.8,
                             zorder=4))
        if st == "missed":
            ax.text(c[:, 0].mean(), c[:, 1].max() + 0.6, "FN", color=STATUS["missed"], fontsize=7, ha="center",
                    zorder=6)
        if st != "ignored" and it.get("future") is not None and len(it["future"]) > 1:
            f = P(it["future"])
            ax.plot(f[:, 0], f[:, 1], ls=(0, (3, 2)), color=INK, lw=0.9, zorder=3)
    for s in snap["tracks"]:
        if int(s.label) != 0:
            continue
        c = P(fx.box_corners(np.r_[s.position, s.size, s.yaw])[:4])
        if not (np.abs(c) < extent).all():
            continue
        col = CLS.get(int(s.label), MUTED)
        ax.add_patch(Polygon(c, closed=True, fc=col, ec=col, alpha=0.35 if s.coasting else 0.55, lw=0.8, zorder=5))
        ax.text(c[:, 0].max() + 0.3, c[:, 1].mean(), "T%d" % s.track_id, color=INK2, fontsize=6.5, zorder=7)
    labels = {s.track_id: int(s.label) for s in snap["tracks"] if int(s.label) == 0}
    start = {s.track_id: s.position[:2] for s in snap["tracks"] if int(s.label) == 0}
    for p in snap["preds"]:
        if p.track_id not in start or not (np.abs(P(start[p.track_id])) < extent).all():
            continue
        for k in np.argsort(-p.probs)[:2]:
            xy = P(np.vstack([start[p.track_id], p.modes[k]]))
            ax.plot(xy[:, 0], xy[:, 1], color=CLS.get(labels[p.track_id], MUTED), lw=1.6,
                    alpha=0.25 + 0.75 * float(p.probs[k]), zorder=6)
    ego = np.array([[-0.865, -1.0], [0.865, -1.0], [0.865, 3.1], [-0.865, 3.1]]) if ego_outline is None \
        else np.asarray(ego_outline, float)        # default: the nuScenes car, rear axle at the origin
    ax.add_patch(Polygon(ego, closed=True, fc=INK2, ec=INK2, zorder=8))
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_xlabel("right [m]", color=MUTED, fontsize=9)
    ax.set_ylabel("forward [m]", color=MUTED, fontsize=9)
    mt = snap.get("metrics") or {}
    ax.set_title("%s  t=%.1f s  |  GT %s, tracked %s, missed %s, false tracks %s, ID switches %s" % (
        snap["scene"], snap["t"] - snap.get("t0", snap["t"]), mt.get("gt", "-"), mt.get("trk_tp", "-"),
        mt.get("trk_fn", "-"), mt.get("trk_fp", "-"), mt.get("idsw", "-")), color=INK, fontsize=10, loc="left")
    handles = [Line2D([], [], color=STATUS["tracked"], lw=1.6, label="GT, tracked"),
               Line2D([], [], color=STATUS["missed"], lw=1.6, label="GT, missed (FN)"),
               Line2D([], [], color=STATUS["ignored"], lw=0.8, label="GT, not evaluated"),
               Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), label="GT future 3 s")]
    handles += [Patch(fc=CLS[c], alpha=0.55, label="track: %s" % n) for c, n in enumerate(CLASSES) if n == "vehicle"]
    handles += [Line2D([], [], color=INK2, lw=1.6, label="predicted modes (opacity = prob.)")]
    if sector is not None:
        handles.append(Patch(fc=GRID, alpha=0.6, label="lidar blind sector"))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=True, facecolor=SURF, edgecolor=GRID,
              labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURF)
    plt.close(fig)
