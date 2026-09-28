"""Four-stage lidar perception for a wheel-legged robot dog with a Hesai XT32.

1. preprocess  : self-filter, per-point deskew, gravity-aligned det frame, multi-sweep
2. detection   : CenterPoint-Pillar / CenterPoint-Voxel (det3d) behind one interface
3. tracking    : IMM (CV/CT) Kalman tracker in the world frame, two-stage association
4. prediction  : tier 1 IMM rollout, tier 2 learned multi-modal predictor
"""
