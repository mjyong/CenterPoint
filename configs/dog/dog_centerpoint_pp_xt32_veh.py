"""CenterPoint-Pillar fine-tuned on robot-dog XT32 data: vehicles only.

Heads: car, truck; output: vehicle (the detector's classes stay vehicle / pedestrian / cyclist, so
tracking, prediction and ROS are unchanged and pedestrian / cyclist heads can be added back later).
Labels are assigned to a head by box size (dog_perception.detection.boxes.vehicle_head); buses,
trailers and construction vehicles train the truck head. Dataset: export_dataset.py --classes vehicle,
which also marks uncertain tracks as ignore boxes (no loss either way). docs/DOG_FINETUNE.md, section 13.


Input: 5 sweeps @10 Hz from dog_perception.preprocess, mapped to the virtual
nuScenes lidar frame (ModelFrame), so z priors from nuScenes still hold.
Warm start: tools/dog/convert_nusc_ckpt.py on the nuScenes pillar weights.
Deployment (Orin): dense 2D convs only -> tools/dog/export_pillar_onnx.py -> two TensorRT engines.
"""
import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

# two vehicle heads split by size (< 5.5 m long and < 2.5 m high: car), warm-started from the nuScenes
# car and truck heads (tools/dog/convert_nusc_ckpt.py). The detector merges both into "vehicle".
tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=1, class_names=["truck"]),       # + buses, trailers, construction vehicles
]
class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

XY = 40.8                       # multiple of 0.2 * 8 (RPN stride)
PC_RANGE = [-XY, -XY, -5.0, XY, XY, 3.0]   # z in the virtual-nuScenes frame (ground at -1.84)
VOXEL = [0.2, 0.2, 8]

target_assigner = dict(tasks=tasks)

model = dict(
    type="PointPillars",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64, 64],
        num_input_features=5,
        with_distance=False,
        voxel_size=tuple(VOXEL),
        pc_range=tuple(PC_RANGE),
    ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[2, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[0.5, 1, 2],
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([128, 128, 128]),
        tasks=tasks,
        dataset="nuscenes",
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={"reg": (2, 2), "height": (1, 2), "dim": (3, 2), "rot": (2, 2), "vel": (2, 2)},
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-XY - 1, -XY - 1, -10.0, XY + 1, XY + 1, 10.0],
    max_per_img=500,
    nms=dict(nms_pre_max_size=1000, nms_post_max_size=83, nms_iou_threshold=0.2),
    circular_nms=True,
    min_radius=[4.0, 12.0],     # squared meters, per task (as nuScenes)
    score_threshold=0.1,
    pc_range=PC_RANGE[:2],
    out_size_factor=get_downsample_factor(model),
    voxel_size=VOXEL[:2],
)

dataset_type = "DogDataset"
nsweeps = 5
data_root = "data/dog_veh"

db_sampler = dict(
    type="GT-AUG",
    enable=True,
    db_info_path="data/dog_veh/dbinfos_train_5sweeps_withvelo.pkl",
    sample_groups=[dict(car=2), dict(truck=3)],     # trucks are the rarer, harder class
    db_prep_steps=[
        dict(filter_by_min_num_points=dict(car=10, truck=10)),
        dict(filter_by_difficulty=[-1]),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.785, 0.785],
    global_scale_noise=[0.95, 1.05],
    global_translate_std=0.1,
    global_pitch_roll_noise=0.035,    # +-2 deg: residual gravity-alignment error / slopes
    db_sampler=db_sampler,
    class_names=class_names,
)
val_preprocessor = dict(mode="val", shuffle_points=False)

voxel_generator = dict(
    range=PC_RANGE,
    voxel_size=VOXEL,
    max_points_in_voxel=20,
    max_voxel_num=[30000, 30000],     # = static P_max of the exported PFN graph
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    # older sweeps shifted / turned as INS pose errors would (det3d/datasets/pipelines/dog_aug.py)
    dict(type="SweepPoseJitter", prob=0.3, sigma_v=0.5, max_v=1.5, sigma_w_deg=3.0, max_w_deg=10.0),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "data/dog_veh/infos_train.pkl"
val_anno = "data/dog_veh/infos_val.pkl"
test_anno = "data/dog_veh/infos_test.pkl"      # export_dataset.py --test-recordings; dist_test.py --testset

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    # half the 0921 training frames have no vehicle; a quarter as many as the class-balanced set are
    # added back so background-only scenes are seen too
    train=dict(type=dataset_type, root_path=data_root, info_path=train_anno, ann_file=train_anno,
               nsweeps=nsweeps, class_names=class_names, pipeline=train_pipeline, empty_frames=0.25),
    val=dict(type=dataset_type, root_path=data_root, info_path=val_anno, test_mode=True, ann_file=val_anno,
             nsweeps=nsweeps, class_names=class_names, pipeline=test_pipeline),
    test=dict(type=dataset_type, root_path=data_root, info_path=test_anno, test_mode=True, ann_file=test_anno,
              nsweeps=nsweeps, class_names=class_names, pipeline=test_pipeline),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False)
# fine-tuning: lower peak LR than the from-scratch nuScenes schedule
lr_config = dict(type="one_cycle", lr_max=0.0005, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.3)

checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type="TextLoggerHook")])
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "./work_dirs/{}/".format(__file__[__file__.rfind("/") + 1:-3])
load_from = "work_dirs/dog_pp_veh_init.pth"   # tools/dog/convert_nusc_ckpt.py output
resume_from = None
workflow = [("train", 1)]
