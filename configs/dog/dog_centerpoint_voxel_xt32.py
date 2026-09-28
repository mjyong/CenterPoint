"""CenterPoint-Voxel (0.075 m, sparse 3D conv) fine-tuned on robot-dog XT32 data.

Higher accuracy / offline teacher for auto-labelling; needs spconv, so it
runs on a GPU (Orin / workstation), not on the BPU.
Warm start: tools/dog/convert_nusc_ckpt.py on nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.
z range must stay [-5, 3] (41 z cells -> 256-channel BEV expected by the neck).
"""
import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

# one task per class: each gets its own heat-map head (warm-started from car / pedestrian / bicycle)
tasks = [
    dict(num_class=1, class_names=["vehicle"]),
    dict(num_class=1, class_names=["pedestrian"]),
    dict(num_class=1, class_names=["cyclist"]),
]
class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

XY = 40.8                       # 1088 cells: multiple of 16 (sparse backbone x RPN stride)
PC_RANGE = [-XY, -XY, -5.0, XY, XY, 3.0]   # z in the virtual-nuScenes frame (ground at -1.84)
VOXEL = [0.075, 0.075, 0.2]

target_assigner = dict(tasks=tasks)

model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(type="VoxelFeatureExtractorV3", num_input_features=5),
    backbone=dict(type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset="nuscenes",
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={"reg": (2, 2), "height": (1, 2), "dim": (3, 2), "rot": (2, 2), "vel": (2, 2)},
        share_conv_channel=64,
        dcn_head=False,
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
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
    min_radius=[4.0, 0.175, 0.85],     # squared meters: vehicle, pedestrian, cyclist
    score_threshold=0.1,
    pc_range=PC_RANGE[:2],
    out_size_factor=get_downsample_factor(model),
    voxel_size=VOXEL[:2],
)

dataset_type = "DogDataset"
nsweeps = 5
data_root = "data/dog"

db_sampler = dict(
    type="GT-AUG",
    enable=True,
    db_info_path="data/dog/dbinfos_train_5sweeps_withvelo.pkl",
    sample_groups=[dict(vehicle=2), dict(pedestrian=4), dict(cyclist=4)],
    db_prep_steps=[
        dict(filter_by_min_num_points=dict(vehicle=10, pedestrian=5, cyclist=5)),
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
    max_points_in_voxel=10,
    max_voxel_num=[120000, 160000],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
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

train_anno = "data/dog/infos_train.pkl"
val_anno = "data/dog/infos_val.pkl"
test_anno = None

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(type=dataset_type, root_path=data_root, info_path=train_anno, ann_file=train_anno,
               nsweeps=nsweeps, class_names=class_names, pipeline=train_pipeline),
    val=dict(type=dataset_type, root_path=data_root, info_path=val_anno, test_mode=True, ann_file=val_anno,
             nsweeps=nsweeps, class_names=class_names, pipeline=test_pipeline),
    test=dict(type=dataset_type, root_path=data_root, info_path=val_anno, test_mode=True, ann_file=val_anno,
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
load_from = "work_dirs/dog_voxel_init.pth"   # tools/dog/convert_nusc_ckpt.py output
resume_from = None
workflow = [("train", 1)]
