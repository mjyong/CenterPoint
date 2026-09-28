from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset
from .dog import DogDataset

dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset,
    "DOG": DogDataset,
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
