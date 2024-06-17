import json
from pathlib import Path
import sys
sys.path.append("/homes/ljchen/code/EMRxTask1-master")
from utils.Transform import BackUp, Loadh5
from utils.utils import init_seeds, record_transform
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CenterSpatialCropd,
    RandFlipd,
    RandZoomd,
    RandRotated,
    LoadImaged,
    EnsureTyped,
    ThresholdIntensityd,
    CenterSpatialCropd,
)
from monai.data import CacheDataset
from monai.apps.reconstruction.transforms.dictionary import ReferenceBasedNormalizeIntensityd

def generate_train_val_ds(plans_args):
    train_transform = Compose([
        LoadImaged(keys=["kspace_masked_ifft", "reconstruction_rss"], image_only=True),
        EnsureChannelFirstd(keys=["kspace_masked_ifft", "reconstruction_rss"]),
        BackUp(keys=["kspace_masked_ifft", "reconstruction_rss"]),
        #ReferenceBasedSpatialCropd(keys=["kspace_masked_ifft"], ref_key="reconstruction_rss"), 大小本来就一致，没必要裁剪
        CenterSpatialCropd(keys=["kspace_masked_ifft", "reconstruction_rss",
                                "kspace_masked_ifft_copy", "reconstruction_rss_copy"], roi_size=[162, 512]),
        ReferenceBasedNormalizeIntensityd(
            keys=["kspace_masked_ifft", "reconstruction_rss"], ref_key="kspace_masked_ifft", channel_wise=True
        ),
        ThresholdIntensityd(
            keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=6.0, above=False, cval=6.0
        ),
        ThresholdIntensityd(
            keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=-6.0, above=True, cval=-6.0
        ),
        EnsureTyped(keys=["kspace_masked_ifft", "reconstruction_rss"]),
        ])

    val_transform = Compose([
        LoadImaged(keys=["kspace_masked_ifft", "reconstruction_rss"], image_only=True),
        EnsureChannelFirstd(keys=["kspace_masked_ifft", "reconstruction_rss"]),
        BackUp(keys=["kspace_masked_ifft", "reconstruction_rss"]),
        #ReferenceBasedSpatialCropd(keys=["kspace_masked_ifft"], ref_key="reconstruction_rss"), 大小本来就一致，没必要裁剪
        CenterSpatialCropd(keys=["kspace_masked_ifft", "reconstruction_rss",
                                "kspace_masked_ifft_copy", "reconstruction_rss_copy"], roi_size=[162, 512]),
        ReferenceBasedNormalizeIntensityd(
            keys=["kspace_masked_ifft", "reconstruction_rss"], ref_key="kspace_masked_ifft", channel_wise=True
        ),
        ThresholdIntensityd(
            keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=6.0, above=False, cval=6.0
        ),
        ThresholdIntensityd(
            keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=-6.0, above=True, cval=-6.0
        ),
        EnsureTyped(keys=["kspace_masked_ifft", "reconstruction_rss"]),
        ])

    # 记录一下用了哪些transform，概率是多少
    trans = record_transform(train_transform)
    plans_args.config["trans"] = trans

    # setup data_loader instances
    with open(plans_args.json_path) as f:
        data = json.load(f)
    train_ds = CacheDataset(
        data["train"],
        transform=train_transform, num_workers=8, cache_rate=0
        )
    val_ds = CacheDataset(
        data["validation"],
        transform=val_transform, num_workers=8, cache_rate=0
        )
    return train_ds, val_ds, plans_args

def test():
    import argparse
    with open("/homes/syli/python/CMRxRecon2024/config/base.json") as f:
        config = json.load(f)
    
    plans_args = argparse.Namespace(**config)
    plans_args.config = config
    train_ds, val_ds, _ = generate_train_val_ds(plans_args)
    for i in train_ds:
        print(i["kspace_masked_ifft"].shape)

def promptmr_test(plans_args):
    train_data = [str(i) for i in Path(plans_args.json_path).iterdir()]
    print(train_data)
    loader = Loadh5()
    train_ds = CacheDataset(
        data=train_data,
        transform=loader,
        cache_rate=0
    )
    val_ds = CacheDataset(
        data=train_data,
        transform=loader,
        cache_rate=0
    )
    return train_ds, val_ds, plans_args
    

if __name__ == "__main__":
    train_ds, _ = promptmr_test("/homes/ljchen/data/cmrecon_temp")
    for i in train_ds:
        print(i["kspace"].shape)
        print(i["mask"].shape)
