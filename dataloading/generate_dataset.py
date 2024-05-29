import json
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
from utils.Transform import BackUp
from monai.apps.reconstruction.transforms.dictionary import ReferenceBasedNormalizeIntensityd
from utils.utils import init_seeds, record_transform

def generate_train_val_ds(plans_args):
    train_transform = Compose([
        LoadImaged(keys=["kspace_masked_ifft", "reconstruction_rss"], image_only=True),
        EnsureChannelFirstd(keys=["kspace_masked_ifft", "reconstruction_rss"]),
        BackUp(keys=["kspace_masked_ifft", "reconstruction_rss"]),
        #ReferenceBasedSpatialCropd(keys=["kspace_masked_ifft"], ref_key="reconstruction_rss"), 大小本来就一致，没必要裁剪
        # CenterSpatialCropd(keys=["kspace_masked_ifft", "reconstruction_rss",
        #                         "kspace_masked_ifft_copy", "reconstruction_rss_copy"], roi_size=[162, 512]),
        # ReferenceBasedNormalizeIntensityd(
        #     keys=["kspace_masked_ifft", "reconstruction_rss"], ref_key="kspace_masked_ifft", channel_wise=True
        # ),
        # ThresholdIntensityd(
        #     keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=6.0, above=False, cval=6.0
        # ),
        # ThresholdIntensityd(
        #     keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=-6.0, above=True, cval=-6.0
        # ),
        # EnsureTyped(keys=["kspace_masked_ifft", "reconstruction_rss"]),
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


if __name__ == "__main__":
    import argparse
    with open("/homes/syli/python/CMRxRecon2024/config/base.json") as f:
        config = json.load(f)
    
    plans_args = argparse.Namespace(**config)
    plans_args.config = config
    train_ds, val_ds, _ = generate_train_val_ds(plans_args)
    for i, j in zip(train_ds, val_ds):
        print(i.keys, j.keys)

