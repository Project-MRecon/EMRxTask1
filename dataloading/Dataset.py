import os
import sys
import numpy as np
import random
from pathlib import Path
from monai.data import CacheDataset
from monai.transforms import (
    LoadImaged, 
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ToTensord,)
from monai.utils import set_determinism
from typing import Union, List, Sequence, Callable


class CMRxRecon2024Dataset(CacheDataset):
    """
    The Dataset for MICCAI 2024 CMRxRecon2024 Challenge. Supports two tasks (task1 and task2) with different mask selections.
    This dataset class inherits from MONAI's CacheDataset to efficiently handle large medical image datasets with caching capabilities.

    Args:
        root_dir: The path to the directory containing the dataset's JSON configuration file.
        task: An integer specifying the task (1 or 2).
        mask_type: An integer or list of integers specifying the mask type(s) to use for the dataset. Default is None.
        transform: A callable or sequence of callables that define the transformations to apply to the data.
        section: A string specifying the dataset section to use: 'training', 'validation', or 'test'. Default is 'training'.
        val_frac: A float representing the fraction of the dataset to use for validation. Default is 0.2.
        seed: An integer for seeding random number generation to ensure reproducibility. Default is 0.
        k_fold: An integer specifying the number of folds for cross-validation. Default is 5.
        fold_id: An integer specifying which fold to use for the dataset split. Default is 0.
        cache_num: The number of items to cache. Default is sys.maxsize.
        cache_rate: The fraction of the dataset to cache. Default is 1.0.
        num_workers: The number of worker threads to use for caching. Default is 1.
        training_mode: An integer specifying the training mode. Default is 1.
        progress: A boolean indicating whether to display a progress bar during data processing. Default is True.
        copy_cache: A boolean indicating whether to deepcopy the cached content before applying random transforms. Default is True.
        as_contiguous: A boolean indicating whether to convert cached data to contiguous arrays/tensors. Default is True.
        runtime_cache: A boolean indicating whether to compute the cache at runtime. Default is False.
        categories: A list of strings representing the categories to include in the dataset. Default is None.

    Raises:
        ValueError: If the root directory is not a valid directory.
        ValueError: If the task is not 1 or 2.
        ValueError: If the section is not 'training', 'validation', or 'test'.

    Example::

        transform = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image"]),
            ]
        )

        dataset = CMRxRecon2024Dataset(
            root_dir="./data",
            task=1,
            mask_type=4,
            section="training",
            val_frac=0.2,
            transform=transform,
            training_mode=1,
            seed=42,
            categories=["a", "b", "c"]
        )

        print(dataset[0]["image"])

    """
    def __init__(
        self,
        root_dir: Union[Path, str],
        task: int,
        mask_type: Union[int, List[int]] = None,
        transform: Union[Sequence[Callable], Callable] = (),
        section: str = "training",
        val_frac: float = 0.2,
        seed: int = 0,
        k_fold: int = 5,
        fold_id: int = 0,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 1,
        training_mode: int = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        runtime_cache: bool = False,
        categories: List[str] = None
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise ValueError(f"Root directory {root_dir} is not a valid directory.")
        self.task = task
        self.mask_type = mask_type
        self.section = section
        self.val_frac = val_frac
        self.seed = seed
        self.k_fold = k_fold
        self.fold_id = fold_id
        self.training_mode = training_mode
        self.categories = categories or []
        if task not in [1, 2]:
            raise ValueError("Supported tasks are 1 or 2.")
        if section not in ["training", "validation", "test"]:
            raise ValueError("Section must be one of ['training', 'validation', 'test'].")
        set_determinism(seed=seed)
        # 用一个随机数生成器保证掩码数生成的可重复性
        self.rng = np.random.RandomState(seed)
        # 加载数据列表
        datalist = self._generate_data_list(self.root_dir / f"task{task}", section)
        if transform == ():
            transform = LoadImaged(keys=["image"], reader=)
        super().__init__(
            data=datalist,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
            runtime_cache=runtime_cache
        )

    def _generate_data_list(self, dataset_dir: Path, section: str):
        # 支持 .nii.gz 和 .mat 文件格式
        all_files = sorted(list(dataset_dir.glob(".nii.gz")) + list(dataset_dir.glob(".mat")))
        if section == "test":
            return [{"image": str(img), "category": self._extract_category(img)} for img in all_files]
        random.seed(self.seed)
        random.shuffle(all_files)
        # 按类别分组，并进行分层拆分
        category_groups = {category: [] for category in self.categories}
        for img in all_files:
            category = self._extract_category(img)
            if category in category_groups:
                category_groups[category].append(img)
        training_files = []
        validation_files = []
        for category, files in category_groups.items():
            split_idx = int(len(files) * (1 - self.val_frac))
            if section == "training":
                training_files.extend(files[:split_idx])
            else:
                validation_files.extend(files[split_idx:])
        selected_files = training_files if section == "training" else validation_files
        # 数据集上采样
        upsampled_files = self._upsample_data(selected_files)
        filtered_files = []
        if self.task == 1:
            for img_file in upsampled_files:
                if self.mask_type in [4, 8, 10]:
                    mask_value = self.mask_type
                    filtered_files.append({
                        "image": str(img_file),
                        "mask": mask_value
                    })
        elif self.task == 2:
            mask_choices = list(range(4, 25, 2))
            for img_file in upsampled_files:
                mask_value = self.rng.choice(mask_choices)
                filtered_files.append({
                    "image": str(img_file),
                    "mask": mask_value
                })
        return filtered_files

    def _extract_category(self, filepath: Path) -> str:
        # 提取文件名中的类别
        for category in self.categories:
            if category in filepath.stem:
                return category
        return "Unknown"

    def _upsample_data(self, file_list) -> List[Path]:
        # 上采样至类别最多的那类
        category_counts = {category: 0 for category in self.categories}
        categorized_files = {category: [] for category in self.categories}
        for file in file_list:
            category = self._extract_category(file)
            if category in categorized_files:
                category_counts[category] += 1
                categorized_files[category].append(file)
        max_count = max(category_counts.values())
        upsampled_files = []
        for files in categorized_files.values():
            if len(files) == 0:
                continue
            upsampled_files.extend(files)
            additional_files_needed = max_count - len(files)
            if additional_files_needed > 0:
                upsampled_files.extend(random.choices(files, k=additional_files_needed))
        return upsampled_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def get_properties(self):
        return {
            "root_dir": self.root_dir,
            "task": self.task,
            "mask_type": self.mask_type,
            "section": self.section,
            "val_frac": self.val_frac,
            "seed": self.seed,
            "k_fold": self.k_fold,
            "fold_id": self.fold_id,
            "training_mode": self.training_mode,
            "categories": self.categories
        }
    
if __name__ == "__main__":
    transform = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image"]),
            ]
        )

    dataset = CMRxRecon2024Dataset(
        root_dir="",
        task=1,
        mask_type=4,
        section="training",
        val_frac=0.2,
        transform=transform,
        training_mode=1,
        seed=42,
        categories=["a", "b", "c"]
    )
{"masked_kspace", "mask", "image", "": 16}