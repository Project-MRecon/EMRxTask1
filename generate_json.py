import json
import os
from pathlib import Path


root = Path("/homes/syli/dataset/MultiCoil")
sequence_list = ["Cine"]
acceleration_methods, acceleration_factors = ["Uniform"], [4]


def generate_list(data_path: Path, data_list: list):
    img_path = data_path / "FullSample"
    mask_path = data_path / "Mask_Task1"
    for case in img_path.iterdir():
        case_mask_path = mask_path / case.name
        for file in case.iterdir():
            if "cine_lax" not in file.name:
                continue
            for method in acceleration_methods:
                for factor in acceleration_factors:
                    single_sample = {}
                    single_sample["kspace"] = str(file)
                    image_file_name = file.name
                    mask_file_name = image_file_name.replace(".mat", f"_mask_{method}{factor}.mat")
                    mask_file = case_mask_path / mask_file_name
                    single_sample["mask"] = str(mask_file)
                    data_list.append(single_sample)

train_list, val_list = [], []
for sequence in sequence_list:
    train_path  = root / sequence / "TrainingSet"
    generate_list(train_path, train_list)


d = json.dumps(train_list)
with open(root / "Cine" / "train_list.json", "w") as f:
    f.write(d)
