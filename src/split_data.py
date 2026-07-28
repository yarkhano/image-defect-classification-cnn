import os, shutil, random

random.seed(42)
RAW = "data/raw"
PROC = "data/processed"
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

for split in SPLITS:
    for cls in ["defective", "non_defective"]:
        os.makedirs(f"{PROC}/{split}/{cls}", exist_ok=True)

for cls in ["defective", "non_defective"]:
    files = os.listdir(f"{RAW}/{cls}")
    random.shuffle(files)
    n = len(files)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    splits_files = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }
    for split, split_files in splits_files.items():
        for f in split_files:
            shutil.copy(f"{RAW}/{cls}/{f}", f"{PROC}/{split}/{cls}/{f}")

print("Split complete.")