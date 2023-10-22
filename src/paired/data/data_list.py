import shutil
from pathlib import Path

import joblib
from torch.utils.data import Dataset


class DataList(Dataset):
    def __init__(self, root, transforms=None):
        super().__init__()

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.file_paths = list(self.root.glob("*.joblib"))

        self.transforms = transforms

    def __getitem__(self, index):
        path = self.file_paths[index]
        data = joblib.load(path)

        if self.transforms is not None:
            return self.transforms(data)
        else:
            return data

    def __len__(self):
        return len(self.file_paths)

    def add(self, data):
        new_i = len(self)
        new_path = self.root / f"{new_i}.joblib"
        joblib.dump(data, new_path)
        self.file_paths.append(new_path)

    def clear(self):
        shutil.rmtree(self.root)
