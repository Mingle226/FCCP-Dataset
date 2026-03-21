# FCCP: Fine-Grained Cloud Phase Recognition Benchmark

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20TGRS%2FJournal-blue)](https://doi.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BaiduNetdisk-green)](https://pan.baidu.com/s/1zHpGDFkdLfCFIFMpQScyMQ?pwd=2026)

This repository hosts the **FCCP (Fine-Grained Cloud Phase)** benchmark dataset, introduced in our paper: **"From Patch to Point: A Distance-Decay Attention Network for Fine-Grained Cloud Phase Recognition"**.

FCCP is the first pixel-level aligned dataset designed to bridge the gap between passive wide-field geostationary imagery (2D) and active narrow-beam vertical profiling labels (1D).

---

## 📅 Dataset Overview

FCCP is constructed by harmonizing observations from **FY-4A AGRI** (passive imager) and **CloudSat/CALIPSO** (active profiling sensors). It focuses on the challenging task of thermodynamic phase recognition, particularly for **Mixed-Phase Clouds**.

- **Total Samples:** ~350,000
- **Time Span:** Oct 10, 2018 – July 10, 2019 (R05 Release)
- **Problem Type:** Multi-class Classification (Patch-to-Point)
- **Data Format:** `.npz` (NumPy Compressed Archive)

### Class Definition & Folder Structure

The dataset is organized into folders representing the physical cloud phase categories.

| Folder Name | Class ID | Semantic Label  | Description                              |
| :---------: | :------: | :-------------- | :--------------------------------------- |
|    **1**    |  **0**   | **Ice Cloud**   | Pure ice crystals                        |
|    **2**    |  **1**   | **Mixed Cloud** | Coexistence of supercooled water and ice |
|    **3**    |  **2**   | **Water Cloud** | Liquid water droplets                    |

> **Note:** The folder names `1`, `2`, `3` correspond to the raw labels from CloudSat. In typical machine learning setups (0-indexed), these map to 0 (Ice), 1 (Mixed), and 2 (Water).

---

## 📁 File Organization

The dataset is distributed as individual `.npz` files sorted into class-specific subdirectories. It does **not** come pre-split into train/val/test sets. Users must perform the split manually (see Usage section).

```text
FCCP_Dataset (data6)/
├── 1/                   # Ice Cloud Samples
│   ├── 2018283_000001.npz
│   ├── 2018283_000002.npz
│   └── ...
├── 2/                   # Mixed Cloud Samples
│   ├── 2018283_0000XX.npz
│   └── ...
└── 3/                   # Water Cloud Samples
    ├── 2018283_0000XX.npz
    └── ...
```

### Data Sample Format

Each `.npz` file contains a single sample with the following keys (example):

- `data`: The input tensor of shape `(14, 28, 28)`.
- `label`: The ground truth label (optional, as the folder indicates the label).

------

## 🚀 Usage

### 1. Prerequisite: Chronological Splitting

To prevent **temporal data leakage** (a critical issue in meteorological time-series analysis), the dataset **must be split chronologically** rather than randomly.

The filenames contain timestamps (e.g., `2018283` represents Year 2018, Day 283). **Recommended Split Strategy:**

1. Collect all file paths.
2. **Sort** them strictly by filename (which aligns with time).
3. Split the sorted list:
   - **Train:** First 70%
   - **Val:** Next 15%
   - **Test:** Final 15%

### 2. PyTorch DataLoader Example

Below is a Python script to load the data, perform the chronological split, and create DataLoaders.

```python
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class FCCPDataset(Dataset):
    def __init__(self, base_dir, mode='train', means=None, stds=None, 
                 files_names=None, ratio=1.0, dataset_path="", device='cpu'):
        
        self.base_dir = os.path.join(base_dir, dataset_path)
        self.device = torch.device(device)
        
        # 1. 简化文件获取逻辑
        if files_names is None:
            all_files = []
            for sub in ["1", "2", "3"]:
                sub_path = os.path.join(self.base_dir, sub)
                if os.path.exists(sub_path):
                    all_files.extend([os.path.join(sub, f) for f in os.listdir(sub_path) if f.endswith('.npz')])
            all_files.sort(key=lambda x: x[2:])
            
            # 2. 统一划分逻辑 (70/15/15)
            n = len(all_files)
            train_idx, valid_idx = int(n * 0.7), int(n * 0.85)
            
            if mode == 'train':
                self.files_names = all_files[:train_idx]
            elif mode == 'valid':
                self.files_names = all_files[train_idx:valid_idx]
            else:
                self.files_names = all_files[valid_idx:]
            
            # 抽样比例缩减
            if ratio < 1.0:
                np.random.shuffle(self.files_names)
                self.files_names = self.files_names[:int(len(self.files_names) * ratio)]
        else:
            self.files_names = files_names

        # 3. 均值标准差：建议直接传入常数，避免在线计算
        self.means = torch.tensor(means).view(14, 1, 1) if means is not None else None
        self.stds = torch.tensor(stds).view(14, 1, 1) if stds is not None else None

    def __len__(self):
        return len(self.files_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.base_dir, self.files_names[idx])
        with np.load(file_path) as data:
            image = torch.from_numpy(data['fy4a_data']).float() # (14, H, W)
            label = torch.from_numpy(data['cloudPhase']).long()

        # 标准化
        if self.means is not None:
            image = (image - self.means) / self.stds

        # 4. 向量化计算亮温差 (BTD)
        # 索引对应关系: btd12_13(11-12), btd7_12(6-11), btd11_12(10-11), btd9_12(8-11), btd5_6(4-5)
        indices_a = [11, 6, 10, 8, 4]
        indices_b = [12, 11, 11, 11, 5]
        btd = image[indices_a] - image[indices_b] # (5, H, W)

        # 5. 中心点通道
        C, H, W = image.shape
        center_channel = torch.zeros((1, H, W))
        center_channel[0, H // 2, W // 2] = 1.0

        # 合并所有通道 (14 + 5 + 1 = 20)
        image = torch.cat([image, btd, center_channel], dim=0)
        
        # 标签处理 (假设原始标签从1开始)
        return image.to(self.device), (label - 1).to(self.device)
```

## 📝 Citation

If you use the FCCP dataset or find our work helpful, please consider citing:

```
@article{yao2026patchtopoint,
  title={From Patch to Point: A Distance-Decay Attention Network for Fine-Grained Cloud Phase Recognition},
  author={Yao, Yiming and Ma, Jianghong and Luo, Chuyao and Li, Xutao and Ye, Yunming},
  journal={IEEE Transactions on Geoscience and Remote Sensing (Submitted)},
  year={2026}
}
```

## 📄 License

This dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**. You are free to share and adapt the material as long as you give appropriate credit.

------

**Contact:** For questions regarding the dataset, please contact Yiming Yao at `yaoyiming@stu.hit.edu.cn`.
