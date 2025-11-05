# Hair-GS: Hair Reconstruction based on 3D Gaussian Splatting
This repository contains the implementation of the paper [Hair Reconstruction based on 3D Gaussian Splatting](https://arxiv.org/abs/2508.08614).

## 🛠️ Installation

### Requirements

- **[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)**: for managing the dependencies.
- **[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)**: any recent version should work, however the code is only tested with CUDA 11.X and 12.X.
- **[MSVC](https://visualstudio.microsoft.com/downloads/)**: required in Windows to compile pytorch CUDA extensions.

### Linux

Create a new conda environment with all the dependencies:
```bash
conda env create -f environment.yml
conda activate hair-gs
```
### Windows 
Create a new conda environment with all the dependencies:
```bash
conda env create -f environment_win.yml
conda activate hair-gs
```
Install [Pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) manually from source:
```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```

### Common Issues

#### CUDA / CUB environment variables missing
CUDA extension builds can fail if the CUDA toolchain or CUB headers are not visible to the compiler. Run the following and try the previous installion steps again. 

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUB_HOME=${CUB_HOME:-$CUDA_HOME/include}
```
## 📂 Dataset

### Data Format
We follow the COLMAP format used in the original 3DGS for data storage. Each strand sample contains the following:

```
sample_id/
├── images/              # RGB images captured from different views
├── masks/              # Binary hair masks corresponding to each view
├── sparse/             # Camera calibration and sparse point cloud (COLMAP format)
├── orientations/       # 2D orientation maps and confidence maps
├── hair_eval_data.npz  # Ground truth strands for evaluation
└── head_reconstruction_data.npz  # Head mesh data (head_verts, scalp_verts)
```

### Data Preparation
In this project, we utilized three distinct datasets for our experiments. The raw data must be preprocessed and converted into the COLMAP format. The folder structure is organized as follows:

```
dataset/
├── raw/              # Raw dataset files
│   ├── usc_hairsalon/
│   ├── cem_yuksel/
│   └── nersemble/
├── parsed/           # Preprocessed COLMAP-format datasets
│   ├── usc_hairsalon/
│   ├── cem_yuksel/
│   └── nersemble/
└── FLAME/            # FLAME model files
    ├── flame2023.pkl
    ├── flame_static_embedding.pkl
    ├── flame_dynamic_embedding.npy
    └── FLAME_masks.pkl
```
### FLAME Model
Download the [FLAME](https://flame.is.tue.mpg.de/index.html) model manually and extract the files to `dataset/FLAME/`. 

### USC-HairSalon Dataset

1. **Download** the [dataset](https://huliwenkidkid.github.io/liwenhu.github.io/)

2. **Extract** the downloaded zip into `dataset/raw/usc_hairsalon`.
   ```bash
   mkdir -p dataset/raw/usc_hairsalon
   unzip hairstyles.zip -d dataset/raw/usc_hairsalon
   ```

3. **Parse the dataset**:
   
   Parse all samples (this may take some time):
   ```bash
   python scripts/parse_usc_hairsalon.py
   ```
   
   Or parse a specific sample (e.g., `00001`):
   ```bash
   python scripts/parse_usc_hairsalon.py -i <sample_id>
   ``` 

### Cem-Yuksel Dataset

Download and parse the 4 hair samples from [Cem-Yuksel](http://cemyuksel.com/research/hairmodels/):

```bash
python scripts/download_parse_cy.py
```

This will automatically download and process the dataset. Parsed dataset is saved into `dataset/raw/cem_yuksel`.

### NeRSemble Dataset [Work in Progress]

1. **Request access** by submitting the form from [NeRSemble](https://github.com/tobias-kirschstein/nersemble-data).

2. **Download and extract the data** into `dataset/raw/nersemble`.

3. **Parse the raw data**:
    
    The entire dataset (this may take some time):
    ```bash
    python scripts/parse_nersemble.py
    ``` 

    Or a specific sample  (e.g., 050)
    ```bash
    python scripts/parse_nersemble.py -i <sample_id>
    ``` 

## 🚀 Usage

### Running the Full Pipeline

The training and evaluation information will be logged to **TensorBoard** by default. To use **Weights & Biases**, add the `--logger wandb` option.

#### Stage I: initialize and optimize GaussianModel

```bash
python train.py -s dataset/parsed/<dataset>/<sample_id>
```

#### Stage II: generate HairGaussianModel from GaussianModel

```bash
python merge.py -s dataset/parsed/<dataset>/<sample_id> -m output/<dataset>/<sample_id>
```

**Note**: The `-m` (model_path) argument points to the directory under `output/` where the Stage I results are saved.

#### Stage III: optimize HairGaussianModel
```bash
python train.py -s dataset/parsed/<dataset>/<sample_id> -m output/<dataset>/<sample_id>
```

### Evaluation

Evaluate the trained model:

```bash
python eval.py -s <path_to_GT> -p <path_to_final_ply> -t <method_type>
```

### Visualization

**During Training**: Use the interactive viewer (network GUI) provided by 3DGS. For more information, see [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).

**After Training**: Convert the final HairGaussianModel to a format that is viewable in MeshLab/Blender:

```bash
python scripts/convert_output.py -i <path_to_final_ply>
```

**Note**: Edges will be converted into triangle polygons with zero area for visualization. The same script can also be used to convert into ply polyline if **--edge** option is added.

### Helper Script

For convenience, we provide a script that runs all three stages given the dataset and sample id:

```bash
./run_full_pipeline_single.sh <dataset> <sample_id>
```

Example:
```bash
./run_full_pipeline_single.sh usc_hairsalon 00001
```



## 📝 Citation

If you find this project useful, please consider giving a ⭐ and citing us:

```bibtex
@inproceedings{Pan_2025_BMVC,
author    = {Yimin Pan and Matthias Nießner and Tobias Kirschstein},
title     = {Hair Strand Reconstruction based on 3D Gaussian Splatting},
booktitle = {36th British Machine Vision Conference 2025, {BMVC} 2025, Sheffield, UK, November 24-27, 2025},
publisher = {BMVA},
year      = {2025},
url       = {https://bmva-archive.org.uk/bmvc/2025/papers/Paper_1220/paper.pdf}
}
```

## 📄 License

This project contains code under multiple licenses:

- **Parts derived from Gaussian-Splatting**  
  Copyright © Inria and MPII  
  Licensed under the [Gaussian-Splatting License](./LICENSE-GAUSSIAN-SPLATTING).  
  These parts are restricted to **non-commercial research use only**.

- **Original contributions** (not derived from Gaussian-Splatting)  
  Released under the [MIT License](./LICENSE).  
  This allows free use, including commercial use, of those components.

⚠️ **Important**: When using or redistributing the combined project, the most restrictive license (Gaussian-Splatting) applies. The overall project is therefore **non-commercial**.
