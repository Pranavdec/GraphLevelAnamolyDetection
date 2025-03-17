# Graph Level Anomaly Detection

This repository contains the implementation of a graph-level anomaly detection model using contrastive learning, as described in the paper:

**Title:** Deep Graph Level Anomaly Detection with Contrastive Learning  
**Reference Paper:**  
@article{luo2022deep,  
title={Deep graph level anomaly detection with contrastive learning},  
author={Luo, Xuexiong and Wu, Jia and Yang, Jian and Xue, Shan and Peng, Hao and Zhou, Chuan and Chen, Hongyang and Li, Zhao and Sheng, Quan Z},  
journal={Scientific Reports},  
volume={12},  
number={1},  
pages={19867},  
year={2022},  
publisher={Nature Publishing Group UK London}  
}

---

## Installation Steps

Follow these steps to set up the environment and install the required dependencies:

---

### 1. **Create a Python Environment**
The code is tested on **Python 3.10.16**. Ensure you use this version when creating the environment.  

```bash
python3.10 -m venv env
```

- **Activate the Environment**:
  - On Linux/Mac:
    ```bash
    source env/bin/activate
    ```
  - On Windows:
    ```bash
    env\Scripts\activate
    ```

---

### 2. **Install PyTorch**
The code uses **PyTorch 2.4.0** (also compatible with PyTorch 2.6). Install the appropriate version for your system:

- **For Linux**:
  ```bash
  pip install torch==2.4.0
  ```

- **For Windows**:
  Visit the official PyTorch installation page for the correct command:  
  [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

  Example command for Windows:
  ```bash
  pip install torch==2.4.0 torchvision==0.15.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
  ```

---

### 3. **Install Other Dependencies**
Install the remaining dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

### 4. **Verify Installation**
Ensure all dependencies are installed correctly by running:

```bash
pip list
```

Check for the following key packages:
- `torch==2.4.0` (or `torch==2.6.0`)
- Other packages listed in `requirements.txt`.

---

### Troubleshooting
- **PyTorch Installation Issues**:  
  If you encounter issues installing PyTorch, refer to the official PyTorch installation guide:  
  [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

- **Environment Issues**:  
  If the environment setup fails, ensure you are using **Python 3.10.16** and try recreating the environment.

---

## Datasets

The program is designed to work on graph datasets with 2 classes, where the class with fewer records (class 1) is considered the anomaly class. A few datasets are provided in the datasets folder for convenience. Additional datasets can be downloaded from:

https://chrsmrrs.github.io/datasets/docs/datasets/

---

## Architecture

The model architecture consists of the following components:

1. **Graph Convolutional Encoder**:  
   A graph convolutional network (GCN) is used to encode the input graph and nodes with features.

2. **Perturbed Version for Contrastive Learning**:  
   A perturbed version of the graph is created for contrastive learning.

3. **Reconstruction and Anomaly Detection**:  
   The reconstructed graph is passed through the same GCN encoder. Reconstruction loss is used to detect anomalies.

---

## Results

Below are the results of experiments conducted on a few datasets:

| Dataset Name       | GCN with Contrastive Learning | GCN without Contrastive Learning |
|--------------------|-------------------------------|----------------------------------|
| Dataset A          | X.XX%                         | X.XX%                            |
| Dataset B          | X.XX%                         | X.XX%                            |
| Dataset C          | X.XX%                         | X.XX%                            |

---

## Arguments

The following arguments can be passed to the `main.py` script:

- `--datadir`: Directory where the dataset is located (default: `dataset`).
- `--DS`: Dataset name (default: `Tox21_HSE`).
- `--max-nodes`: Maximum number of nodes (default: `0`, meaning no limit).
- `--num_epochs`: Total number of epochs (default: `100`).
- `--batch-size`: Batch size (default: `2000`).
- `--hidden-dim`: Hidden dimension size (default: `256`).
- `--output-dim`: Output dimension size (default: `128`).
- `--num-gc-layers`: Number of graph convolution layers (default: `2`).
- `--nobn`: Disable batch normalization (default: `True`).
- `--dropout`: Dropout rate (default: `0.1`).
- `--lr`: Learning rate (default: `0.00001`).
- `--nobias`: Disable bias (default: `True`).
- `--feature`: Node feature type (default: `deg-num`).
- `--seed`: Random seed (default: `2`).
- `--contrasive_lg`: Enable contrastive learning (default: `True`).
- `--patience`: Patience for learning rate scheduler (default: `5`).
- `--early_stopping_patience`: Patience for early stopping (default: `10`).
- `--threshold_lr`: Threshold for learning rate scheduler (default: `1e-7`).

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## Contact

For questions or feedback, please contact Pallerla Pranav at pallerlapranavdec27@gmail.com.