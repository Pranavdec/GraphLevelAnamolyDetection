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

### Datasets Description

The datasets used in the experiments have the following characteristics:

| Dataset Name | Total Graphs | Training Graphs | Testing Graphs | Anomaly Graphs in Testing | Non-Anomaly Graphs in Testing |
|--------------|--------------|-----------------|----------------|---------------------------|------------------------------|
| DHFR         | 756          | 368             | 388            | 295                       | 93                           |
| COX2         | 467          | 289             | 178            | 102                       | 76                           |
| BZR          | 405          | 255             | 150            | 86                        | 64                           |
| AIDS         | 2000         | 1285            | 715            | 400                       | 315                          |

---

## Architecture

The model architecture consists of the following components:

1. **Graph Convolutional Encoder**:  
   A graph convolutional network (GCN) is used to encode the input graph and nodes with features with Resuidal layers.

2. **Perturbed Version for Contrastive Learning**:  
   A perturbed version of the graph is created for contrastive learning.

3. **Reconstruction and Anomaly Detection**:  
   The reconstructed graph is passed through the same GCN encoder. Reconstruction loss is used to detect anomalies.

---
## Usage
Train the model with:
```bash
python main.py --datadir dataset --DS BZR --batch-size 2000 --hidden-dim 256 --output-dim 128 --num_epochs 100
```

### Key Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--DS` | Dataset name | `BZR` |
| `--hidden-dim` | Hidden layer size | `256` |
| `--output-dim` | Output embedding size | `128` |
| `--num-gc-layers` | Number of GCN layers | `2` |
| `--contrasive_lg` | Enable contrastive learning | `True` |
| `--feature` | Node features (`deg-num`/`default`) | `deg-num` |

---

# Results and Observations  

## Experimental Setup  
Experiments were conducted on **4 datasets** (BZR, COX2, DHFR, AIDS) with varying graph sizes and anomaly ratios.  
- **Evaluation Metrics**: AUC ROC (Area Under ROC Curve) and AUC PR (Area Under Precision-Recall Curve).  
- **Baseline Models**:  
  - **GCN with Contrastive Learning**: Combines graph convolutional layers with contrastive loss.  
  - **GCN without Contrastive Learning**: Standard GCN with reconstruction loss only.  
- **Hyperparameters**: Tuned individually for each dataset (hidden dimension, GCN layers, output dimension).  

---

## Results  

### Model Configuration 1: ReLU Activation, Adam Optimizer, No Residual Connections  
#### AUC ROC Scores  
| Dataset | GCN **with** Contrastive Learning | GCN **without** Contrastive Learning |  
|---------|------------------------------------|---------------------------------------|  
| BZR     | 0.62 (128, 2, 128)                | 0.62 (128, 2, 128)                   |  
| COX2    | 0.57 (128, 4, 128)                | 0.57 (128, 4, 128)                   |  
| DHFR    | 0.56 (256, 3, 128)                | 0.56 (256, 3, 128)                   |  

#### AUC PR Scores  
| Dataset | GCN **with** Contrastive Learning | GCN **without** Contrastive Learning |  
|---------|------------------------------------|---------------------------------------|  
| BZR     | 0.68                               | 0.68                                  |  
| COX2    | 0.66                               | 0.66                                  |  
| DHFR    | 0.80                               | 0.80                                  |  

---

### Model Configuration 2: LeakyReLU Activation, Adam Optimizer, Residual Connections  
#### AUC ROC Scores  
| Dataset | GCN **with** Contrastive Learning | GCN **without** Contrastive Learning |  
|---------|------------------------------------|---------------------------------------|  
| BZR     | 0.69 (512, 4, 256)                | 0.69 (512, 4, 256)                   |  
| COX2    | 0.55 (64, 3, 256)                 | 0.55 (64, 3, 256)                    |  
| DHFR    | 0.50 (64, 3, 128)                 | 0.50 (64, 3, 128)                    |  
| AIDS    | 0.67 (512, 4, 128)                | 0.67 (512, 4, 128)                   |  

#### AUC PR Scores  
| Dataset | GCN **with** Contrastive Learning | GCN **without** Contrastive Learning |  
|---------|------------------------------------|---------------------------------------|  
| BZR     | 0.76                               | 0.76                                  |  
| COX2    | 0.64                               | 0.64                                  |  
| DHFR    | 0.77                               | 0.77                                  |  
| AIDS    | 0.84                               | 0.84                                  |  

---

## Key Observations  

### 1. **Performance of Contrastive Learning**  
- **No Significant Improvement**: Contrastive learning **did not enhance performance** in anomaly detection for any dataset. Both models (with/without contrastive learning) achieved **identical AUC ROC and AUC PR scores** across all configurations.  
- **Dominant Reconstruction Loss**: The lack of improvement may stem from the overwhelming contribution of the reconstruction loss (encoder-decoder architecture) compared to the contrastive loss, diminishing its impact during training.  

### 2. **Hyperparameter Sensitivity**  
- **Critical Hyperparameters**: Performance heavily depended on combinations of **hidden dimension**, **number of GCN layers**, and **output dimension** (see values in parentheses).  
- **Dataset-Specific Tuning**: Optimal hyperparameters varied across datasets (e.g., BZR required larger hidden dimensions with residual connections).  

### 3. **Architectural Robustness**  
- **Addressing Vanishing Gradients**:  
  - The ReLU-based model (no residuals) faced **vanishing gradients** on datasets like AIDS, Tox21_HSE, and Tox21_p53.  
  - Switching to **LeakyReLU activations** and **residual connections** stabilized training and improved convergence.  
- **Learning Rate Strategy**: A low initial learning rate with reduction based on validation metrics (AUC ROC, AUC PR, loss) prevented overfitting and enhanced stability.  

### 4. **Dataset Generalization**  
- **Scalability**: The model showed robustness across datasets of varying sizes and anomaly ratios.  
- **Limitation**: Contrastive learning may still be beneficial for datasets with **subtler distinctions** between normal and anomalous graphs.  

### 5. **Future Directions**  
- **Larger Datasets**: Testing on datasets with **more graphs** could better validate the utility of contrastive learning.  
- **Loss Balancing**: Adjusting the weight of contrastive loss relative to reconstruction loss might amplify its impact.  

--- 

## Conclusion  
While contrastive learning did not improve performance in these experiments, architectural modifications (residual connections, LeakyReLU) effectively addressed training challenges like vanishing gradients. Further exploration on larger datasets and loss-balancing strategies is recommended to fully leverage contrastive learning.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## Contact

For questions or feedback, please contact Pallerla Pranav at pallerlapranavdec27@gmail.com.