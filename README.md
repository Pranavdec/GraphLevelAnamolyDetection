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
## Usage

To run the anomaly detection model, execute the `main.py` script with the desired arguments. The script will train the model on the specified dataset and evaluate its performance.

```bash
python main.py --datadir dataset --DS BZR --max-nodes 0 --num_epochs 100 --batch-size 2000 --hidden-dim 256 --output-dim 128 --num-gc-layers 2 --nobn True --dropout 0.1 --lr 0.00001 --nobias True --feature deg-num --seed 2 --contrasive_lg True --patience 5 --early_stopping_patience 10 --threshold_lr 1e-7
```
---

## Arguments

The following arguments can be passed to the `main.py` script:

- `--datadir`: Directory where the dataset is located (default: `dataset`).
- `--DS`: Dataset name (default: `BZR`).
- `--max-nodes`: Maximum number of nodes (default: `0`, meaning no limit).
- `--num_epochs`: Total number of epochs (default: `100`).
- `--batch-size`: Batch size (default: `512`).
- `--hidden-dim`: Hidden dimension size (default: `256`).
- `--output-dim`: Output dimension size (default: `128`).
- `--num-gc-layers`: Number of graph convolution layers (default: `2`).
- `--nobn`: Disable batch normalization (default: `True`).
- `--dropout`: Dropout rate (default: `0.1`).
- `--lr`: Learning rate (default: `0.00001`).
- `--nobias`: Disable bias (default: `True`).
- `--feature`: Node feature type (default: `deg-num`). Options: `deg-num`, `default`.
- `--seed`: Random seed (default: `2`).
- `--contrasive_lg`: Enable contrastive learning (default: `True`).
- `--patience`: Patience for learning rate scheduler (default: `5`).
- `--early_stopping_patience`: Patience for early stopping (default: `10`).
- `--threshold_lr`: Threshold for learning rate scheduler (default: `1e-7`).
- `--anamoly-label`: Anomaly Class label (default: `1`).

---

## Results

Below are the results of experiments conducted on a few datasets using **Graph Convolutional Networks (GCN)** with and without **Contrastive Learning**. The evaluation metrics used are **AUC ROC** (Area Under the Receiver Operating Characteristic Curve) and **AUC PR** (Area Under the Precision-Recall Curve). The values in parentheses represent the **hidden embedding size** and the **number of GCN layers** in the encoder, which were found to yield the best metrics after testing multiple hyperparameter combinations.

---

### Results Table

#### AUC ROC Scores

| Dataset Name | GCN with Contrastive Learning (AUC ROC) | GCN without Contrastive Learning (AUC ROC) |
|--------------|-----------------------------------------|--------------------------------------------|
| BZR          | 0.62 (128, 2)                          | 0.62 (128, 2)                              |
| COX2         | 0.57 (128, 4)                          | 0.57 (128, 4)                              |
| DHFR         | 0.56 (256, 3)                          | 0.56 (256, 3)                              |

#### AUC PR Scores

| Dataset Name | GCN with Contrastive Learning (AUC PR) | GCN without Contrastive Learning (AUC PR) |
|--------------|----------------------------------------|-------------------------------------------|
| BZR          | 0.68                                   | 0.68                                      |
| COX2         | 0.66                                   | 0.66                                      |
| DHFR         | 0.80                                   | 0.80                                      |

---

### Datasets Description

The datasets used in the experiments have the following characteristics:

| Dataset Name | Total Graphs | Training Graphs | Testing Graphs | Anomaly Graphs in Testing | Non-Anomaly Graphs in Testing |
|--------------|--------------|-----------------|----------------|---------------------------|------------------------------|
| DHFR         | 756          | 368             | 388            | 295                       | 93                           |
| COX2         | 467          | 289             | 178            | 102                       | 76                           |
| BZR          | 405          | 255             | 150            | 86                        | 64                           |

---

### Observations

1. **Performance Comparison**:
   - The results indicate that **GCN with Contrastive Learning** performs similarly to **GCN without Contrastive Learning** across all datasets in terms of both **AUC ROC** and **AUC PR** scores.
   - This suggests that, for these specific datasets, the addition of contrastive learning does not significantly enhance anomaly detection performance.

2. **Hyperparameter Sensitivity**:
   - The best results were achieved with specific combinations of **hidden embedding size** and **number of GCN layers**, as indicated in the parentheses in the results table.

3. **Learning Rate Reduction**:
   - A learning rate reduction strategy was applied during training. This helped stabilize the training process and prevent overfitting.
    - The model was trained with a low initial learning rate, which was reduced by a factor of 10 based on validaiton AUC ROC AUC PR and loss values.

4. **Model Robustness**:
   - The model demonstrates consistent performance across datasets of varying sizes and anomaly distributions, indicating robustness to different data characteristics.

5. **Contrastive Learning Impact**:
   - While contrastive learning did not improve performance in these experiments, it may still be beneficial for other datasets or tasks where the separation between normal and anomalous graphs is less distinct.

---

### Conclusion

The experiments demonstrate that the proposed GCN-based anomaly detection model performs well across multiple datasets. However, the addition of contrastive learning did not yield significant improvements in this case. Key findings include:

- The model is robust to variations in dataset size and anomaly distribution.
- Hyperparameters such as **hidden embedding size** and **number of GCN layers** play a critical role in achieving optimal performance.
- Contrastive learning may not always enhance performance, and its effectiveness could depend on the specific characteristics of the dataset.

**Future Work**:
- Explore alternative contrastive learning strategies or architectures.
- Evaluate the model on datasets with more challenging anomaly detection scenarios to further assess its capabilities.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## Contact

For questions or feedback, please contact Pallerla Pranav at pallerlapranavdec27@gmail.com.