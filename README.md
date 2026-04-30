# KD-for-Social-Media-Misinformation-Detection

## Description

This repository provides the implementation code and reproducibility materials for a multi-teacher knowledge distillation framework for social media misinformation detection.

The project is primarily evaluated on the Weibo21 dataset. The framework integrates multiple pre-trained teacher models, such as BERT and RoBERTa, and distills their knowledge into a lighter student model, such as Chinese-RoBERTa. During training, an agent-based dynamic teacher-weighting strategy is used to evaluate and fuse the logits of different teacher models. The goal is to improve fake news detection performance while maintaining efficient inference.

## Core Features

### Multi-Teacher Knowledge Distillation

The framework integrates multiple teacher models to provide richer soft-label supervision for the student model. In this project, BERT and RoBERTa are used as teacher models.

### Agent-Based Dynamic Teacher Weighting

A multilayer perceptron-based agent is used as a dynamic teacher-weight allocator. It receives information from the student model and teacher models, and dynamically assigns importance weights to different teacher models during the distillation process.

### Five-Fold Cross-Validation

The repository includes a five-fold cross-validation pipeline to evaluate the stability and robustness of the proposed method.

### Ablation Experiments

The code supports several ablation settings, including:

- removing the dynamic agent module;
- using only unprivileged input;
- comparing single-teacher baselines;
- evaluating the contribution of different knowledge distillation settings.

### Sensitivity Analysis

The repository includes scripts for analyzing the sensitivity of important knowledge distillation hyperparameters, such as temperature and alpha.

### GPU Training Optimization

The implementation supports GPU training and includes optimization strategies such as automatic mixed precision training and TF32 matrix multiplication support when compatible hardware is available.

## Dataset Information

### Dataset Source

This project uses the Weibo21 dataset for fake news detection.

The original dataset source is the MDFEND-Weibo21 repository:

```text
https://github.com/kennqiang/MDFEND-Weibo21
```

The original repository provides the official source and description of the Weibo21 dataset. In this project, the processed dataset files used for experiments are provided in CSV format to support human and machine readability.

### Processed Data Files

The processed dataset files used in this project are provided in CSV format.

If the repository uses pre-split files, the expected files are:

```text
train.CSV
val.CSV
test.CSV
```

If the repository uses a single merged data file, the expected file is:

```text
weibo21.csv
```

Please keep the original column names unchanged when running the code.

### Data Format

The processed data files should contain the text content and label fields required by the training scripts.

The key fields used by the scripts include:

```text
news_content: textual content of the news item or post
label: fake/real label used for supervised training and evaluation
```

If comment or social-context information is used in the privileged-input setting, the corresponding comment-related fields should also be retained in the CSV file.

### Data Access

The processed Weibo21 data files are provided in CSV format in this repository and/or through the supplementary materials associated with the manuscript.

A temporary data-sharing mirror is also available at:

```text
https://pan.quark.cn/s/57ef1069c4cb?pwd=Nw3n
Extraction code: Nw3n
```

This link is provided as a temporary mirror for data access. For long-term reproducibility, the processed CSV files should also be provided as supplementary files, GitHub Release assets, or a Zenodo archive.

## Code Information

This repository contains the full implementation pipeline, including teacher model training, student model training, knowledge distillation, validation, evaluation, ablation studies, and sensitivity analysis.

### Main Files

#### `train_teachers_init.py`

Initializes and trains the pre-trained teacher models, including BERT and RoBERTa. The teacher models are trained using cross-entropy loss, and the best-performing teacher checkpoints are saved for subsequent distillation.

#### `run_main_ours.py`

Runs the main proposed multi-teacher knowledge distillation method with the agent-based dynamic teacher-weighting strategy.

#### `run_5fold_final.py`

Runs the five-fold cross-validation experiment for the proposed method. This script performs student training, dynamic teacher-weight allocation, KL-divergence-based distillation, and validation for each fold.

#### `run_5fold_ablation_no_agent.py`

Runs the ablation experiment without the dynamic agent module. In this setting, the teacher logits are fused using a static averaging strategy.

#### `run_5fold_ablation_input.py`

Runs the unprivileged-input ablation experiment. The model is forced to use only the news content field, excluding additional social-context features such as comments.

#### `run_single_teacher.py`

Runs the single-teacher baseline experiment. This script evaluates the performance of the student model when guided by only one teacher model, such as BERT or RoBERTa.

#### `run_sensitivity.py`

Runs hyperparameter sensitivity analysis for knowledge distillation parameters, including temperature and alpha.

#### `model.py`

Defines the main teacher and student model architectures, including the modules used for dynamic feature computation and teacher-student learning.

#### `evaluation.py`

Provides evaluation functions for calculating classification metrics, including Macro F1, Accuracy, AUC, Precision, Recall, and class-specific F1 scores.

#### `trainer.py` and `trainer_twstd.py`

Provide training wrappers for batch training, loss computation, optimizer scheduling, and model update logic.

#### `utils.py`

Contains utility functions, including random seed setting, loss functions, and feature aggregation operations.

## Requirements

The experiments were conducted in a Python environment with the following main dependencies:

```text
python == 3.10.19
torch == 2.6.0
transformers == 5.3.0
accelerate == 1.12.0
datasets == 4.6.1
sentence-transformers == 5.2.3
sentencepiece == 0.2.1
numpy == 1.26.4
pandas == 2.3.3
scikit-learn == 1.7.2
scipy == 1.15.3
tqdm == 4.67.1
PyYAML == 6.0.3
matplotlib == 3.10.8
seaborn == 0.13.2
optuna == 4.6.0
protobuf == 3.20.3
requests == 2.32.5
```

A CUDA-compatible GPU is recommended for efficient training. The original experiments were conducted in a CUDA-enabled PyTorch environment.

Install the required packages using:

```bash
pip install -r requirements.txt
```

If CUDA-specific installation is required, please install the PyTorch version that matches the local CUDA environment.

## Usage Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/K497Z/KD-for-Social-Media-Misinformation-Detection.git
cd KD-for-Social-Media-Misinformation-Detection
```

### 2. Prepare the Dataset

If the repository uses pre-split CSV files, place the following files in the project directory or in the data path specified by the scripts:

```text
train.CSV
val.CSV
test.CSV
```

If the repository uses a single merged CSV file, place the file under:

```text
data/weibo21/weibo21.csv
```

Please make sure that the dataset path in the scripts is consistent with the actual file location.

### 3. Train Teacher Models

Run:

```bash
python train_teachers_init.py
```

This step trains the teacher models and saves the best teacher checkpoints for later use in student distillation.

### 4. Run the Proposed Five-Fold Distillation Experiment

Run:

```bash
python run_5fold_final.py
```

This script performs the main five-fold cross-validation experiment using the dynamic multi-teacher knowledge distillation framework.

The results are saved to:

```text
5fold_results_final.json
```

Detailed logs are also written to the corresponding log file generated by the script.

### 5. Run Ablation Experiments

#### Without Dynamic Agent

```bash
python run_5fold_ablation_no_agent.py
```

#### Unprivileged Input Only

```bash
python run_5fold_ablation_input.py
```

### 6. Run Single-Teacher Baselines

```bash
python run_single_teacher.py
```

### 7. Run Sensitivity Analysis

```bash
python run_sensitivity.py
```

## Evaluation Metrics

The project reports multiple evaluation metrics to assess misinformation detection performance.

The main metrics include:

```text
Macro F1
Accuracy
AUC
```

Class-specific metrics include:

```text
Precision for fake and real classes
Recall for fake and real classes
F1 score for fake and real classes
```

## Methodology

The method follows a multi-teacher knowledge distillation framework for social media misinformation detection.

First, multiple teacher models are trained on the Weibo21 dataset. Then, a lighter student model is trained using both hard labels and soft-label supervision from the teacher models. During student training, an agent-based dynamic weighting module assigns different weights to teacher models according to the model outputs and training state.

The final training objective combines supervised classification loss and knowledge distillation loss.

The main training components include:

```text
Cross-entropy loss for supervised classification
KL-divergence loss for teacher-student distillation
Dynamic teacher-weight allocation through the agent module
```

Five-fold cross-validation is used to evaluate the stability of the method. Ablation experiments are conducted to examine the contribution of the dynamic agent module, input settings, and different teacher configurations.

## Output Files

The main experiment produces the following outputs:

```text
5fold_results_final.json
training logs
teacher checkpoint files
student checkpoint files
evaluation results for each fold
```

The exact output file names may vary depending on the script configuration.

## Reproducibility Notes

To reproduce the experiments:

1. Clone this repository.
2. Install the required Python dependencies.
3. Download or prepare the processed Weibo21 CSV data files.
4. Place the data files in the path required by the scripts.
5. Train the teacher models using `train_teachers_init.py`.
6. Run the main five-fold experiment using `run_5fold_final.py`.
7. Use the same random seed settings defined in the scripts.

For improved reproducibility, users should avoid changing the dataset column names, model checkpoint paths, or random seed configuration unless they also update the corresponding script arguments or path settings.

If the local directory structure differs from the default setting, please update the dataset path and checkpoint path in the corresponding scripts before running the experiments.

## Citation

If you use this code or dataset, please cite the associated manuscript and the original MDFEND-Weibo21 dataset source:

```text
https://github.com/kennqiang/MDFEND-Weibo21
```

A formal citation for the associated manuscript should be added here after publication.

## License

This repository is released for research and reproducibility purposes.

Users should follow the license and access conditions of the original Weibo21 dataset and the MDFEND-Weibo21 repository. The processed data files are provided only for research and reproducibility purposes.

## Contact

For questions about the code or reproduced experiments, please open an issue in this repository.
