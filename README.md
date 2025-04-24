# VQA Model Comparison: QFormer, Cross Attention & Concat

This repository presents a comparative study of three deep learning architectures for Visual Question Answering (VQA) on a binary yes/no classification dataset.

## 📌 Project Overview

The goal is to benchmark the following models:
- **QFormer**: A multi-task transformer with cross-modal attention and diverse objectives.
- **Cross Attention**: A model that directly applies cross-attention between visual and textual embeddings.
- **Concat Model**: A simpler baseline concatenating visual and text features before classification.

Each model supports two encoder types:
- **CLIP**: For both image and text.
- **BERT + ViT**: BERT for text and ViT-CLIP for image encoding.

## 📂 Dataset

The dataset used for this project was sourced from the following repository:
[Visual_Question_Answering by dinhquy-nguyen-1704](https://github.com/dinhquy-nguyen-1704/Visual_Question_Answering)


- **Train**: 7,846 samples  
- **Validation**: 1,952 samples  
- **Test**: 2,022 samples

## 🧾 Directory Structure

```
QFormer/
├── configs/               # JSON configuration files per model
├── data/vqa/              # Dataset and resized image files
├── logs/                  # Training logs per model/encoder
├── model/                 # Core model architectures
├── my_datasets/           # Data loading utilities
├── my_lightning_model/    # PyTorch Lightning wrappers
├── plots/                 # Generated result plots
├── results/               # Metrics and test outputs
├── saved_models/          # Saved checkpoints
├── scripts/               # Utility scripts for running and plotting
├── tests/                 # Unit tests
├── trainers/              # Main training logic
└── requirements.txt       # Dependencies
```

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/vqa-model-comparison.git
cd vqa-model-comparison
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

### Train a Single Model

```bash
python trainers/trainer.py \
    --model_name qformer \
    --use_clip_for_text True \
    --gpu_device 0
```

### Run All Experiments

```bash
python scripts/run.py --gpus 0,1 --models all --encoders all
```

### Generate Comparison Plots

```bash
python scripts/make_plots.py --results_dir results --output_dir plots
```

## 📈 Results and Visualizations

- **Accuracy Comparison**: Plots for training and validation accuracy.
- **Test Accuracy**: Horizontal bar chart for model performance.
- **QFormer Loss Components**: Breakdown of ITC, ITM, IGT, and Answer loss.

## 🧠 Model Details

### QFormer
Multi-objective model using:
- Answer classification
- Image-Text Contrastive (ITC)
- Image-Text Matching (ITM)
- Image-Grounded Text (IGT)

### Cross Attention
Focuses on cross-attention mechanisms between image and question embeddings.

### Concat Model
Simple concatenation of features followed by classification layers.

## 📚 Acknowledgements

- [CLIP](https://github.com/openai/CLIP)
- [BERT](https://huggingface.co/bert-base-uncased)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Transformers](https://huggingface.co/transformers/)

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
