# Multimodal Emotion Recognition

A PyTorch implementation comparing unimodal and multimodal approaches for emotion classification using visual, acoustic, and textual features from the IEMOCAP dataset.

## Overview

This project explores how different feature fusion strategies affect emotion recognition performance. The system classifies four emotional states (anger, sadness, happiness, neutral) by combining:
- **Visual**: ResNet face embeddings 
- **Acoustic**: VGGish audio embeddings
- **Textual**: BERT sentence embeddings

**Key Finding**: Early fusion (62.56% F1) significantly outperformed late fusion (54.31% F1), demonstrating the importance of learning cross-modal feature interactions.

## Dataset

**IEMOCAP** (Interactive Emotional Dyadic Motion Capture)
- 10 actors across 5 dyadic sessions
- Pre-extracted features provided for coursework
- 4-class emotion classification task
- Subject-independent evaluation protocol

**Note**: The IEMOCAP dataset requires permission from USC SAIL Lab. The pre-extracted features used in this project were provided as part of CSCI 535 coursework and are not included in this repository. To access the original dataset:

1. Visit the [USC SAIL IEMOCAP page](https://sail.usc.edu/iemocap/)
2. Submit a request form with your research/academic affiliation
3. Follow the licensing agreement procedures

For questions about the specific feature extraction pipeline used in this coursework, please contact the course instructors.

## Implementation Details

### Architecture
```python
class ModalityClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
```

### Data Processing
- **Temporal Pooling**: Mean pooling to reduce temporal dimensions
- **Normalization**: StandardScaler for feature stability
- **Cross-Validation**: 5-fold subject-independent splits
- **Missing Data**: Robust handling of corrupted/missing feature files

### Fusion Strategies

**Early Fusion** (Feature-level)
```python
# Concatenate normalized features before training
X_combined = np.hstack((visual_features, acoustic_features, text_features))
```

**Late Fusion** (Decision-level)
```python
# Majority voting across individual model predictions  
majority_votes = np.apply_along_axis(
    lambda x: np.bincount(x, minlength=4).argmax(),
    axis=0, arr=stacked_predictions
)
```

## Results

| Approach | F1-Score (%) | Std Dev | Notes |
|----------|-------------|---------|--------|
| **Early Fusion** | **62.56** | ±0.08 | Best performance |
| Text Only | 62.25 | ±0.12 | BERT's strong language understanding |
| Late Fusion | 54.31 | ±0.15 | Limited cross-modal learning |
| Audio Only | 53.27 | ±0.18 | Moderate VGGish performance |
| Visual Only | 38.39 | ±0.22 | Challenging facial emotion recognition |

## Key Insights

1. **Cross-modal interactions matter**: Early fusion's 8.25% improvement over late fusion shows the value of learning feature interactions during training
2. **Text dominance**: Language features carry the most emotion information in this dataset
3. **Feature complementarity**: Multimodal approaches capture nuances missed by individual modalities
4. **Evaluation rigor**: Subject-independent validation ensures realistic performance estimates

## Requirements

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
```

## Usage

```python
# Load and preprocess data
loader = MultimodalFeatureLoader('features.zip', 'dataset.csv')
dataset_features = loader.load_features()

# Run unimodal classification
visual_f1_scores, visual_conf_matrices = cross_validation(
    dataset_features['visual'], "Visual", 
    dataset_features['visual'].shape[1],
    dataset_features['speakers'], device
)

# Run early fusion
early_fusion_f1_scores, early_fusion_conf_matrices = early_fusion_cv(
    dataset_features['visual'], dataset_features['acoustic'], 
    dataset_features['lexical'], dataset_features['speakers'], device
)
```

## Project Structure

```
├── Narayana_Hitesh_HW4.ipynb    # Main implementation notebook
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── results/
    ├── confusion_matrices/      # Generated confusion matrices
    └── performance_plots/       # Performance comparison plots

Note: Dataset files (dataset.csv, features.zip) are not included due to 
IEMOCAP licensing restrictions. See Dataset section for access instructions.
```

## Technical Details

**Model Training**:
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Loss: CrossEntropyLoss
- Batch Size: 32
- Epochs: 50 with early stopping
- Hardware: CUDA-enabled GPU support

**Evaluation**:
- Metric: F1-micro score
- Validation: 5-fold subject-independent cross-validation
- Statistical Analysis: Mean ± standard deviation across folds

## Course Context

**CSCI 535 - Multimodal Machine Learning**  
University of Southern California, Fall 2024

This project demonstrates fundamental concepts in:
- Multimodal feature fusion
- Cross-validation methodology  
- Deep learning with PyTorch
- Experimental design and evaluation

## License

This project is for educational purposes as part of coursework at USC. 

**Dataset Licensing**: The IEMOCAP dataset is subject to a separate license agreement with USC SAIL Lab. Users must obtain permission directly from USC to access the original dataset.

**Code License**: The implementation code is available under the MIT License for educational and research purposes.
