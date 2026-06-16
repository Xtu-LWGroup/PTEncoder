# PTEncoder: A Temporal-Enhanced Contrastive Learning Framework for Gene Regulatory Network Inference

## Overview

PTEncoder is a deep learning framework designed for inferring gene regulatory networks (GRNs) from single-cell RNA sequencing data with pseudotemporal information. The framework integrates contrastive learning, motif-aware graph neural networks, and neural ordinary differential equations (Neural ODEs) to capture both structural and temporal dynamics of gene regulation.

## Key Features

1. **Temporal-Aware Feature Encoding**: Incorporates pseudotime information through gene-level temporal feature extraction, capturing dynamic expression patterns across cellular trajectories.

2. **Motif-Based Graph Neural Networks**: Utilizes network motifs (edges and triangles) to encode local regulatory patterns with attention mechanisms.

3. **Dual-Path Architecture**: Combines local motif-based encoding with global path encoding using position-weighted graph neural networks.

4. **Neural ODE Integration**: Models continuous-time gene expression dynamics through neural ordinary differential equations.

5. **Contrastive Learning Framework**: Employs momentum contrastive learning with temporal position-weighted InfoNCE loss for robust representation learning.

6. **Attention Fusion Mechanism**: Dynamically balances local and global features through learnable attention weights.

## System Requirements

### Hardware
- GPU with CUDA support (recommended: NVIDIA GPU with ≥8GB VRAM)
- Minimum 16GB RAM
- 50GB free disk space

### Software Dependencies
```
Python >= 3.8
PyTorch >= 1.9.0
DGL (Deep Graph Library) >= 0.7.0
PyTorch Lightning >= 1.5.0
NumPy >= 1.20.0
Pandas >= 1.3.0
NetworkX >= 2.6.0
torchmetrics >= 0.6.0
scikit-learn >= 0.24.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repository/PTEncoder.git
cd PTEncoder

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install pytorch-lightning numpy pandas networkx torchmetrics scikit-learn

# Verify installation
python -c "import torch; import dgl; print('Installation successful')"
```

## Data Format

### Input Files

The framework expects three types of input files in CSV format:

1. **Expression Data** (`ExpressionData.csv`):
   - Format: Genes × Cells matrix
   - Rows: Gene names (indexed)
   - Columns: Cell identifiers
   - Values: Normalized expression levels

2. **Network Topology** (`network.csv` or `*network*.csv`):
   - Format: Edge list
   - Columns: Source gene index, Target gene index
   - Represents known or predicted regulatory interactions

3. **Pseudotime Data** (optional, `PseudoTime.csv`):
   - Format: Single column vector
   - Rows: Cells (matching expression data order)
   - Values: Pseudotime values (will be normalized to [0, 1])

### Example Directory Structure

```
data/
├── Specific Dataset/
│   ├── ChIP-seq_mESC/
│   │   ├── TFs500/
│   │   │   ├── ExpressionData.csv
│   │   │   ├── 500_ChIP-seq_mESC-network.csv
│   │   │   └── PseudoTime.csv
│   │   └── TFs1000/
│   │       ├── ExpressionData.csv
│   │       ├── network.csv
│   │       └── PseudoTime.csv
└── Non-Specific Dataset/
    └── hESC/
        └── TFs500/
            ├── ExpressionData.csv
            └── network.csv
```

## Usage

### Basic Execution

Run the main experiment script:

```bash
cd src
python experiment_enhanced_clvgae.py
```

### Configuration

Modify parameters in `src/utils/Config.py`:

```python
class _Config:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    EPOCH = 2000                    # Maximum training epochs
    LEARNING_RATE = 1e-3           # Initial learning rate
    WEIGHT_DECAY = 0.              # Weight decay for regularization
    TEMPERATURE = 0.2              # Temperature for contrastive loss
    MOMENTUM = 0.01                # Momentum coefficient
    PATIENCE = 500                 # Early stopping patience
```

### Custom Dataset Setup

Edit the dataset configuration in `experiment_enhanced_clvgae.py`:

```python
# Define datasets and cell types
datasets = ['Specific Dataset']
cell_type = ['ChIP-seq_mESC']
dataset_names = ['Specific']
n_tf = ['TFs500']
tf = ['500']

# Set data paths
root_dir = '/path/to/data/directory'
save_dir = '/path/to/output/directory'

# Define train/test split ratios
split_ratios = [[0.8, 0.2]]  # [train_ratio, test_ratio]
```

### Model Parameters

Configure model architecture in the `run_enhanced_experiment()` function:

```python
enhanced_results = run_enhanced_experiment(
    exp_path, tpy_path, sr, 
    use_enhanced=True,
    hidden_dim=406,      # Hidden layer dimension
    num_anchors=10       # Number of anchor nodes for global encoding
)
```

## Algorithm Workflow

### 1. Data Preprocessing

The preprocessing pipeline (`Preprocessor_2`) performs:

- **Expression Matrix Loading**: Reads and validates gene expression data
- **Pseudotime Processing**: Loads and normalizes pseudotime values to [0, 1]
- **Temporal Feature Extraction**: Computes gene-level temporal features including:
  - Time-bin averaged expression profiles
  - Expression variance across time bins
  - Temporal gradients (expression change rates)
  - Peak expression timing
  - Dynamic expression ranges
- **Feature Concatenation**: Appends temporal features to original expression features
- **Graph Construction**: Builds DGL graphs from network topology
- **Motif Adjacency Matrices**: Extracts edge and triangle motifs from the network
- **Train/Test Split**: Partitions edges based on specified ratio

### 2. Model Architecture

#### Dual-Path Encoder

**Local Path Encoder (MotifGCN)**:
- Processes local network structures using motif-specific convolutions
- Applies attention mechanism across different motif types
- Optionally incorporates temporal kernel weighting: `W_temporal = exp(-Δt²/(2σ²))`
- Multi-layer architecture with residual connections

**Global Path Encoder (PGNN)**:
- Selects anchor nodes based on network centrality measures (degree, betweenness, closeness, etc.)
- Computes shortest-path distances from all nodes to anchors
- Encodes global positional information through distance-aware message passing
- Integrates temporal positional encoding when pseudotime is available

**Attention Fusion Layer**:
- Learns adaptive weights for combining local and global features
- Output: `H_fused = α·H_local + (1-α)·H_global`

#### Neural ODE Layer

Models continuous-time dynamics:
```
dz/dt = f(z, t; θ)
z(t) = z(0) + ∫₀ᵗ f(z(τ), τ; θ) dτ
```

Where:
- `z`: Gene embedding state
- `t`: Pseudotime
- `f`: Neural network parameterized by θ
- Solved using Euler or Runge-Kutta methods

#### Variational Graph Autoencoder

- **Encoder**: Dual-path encoder + Neural ODE → latent representations
- **Reparameterization**: `z = μ + ε ⊙ exp(σ)`, where ε ~ N(0, I)
- **Decoder**: Reconstructs adjacency matrix from latent embeddings
- **Momentum Encoder**: Maintains slowly-updated copy for contrastive learning

### 3. Training Procedure

**Loss Function Components**:

1. **Binary Cross-Entropy Loss** (reconstruction):
   ```
   L_BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
   ```

2. **KL Divergence** (regularization):
   ```
   L_KL = -0.5·Σ(1 + log(σ²) - μ² - σ²)
   ```

3. **Temporal Position-Weighted InfoNCE Loss** (contrastive):
   ```
   L_InfoNCE = -log[exp(sim(q,k⁺)/τ) / Σⱼ exp(sim(q,kⱼ)/τ)]
   ```
   Where similarity is weighted by temporal proximity and positional importance.

**Total Loss**:
```
L_total = λ₁·L_BCE + λ₂·L_KL + λ₃·L_InfoNCE + λ₄·L_BCE_negative
```

Default weights: λ₁=1.0, λ₂=0.01, λ₃=0.1, λ₄=1.0

**Optimization**:
- Optimizer: AdamW
- Gradient clipping: norm ≤ 1.0
- Early stopping: patience = 500 epochs
- Learning rate: 1e-3 (fixed)

### 4. Evaluation Metrics

Performance is assessed using multiple metrics across different negative-to-positive ratios (1:1, 4:1, 9:1, 19:1):

- **AUROC** (Area Under Receiver Operating Characteristic curve)
- **AUPRC** (Area Under Precision-Recall Curve)
- **Accuracy** (threshold = 0.5)
- **F1 Score** (threshold = 0.5)
- **Precision** (threshold = 0.5)
- **Recall** (threshold = 0.5)
- **MCC** (Matthews Correlation Coefficient)

Each metric is averaged over 20 independent runs with random negative sampling.

## Output Format

Results are saved as JSON files in the specified output directory:

```json
{
  "enhanced": {
    "results": {
      "ratio_1": {
        "auroc": 0.923,
        "auprc": 0.876,
        "accuracy": 0.854,
        "f1": 0.832,
        "recall": 0.845,
        "precision": 0.819,
        "mcc": 0.708
      },
      "ratio_4": {...},
      "ratio_9": {...},
      "ratio_19": {...}
    },
    "preprocessing_time": 45.23,
    "training_time": 1823.56,
    "total_time": 1868.79,
    "feature_dim": 256,
    "gene_count": 500,
    "model_type": "EnhancedCLVGAE"
  },
  "dataset": "/path/to/dataset",
  "timestamp": "2024-01-15 14:30:22"
}
```

## Advanced Features

### Parameter Sensitivity Analysis

Evaluate model performance across different hyperparameter configurations:

```python
from experiment_enhanced_clvgae import parameter_sensitivity_analysis
parameter_sensitivity_analysis()
```

Tests combinations of:
- Hidden dimensions: [64, 128, 256]
- Anchor numbers: [5, 10, 15]

Results saved to `parameter_sensitivity_results.json`.

### Ablation Studies

Compare different model variants by modifying:
- `use_neural_ode`: Enable/disable Neural ODE layer
- `use_temporal_kernel`: Enable/disable temporal motif weighting
- `centrality_method`: Choose anchor selection strategy ('degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank')

### DeepProfile Integration

Enable dimensionality reduction for high-dimensional expression data:

```python
preprocessor = Preprocessor_2(
    exp_path, tpy_path, sr,
    use_deepprofile=True,
    latent_dim=100,
    hidden_dims=[512, 256, 128],
    deepprofile_epochs=100
)
```

## Performance Optimization

### Memory Management

For large-scale datasets (>10,000 genes):

1. Reduce batch size in ShaDowKHopSampler: `[100, 30, 5]` instead of `[150, 50, 10]`
2. Decrease hidden dimension to 128 or 64
3. Use fewer anchor nodes (5-10)
4. Enable gradient checkpointing in PyTorch Lightning

### GPU Acceleration

Ensure optimal GPU utilization:

```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

Monitor GPU memory:
```bash
nvidia-smi -l 1
```

### Parallel Processing

Distance computation uses multiprocessing:
```python
num_workers = min(mp.cpu_count(), 8)  # Adjust based on system resources
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Solution: Reduce `hidden_dim`, decrease sampler sizes, or process smaller batches
   
2. **NaN Loss Values**:
   - Solution: Check for numerical stability in temporal features, verify pseudotime normalization
   
3. **Slow Training**:
   - Solution: Enable mixed precision training, reduce number of anchors, simplify motif types
   
4. **Poor Convergence**:
   - Solution: Increase `PATIENCE`, adjust learning rate, verify data quality

### Debug Mode

Enable detailed logging:
```python
trainer = Trainer(
    accelerator='gpu',
    max_epochs=CONFIG.EPOCH,
    devices=1,
    logger=True,  # Enable TensorBoard logging
    fast_dev_run=False,
    enable_checkpointing=True
)
```

## Citation

If you use PTEncoder in your research, please cite:

```bibtex
@article{ptencoder2024,
  title={PTEncoder: Temporal-Enhanced Contrastive Learning for Gene Regulatory Network Inference},
  author={Your Name et al.},
  journal={Bioinformatics},
  year={2024},
  volume={XX},
  number={X},
  pages={XXX-XXX}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, bug reports, or feature requests, please contact:
- Email: your.email@institution.edu
- GitHub Issues: https://github.com/your-repository/PTEncoder/issues

## Acknowledgments

We thank the developers of PyTorch, DGL, and PyTorch Lightning for providing excellent deep learning frameworks. This work was supported by [Funding Agency].

## Version History

- **v1.0.0** (2024-01-15): Initial release
  - Core PTEncoder framework
  - Temporal feature extraction
  - Motif-aware GNN
  - Neural ODE integration
  - Contrastive learning module
