# 概要设计 (High-Level Design)

> **Phase 2 — High-Level Design for bipartite-gnn-gui**
>
> This document defines the configuration system, logging and experiment tracking
> architecture, and dependency management strategy. These are the foundational
> infrastructure components that all downstream modules (data, graph, model, eval)
> depend on.

---

## 1. 配置系统设计

### 1.1 Design Principles

- **Single source of truth**: all hyperparameters and paths live in a single YAML file,
  loaded at runtime into pydantic-validated dataclass objects.
- **Separation of concerns**: three sub-configs (`DataConfig`, `ModelConfig`,
  `TrainingConfig`) group related parameters. A composite `Config` wraps them all.
- **Validation at startup**: pydantic `BaseModel` (strict mode) catches type errors and
  invalid values before any GPU compute is wasted.
- **Overridable at CLI**: the experiment runner (`experiments/run.py`) accepts
  `--overrides key=value` to modify any leaf parameter without editing YAML files.

### 1.2 DataConfig

Controls data sourcing, preprocessing paths, and train/val/test split ratios.

| Field | Type | Default | Description |
|---|---|---|---|
| `raw_dir` | `str` | `"data/raw"` | Directory containing downloaded raw datasets (GUI-360°, ScreenSpot). |
| `processed_dir` | `str` | `"data/processed"` | Directory where preprocessed `.pt`/`.pkl` files are cached. |
| `dataset_names` | `list[str]` | `["gui360", "screenspot"]` | Which datasets to load. Order matters: first dataset is used for primary training. |
| `val_split` | `float` | `0.1` | Fraction of training data held out for validation. Must satisfy `val_split > 0 and val_split < 1`. |
| `test_split` | `float` | `0.2` | Fraction of total data held out for final testing. Must satisfy `test_split > 0 and test_split < 1`. |

**Validation rules:**
- `val_split + test_split < 1.0` (residual is training).
- `raw_dir` must exist or be creatable.
- `dataset_names` must be non-empty and contain only known dataset keys.

```python
# Pydantic definition sketch
class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    dataset_names: list[str] = ["gui360", "screenspot"]
    val_split: float = Field(default=0.1, gt=0.0, lt=1.0)
    test_split: float = Field(default=0.2, gt=0.0, lt=1.0)
```

### 1.3 ModelConfig

Controls the GNN encoder architecture and the three prediction heads.

| Field | Type | Default | Description |
|---|---|---|---|
| `hidden_dim` | `int` | `256` | Hidden dimension used throughout the encoder and all heads. |
| `n_layers` | `int` | `2` | Number of message-passing layers (SAGEConv × `n_layers`, wrapped with `to_hetero`). |
| `dropout` | `float` | `0.1` | Dropout probability applied after each ReLU activation in the encoder. |
| `encoder_type` | `str` | `"sage"` | GNN convolution type. Options: `"sage"`, `"gat"`, `"gcn"`. Default `"sage"` uses `SAGEConv`. |
| `head_dims` | `dict` | see below | Dictionary controlling layer dimensions for each of the three prediction heads. |

**`head_dims` default structure:**

| Head key | Default `hidden_layers` | Description |
|---|---|---|
| `coordinate` | `[128, 64]` | Two hidden layers before the delta output (4D: Δcx, Δcy, Δw, Δh). |
| `violation` | `[128, 64]` | Two hidden layers before the sigmoid violation score output. |
| `existence` | `[128, 64]` | Two hidden layers before the sigmoid existence probability output. |

Each head always ends with an output layer of the appropriate dimension (4, 1, 1
respectively). The `hidden_layers` list specifies only intermediate layers.

```python
class HeadConfig(BaseModel):
    hidden_layers: list[int] = [128, 64]

class ModelConfig(BaseModel):
    hidden_dim: int = Field(default=256, gt=0)
    n_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    encoder_type: str = Field(default="sage", pattern="^(sage|gat|gcn)$")
    head_dims: dict[str, HeadConfig] = {
        "coordinate": HeadConfig(hidden_layers=[128, 64]),
        "violation": HeadConfig(hidden_layers=[128, 64]),
        "existence": HeadConfig(hidden_layers=[128, 64]),
    }
```

### 1.4 TrainingConfig

Controls optimizer, scheduler, batch size, reproducibility, and mixed-precision
settings.

| Field | Type | Default | Description |
|---|---|---|---|
| `lr` | `float` | `1e-3` | Peak learning rate for AdamW (after warmup). |
| `epochs` | `int` | `100` | Maximum number of training epochs (early stopping may cut this short). |
| `batch_size` | `int` | `32` | Number of samples per batch. Each sample is a single screenshot with its HeteroData graph. |
| `seed` | `int` | `42` | Global random seed for reproducibility (`torch`, `numpy`, `random`). |
| `weight_decay` | `float` | `1e-5` | L2 regularization coefficient applied to all non-bias parameters. |
| `warmup_steps` | `int` | `1000` | Number of warmup steps during which LR linearly increases from 0 to `lr`. |
| `grad_clip` | `float` | `1.0` | Max L2 norm for gradient clipping. Set to `0.0` to disable. |
| `amp` | `bool` | `True` | Enable automatic mixed precision (FP16) during training when a CUDA GPU is available. Falls back gracefully on CPU. |
| `early_stopping_patience` | `int` | `20` | Number of validation epochs without improvement before stopping. |
| `checkpoint_dir` | `str` | `"checkpoints"` | Directory for saving model checkpoints. |

```python
class TrainingConfig(BaseModel):
    lr: float = Field(default=1e-3, gt=0.0)
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    seed: int = Field(default=42, ge=0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    warmup_steps: int = Field(default=1000, ge=0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    amp: bool = True
    early_stopping_patience: int = Field(default=20, ge=1)
    checkpoint_dir: str = "checkpoints"
```

### 1.5 Config (Composite)

`Config` is the top-level pydantic model that nests the three sub-configs. It also
provides convenience methods for loading and saving.

```python
class Config(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load and validate config from a YAML file."""
        ...

    def to_yaml(self, path: str | Path) -> None:
        """Save current config to YAML (useful for experiment reproducibility)."""
        ...
```

**Loading flow:**
1. Read YAML file with `pyyaml`.
2. Pass the raw `dict` to `Config(**raw)`.
3. Pydantic validates all fields, nested models, and constraints.
4. If validation fails, raise a descriptive `ValidationError` listing all issues.

**Overrides flow (CLI):**
1. Parse `--overrides` as `key=value` pairs (e.g., `training.lr=5e-4`).
2. Merge into the raw config dict using dotted-key traversal before pydantic
   validation.
3. Validate the merged dict.

### 1.6 Default YAML Layout (`configs/default.yaml`)

Below is the complete, annotated default configuration file. This is the single source
of truth for all hyperparameters and paths.

```yaml
# =============================================================================
# Bipartite-GNN-GUI Default Configuration
# =============================================================================
# Copy this file to start a new experiment:
#   cp configs/default.yaml configs/my_experiment.yaml
# Then run:
#   python experiments/run.py --config configs/my_experiment.yaml
# Or override individual parameters:
#   python experiments/run.py --config configs/default.yaml \
#       --overrides training.lr=5e-4 training.epochs=200

# ---------------------------------------------------------------------------
# Data Configuration
# ---------------------------------------------------------------------------
data:
  # Directory containing raw downloaded datasets (GUI-360°, ScreenSpot)
  raw_dir: data/raw

  # Directory where preprocessed .pt/.pkl cache files are stored
  processed_dir: data/processed

  # Datasets to load for training and evaluation
  # Supported: "gui360", "screenspot"
  dataset_names:
    - gui360
    - screenspot

  # Fraction of data reserved for validation (held out from training set)
  val_split: 0.1

  # Fraction of total data reserved for final testing
  test_split: 0.2

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
model:
  # Hidden dimension for encoder layers and head inputs
  # Larger values increase capacity and memory usage
  hidden_dim: 256

  # Number of message-passing layers (SAGEConv wrapped with to_hetero)
  # 2 layers means each node sees its 2-hop neighborhood
  n_layers: 2

  # Dropout probability after each activation in the encoder
  dropout: 0.1

  # GNN convolution type
  #   "sage"  — GraphSAGE (SAGEConv) — recommended default
  #   "gat"   — Graph Attention (GATConv)
  #   "gcn"   — Graph Convolution (GCNConv)
  encoder_type: sage

  # Prediction head configurations
  # Each head maps from the encoder's hidden_dim through intermediate layers
  # to its output dimension. hidden_layers are before the final output layer.
  head_dims:
    # CoordinateRefinementHead: outputs (Δcx, Δcy, Δw, Δh) — 4D vector
    coordinate:
      hidden_layers: [128, 64]

    # ViolationPredictionHead: outputs violation score (sigmoid) — 1D
    violation:
      hidden_layers: [128, 64]

    # ExistencePredictionHead: outputs existence probability (sigmoid) — 1D
    existence:
      hidden_layers: [128, 64]

# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------
training:
  # Peak learning rate for AdamW optimizer (after warmup)
  lr: 0.001

  # Maximum number of training epochs
  # Early stopping may terminate training earlier
  epochs: 100

  # Batch size — number of HeteroData graphs per batch
  # Reduce if OOM; increase for better GPU utilization
  batch_size: 32

  # Global random seed for reproducibility
  seed: 42

  # L2 weight decay (applied to non-bias parameters only)
  weight_decay: 0.00001

  # Number of linear warmup steps from 0 to peak lr
  warmup_steps: 1000

  # Max L2 norm for gradient clipping (0.0 disables clipping)
  grad_clip: 1.0

  # Enable Automatic Mixed Precision (FP16) when CUDA is available
  amp: true

  # Stop training if validation loss does not improve for this many epochs
  early_stopping_patience: 20

  # Directory to save model checkpoints
  checkpoint_dir: checkpoints
```

---

## 2. 日志与实验跟踪架构

### 2.1 Structured Logging (`setup_logger`)

A module-level `setup_logger` function provides a consistent logging format across all
modules. Each logger writes to both console (stderr) and a rotating file.

**Interface:**

```python
def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a logger with console and file handlers.

    Args:
        name: Logger name, typically __name__ of the calling module.
        log_dir: Directory for log files.
        level: Minimum logging level.

    Returns:
        Configured logger instance.

    The returned logger has two handlers:
      - StreamHandler (stderr): for real-time console output.
      - RotatingFileHandler (log_dir/{name}.log): max 10 MB per file, 5 backups.
    """
```

**Log format (console and file):**

```
2026-05-25 14:32:01.456 | INFO  | data.dataset     | Loaded 1423 samples from gui360
2026-05-25 14:32:01.789 | DEBUG | graph.builder    | Built HeteroData: 45 element nodes, 12 constraint nodes
2026-05-25 14:32:02.012 | WARN  | model.trainer    | Validation loss increased for 5 consecutive epochs
```

Format string:

```
"%(asctime)s | %(levelname)-5s | %(name)-16s | %(message)s"
```

| Component | Example | Description |
|---|---|---|
| `asctime` | `2026-05-25 14:32:01.456` | Millisecond-precision timestamp. |
| `levelname` | `INFO` / `DEBUG` / `WARN` / `ERROR` | Padded to 5 characters. |
| `name` | `data.dataset` | Module-qualified logger name, padded to 16 chars. |
| `message` | Free text | Structured where possible (e.g., `metric=value` pairs). |

### 2.2 MetricsLogger Abstract Base

`MetricsLogger` defines the interface for experiment tracking. Concrete
implementations bridge to external services (Weights & Biases, TensorBoard) while
`NoopMetricsLogger` provides a zero-dependency fallback.

```python
class MetricsLogger(ABC):
    """Abstract base class for experiment metrics tracking.

    Implementations log scalars, hyperparameters, and artifacts to a backend
    (W&B, TensorBoard, local files, or no-op).
    """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log a dictionary of scalar metrics at a given step.

        Args:
            metrics: Mapping of metric name to scalar value.
                     Example: {"train/loss": 0.234, "train/coord_loss": 0.120}
            step: Global step counter (typically batch number or epoch).
        """
        ...

    @abstractmethod
    def log_hyperparams(self, params: dict) -> None:
        """Log hyperparameters once at the start of an experiment.

        Args:
            params: Flat dictionary of hyperparameter name-value pairs.
                    Nested Config objects should be flattened: training.lr -> 1e-3.
        """
        ...

    @abstractmethod
    def save(self) -> None:
        """Persist in-memory buffer to disk (e.g., flush async uploads)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Finalize and clean up backend resources."""
        ...

    def __enter__(self) -> "MetricsLogger":
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit — calls close()."""
        self.close()
```

**Convention: metric name prefixes**

| Prefix | Meaning | Example |
|---|---|---|
| `train/` | Training set metrics (per epoch) | `train/loss`, `train/coord_loss` |
| `val/` | Validation set metrics | `val/loss`, `val/element_recall` |
| `test/` | Test set (final evaluation only) | `test/element_recall` |
| `batch/` | Per-batch metrics (logged every N batches) | `batch/loss` |

### 2.3 NoopMetricsLogger (Fallback)

Always available, zero-dependency logger that silently discards all metrics. Used when
no external tracking service is configured.

```python
class NoopMetricsLogger(MetricsLogger):
    """No-op metrics logger. Accepts all calls and does nothing.

    This is the default logger when no external tracking backend is installed
    or configured. It ensures that training code does not need conditional
    branches for the logger.
    """

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        pass

    def log_hyperparams(self, params: dict) -> None:
        pass

    def save(self) -> None:
        pass

    def close(self) -> None:
        pass
```

### 2.4 WandbMetricsLogger (Optional)

Powers full experiment tracking via Weights & Biases. Guarded by an optional import
so the package works without `wandb` installed.

```python
class WandbMetricsLogger(MetricsLogger):
    """Metrics logger backed by Weights & Biases.

    Requires: pip install bipartite-gnn-gui[wandb]

    Features:
      - Automatic hyperparameter logging via wandb.config.
      - Scalar metrics with step-based x-axis.
      - Artifact logging (model checkpoints, figures).
      - Automatic system metrics (GPU util, memory, CPU).
    """

    def __init__(
        self,
        project: str = "bipartite-gnn-gui",
        name: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
        entity: str | None = None,
    ) -> None:
        """Initialize W&B run.

        If wandb is not installed, raises ImportError with install instructions.
        Callers should catch this and fall back to NoopMetricsLogger.
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Install with: pip install bipartite-gnn-gui[wandb]"
            )
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags or [],
            entity=entity,
        )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._run.log(metrics, step=step)

    def log_hyperparams(self, params: dict) -> None:
        self._run.config.update(params)

    def save(self) -> None:
        # wandb.log flushes immediately; no-op
        pass

    def close(self) -> None:
        self._run.finish()
```

**Optional import guard pattern (used in `create_logger` factory):**

```python
def create_logger(
    backend: str = "noop",
    **kwargs,
) -> MetricsLogger:
    """Factory for creating a MetricsLogger.

    Args:
        backend: One of "noop", "wandb", "tensorboard".
        **kwargs: Forwarded to the specific logger constructor.

    Returns:
        MetricsLogger instance.

    If the requested backend's dependencies are not installed, the factory
    prints a warning and returns a NoopMetricsLogger instead of raising.
    """
    if backend == "wandb":
        try:
            return WandbMetricsLogger(**kwargs)
        except ImportError as e:
            logger.warning(f"Cannot create WandbMetricsLogger: {e}. Falling back to noop.")
            return NoopMetricsLogger()
    elif backend == "tensorboard":
        try:
            return TensorboardMetricsLogger(**kwargs)
        except ImportError as e:
            logger.warning(f"Cannot create TensorboardMetricsLogger: {e}. Falling back to noop.")
            return NoopMetricsLogger()
    else:
        return NoopMetricsLogger()
```

### 2.5 TensorboardMetricsLogger (Optional)

Standard TensorBoard logging via `torch.utils.tensorboard.SummaryWriter`.

```python
class TensorboardMetricsLogger(MetricsLogger):
    """Metrics logger backed by TensorBoard.

    Requires: pip install bipartite-gnn-gui[tensorboard]

    Writes to a log directory compatible with:
      tensorboard --logdir runs/
    """

    def __init__(
        self,
        log_dir: str = "runs",
        comment: str = "",
    ) -> None:
        """Initialize TensorBoard writer.

        If tensorboard/torch.utils.tensorboard is not available, raises ImportError.
        Callers should catch this and fall back to NoopMetricsLogger.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                "TensorBoard is not installed. Install with: pip install bipartite-gnn-gui[tensorboard]"
            )
        self._writer = SummaryWriter(log_dir=log_dir, comment=comment)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, global_step=step)
        self._writer.flush()

    def log_hyperparams(self, params: dict) -> None:
        # TensorBoard hparams requires at least one metric for comparison;
        # we defer actual logging to the first log_metrics call.
        self._hparams = params

    def save(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
```

---

## 3. 依赖管理 (`pyproject.toml`)

### 3.1 Design Principles

- **Lean core**: `dependencies` contains only packages required to run the main
  training/evaluation pipeline. No GUI, no external tracking, no code quality tools.
- **Optional extras**: `wandb` and `tensorboard` are separate install groups so users
  opt into tracking backends.
- **Clean separation of dev/test**: `dev` includes formatting, linting, and type
  checking. `test` is minimal (pytest + coverage). CI can install `[test]` without the
  heavier `[dev]` tools.
- **No hard pins**: minimum version constraints are specified with `>=`. Exact pinning
  is done in a lock file if needed downstream.

### 3.2 Dependency Groups

#### `core` (project.dependencies)

These packages are installed with `pip install bipartite-gnn-gui`. They cover the
full training → evaluation → inference loop.

| Package | Min Version | Purpose |
|---|---|---|
| `torch` | `>=2.1.0` | Deep learning framework; GNN backbone. |
| `torch-geometric` | `>=2.4.0` | Graph neural network layers (`SAGEConv`, `HeteroData`, `to_hetero`). |
| `numpy` | `>=1.24.0` | Numerical arrays; coordinate math. |
| `pillow` | `>=10.0.0` | Screenshot image loading (for visualization). |
| `pyyaml` | `>=6.0` | YAML config file parsing. |
| `pydantic` | `>=2.5.0` | Config schema validation via `BaseModel`. |
| `scipy` | `>=1.11.0` | Hungarian algorithm for bipartite matching (`linear_sum_assignment`), statistical tests (Wilcoxon, bootstrap). |
| `tqdm` | `>=4.66.0` | Progress bars for training and data preprocessing. |

#### `test` (project.optional-dependencies)

Installed with `pip install bipartite-gnn-gui[test]`. Minimal set for running the test
suite.

| Package | Min Version | Purpose |
|---|---|---|
| `pytest` | `>=7.4.0` | Test runner and assertion framework. |
| `pytest-cov` | `>=4.1.0` | Code coverage reports (`--cov`). |

#### `dev` (project.optional-dependencies)

Installed with `pip install bipartite-gnn-gui[dev]`. Code quality and formatting tools.
Used in pre-commit hooks and CI linting jobs.

| Package | Min Version | Purpose |
|---|---|---|
| `black` | `>=24.0.0` | Opinionated code formatter (line length 88). |
| `ruff` | `>=0.3.0` | Fast Python linter (replaces flake8, isort, pyupgrade). |
| `mypy` | `>=1.8.0` | Static type checker with strict mode. |

#### `wandb` (project.optional-dependencies)

Installed with `pip install bipartite-gnn-gui[wandb]`. Weights & Biases experiment
tracking.

| Package | Min Version | Purpose |
|---|---|---|
| `wandb` | `>=0.16.0` | Experiment tracking, hyperparameter sweeps, artifact management. |

#### `tensorboard` (project.optional-dependencies)

Installed with `pip install bipartite-gnn-gui[tensorboard]`. TensorBoard event
logging. Note: `tensorboard` is a dependency of `torch`, but its `SummaryWriter` may
require a separate install in some environments.

| Package | Min Version | Purpose |
|---|---|---|
| `tensorboard` | `>=2.14.0` | TensorBoard event file writing. |

### 3.3 Complete `pyproject.toml` Fragments

Only the sections that change from the current baseline are shown below. Lines
prefixed with `+` are additions; `-` are removals.

```toml
# ---------------------------------------------------------------------------
# [project] dependencies — Core packages
# ---------------------------------------------------------------------------
dependencies = [
    "torch>=2.1.0",
    "torch-geometric>=2.4.0",
    "numpy>=1.24.0",
    "pillow>=10.0.0",
    "pyyaml>=6.0",
    "pydantic>=2.5.0",
    "scipy>=1.11.0",
    "tqdm>=4.66.0",
]

# ---------------------------------------------------------------------------
# [project.optional-dependencies]
# ---------------------------------------------------------------------------
[project.optional-dependencies]
dev = [
    "black>=24.0.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]

test = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]

wandb = [
    "wandb>=0.16.0",
]

tensorboard = [
    "tensorboard>=2.14.0",
]
```

**Rationale for changes from current baseline:**

| Change | Rationale |
|---|---|
| Add `black`, `ruff`, `mypy` to `[dev]` | Standard code quality stack for a Python research project. |
| Remove `pytest` from `[dev]` | `pytest` belongs in `[test]` only. CI can install `[test]` without dev tools. |
| Remove `matplotlib` from `[dev]` | Not a core dev dependency. Visualization code uses `pillow` + custom plotting that may pull in `matplotlib` as an optional extra if needed. |
| Remove `transformers` from `[dev]` | Not a dependency of this project (no HuggingFace models used). |
| Add `pytest-cov` to `[test]` | Enables coverage measurement in CI. |
| Add `[wandb]` extra | Enables `pip install bipartite-gnn-gui[wandb]`. |
| Add `[tensorboard]` extra | Enables `pip install bipartite-gnn-gui[tensorboard]`. |

### 3.4 Install Commands Summary

| Use case | Command |
|---|---|
| Minimal training setup | `pip install -e .` |
| Development with linting | `pip install -e ".[dev]"` |
| Run tests with coverage | `pip install -e ".[test]"` && `pytest --cov` |
| Full setup (all optional) | `pip install -e ".[dev,test,wandb,tensorboard]"` |
| W&B experiment tracking | `pip install -e ".[wandb]"` |
| TensorBoard tracking | `pip install -e ".[tensorboard]"` |
