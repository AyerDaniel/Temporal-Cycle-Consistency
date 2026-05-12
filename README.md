# Temporal Cycle-Consistency Learning
## Self-supervised video representations with TCN $\rightarrow$ TCC using `aegean-ai/tcc`

This notebook is designed to read like a **tutorial** and a **course assignment** at the same time.

It focuses on one question:

> Can a self-supervised video representation learn the **latent phase** of a task purely from temporal structure, without robot control, action labels, or transcription?

The answer developed across the two Google Research papers is:

1. **TCN** learns by enforcing *time-based contrastive alignment* under strong synchronization assumptions.
2. **TCC** generalizes this idea by enforcing *temporal cycle-consistency*, which is more robust to variations in execution speed and alignment.

In this notebook you will:

- study the conceptual evolution from **TCN** to **TCC**
- train the **PyTorch rewrite** in `aegean-ai/tcc`
- extract frame embeddings
- visualize trajectories with **PCA**, **t-SNE**, and **UMAP**
- segment action sequences using representation geometry

## Papers

- Sermanet et al., **Time-Contrastive Networks**, 2018  
  https://arxiv.org/abs/1704.06888

- Dwibedi et al., **Temporal Cycle-Consistency Learning**, 2019  
  https://arxiv.org/abs/1904.07846

## Repository used in this notebook

- `https://github.com/aegean-ai/tcc`  
  This notebook assumes the **`main` branch** and the current PyTorch package layout under `src/tcc/`.

## What you should learn

By the end, you should be able to explain why TCN and TCC are related but not identical:

- TCN: **metric alignment with synchronized positives**
- TCC: **structural temporal alignment via cycles**

A useful mental model is:

- TCN says: *"frames at the same time index should be close."*
- TCC says: *"if I map from one sequence to another and back, I should return to the same temporal phase."*

## 1. Theory recap: from TCN to TCC

### 1.1 TCN: contrastive temporal alignment

TCN learns an embedding
$$
z_t = f_\theta(I_t)
$$
so that synchronized frames from different views become neighbors in feature space.

A canonical triplet-style loss is:
$$
\mathcal{L}_{\mathrm{TCN}}
=
\max\left(
0,\;
\|f(I_t^a)-f(I_t^b)\|_2^2
-
\|f(I_t^a)-f(I_{t'}^b)\|_2^2
+
\alpha
\right).
$$

Interpretation:

- **anchor:** frame $I_t^a$
- **positive:** synchronized frame $I_t^b$
- **negative:** mismatched-time frame $I_{t'}^b$

This already encodes an important idea: **time is supervision**.

But TCN assumes that corresponding frames are available at matching time indices, which is a strong assumption.

### 1.2 Why TCN is not enough

Suppose two people perform the same pouring task:

- one moves slowly
- one moves quickly
- one pauses before tilting
- one starts tilting earlier

Then the frame with semantic phase "tilt begins" is **not** guaranteed to occur at the same time index in both videos.

So absolute time matching becomes fragile.

### 1.3 TCC: align temporal structure, not raw clock time

TCC keeps the idea that embeddings should reflect task progression, but replaces hard synchronized matching with **cycle consistency**.

**Conceptual intuition.** Given frame $i$ in sequence $A$, map it to the most corresponding frame in sequence $B$:
$$
j = \arg\min_k \|f(I_i^A)-f(I_k^B)\|.
$$

Then map back from sequence $B$ to sequence $A$:
$$
i' = \arg\min_l \|f(I_j^B)-f(I_l^A)\|.
$$

TCC encourages $i' \approx i$.

**Differentiable training loss.** The hard `argmin` above is not differentiable, so the actual TCC loss replaces it with a **soft nearest-neighbor** formulation. For frame $i$ in sequence $A$, define a soft correspondence distribution over frames in sequence $B$:

$$
\beta_k^{(i)} = \frac{\exp(-\|f(I_i^A) - f(I_k^B)\|^2 / \tau)}{\sum_{k'} \exp(-\|f(I_i^A) - f(I_{k'}^B)\|^2 / \tau)}
$$

where $\tau$ is a temperature parameter. The cycle-back distribution is computed analogously, and the loss is the cross-entropy between the back-mapped distribution and a target concentrated at the original index $i$. This makes the entire cycle differentiable and trainable with standard gradient descent.

Conceptually:

- TCN aligns **absolute timestamps**
- TCC aligns **latent phase structure**

This is why TCC is more appropriate when demonstrations are semantically similar but **temporally warped**.

### 1.4 What the embedding should look like on pouring

If TCC works, then the learned trajectory in embedding space should behave like a latent phase variable:

- early reach frames cluster near other early reach frames
- grasp transitions appear near one another
- tilt and pour form coherent regions
- embeddings from different videos should trace similar temporal paths

That is the premise you will test below.

## 2. How this notebook and the `aegean-ai/tcc` repo work together

This notebook is **not** a standalone script. It is a guided analysis layer that drives the `aegean-ai/tcc` PyTorch package. The repo provides the training loop, model definitions, dataset utilities, and evaluation code. The notebook provides the experimental protocol: configuring runs, extracting embeddings, and visualizing results.

### Two supported environments

| | **Dev container (recommended)** | **Google Colab** |
|---|---|---|
| **GPU** | Local NVIDIA GPU via Docker | Colab T4/A100 runtime |
| **Package manager** | `uv` (pre-installed in container) | `pip` (Colab default) |
| **Setup effort** | `make start` — one command | Clone + pip install in notebook cells |
| **Persistence** | Full local disk | Session-scoped (data lost on disconnect) |
| **Best for** | Full sweep, large runs | Quick experiments, no local GPU |

Choose **one** environment and follow the corresponding setup path in Section 3.

### Workflow overview

```
┌──────────────────────────────────────────────────────────────────┐
│  This notebook (analysis layer)                                  │
│                                                                  │
│  1. Set up environment (dev container OR Colab)                  │
│  2. Prepare the pouring dataset                                  │
│  3. Configure training via tcc.config.get_default_config()       │
│  4. Launch training via tcc.train.train(cfg)                     │
│  5. Load checkpoints via tcc.train.load_checkpoint()             │
│  6. Extract embeddings via tcc.evaluate.get_embeddings_dataset() │
│  7. Visualize and segment (PCA, UMAP, KMeans — notebook code)   │
└──────────────────────────────────────────────────────────────────┘
         │                        ▲
         │  function calls        │  returns tensors,
         ▼                        │  checkpoints, configs
┌──────────────────────────────────────────────────────────────────┐
│  aegean-ai/tcc  (installed as editable package)                  │
│                                                                  │
│  src/tcc/                                                        │
│  ├── config.py          TCCConfig dataclass + get_default_config │
│  ├── train.py           Training loop, checkpoint save/load      │
│  ├── evaluate.py        Embedding extraction, eval metrics       │
│  ├── datasets.py        DataConfig, create_dataset()             │
│  ├── models.py          ResNet backbone + embedding head         │
│  ├── alignment.py       TCC alignment algorithm                  │
│  ├── losses.py          Cycle-consistency loss                   │
│  └── algos/             Algorithm registry (tcc, tcn, sal, …)    │
│                                                                  │
│  configs/                                                        │
│  └── default.yaml       Default hyperparameters                  │
│                                                                  │
│  scripts/                                                        │
│  └── download_pouring_data.sh                                    │
│                                                                  │
│  src/tcc/dataset_preparation/                                    │
│  ├── videos_to_dataset.py    Raw videos → image folders          │
│  ├── images_to_dataset.py    Images → dataset structure          │
│  └── visualize_dataset.py    Inspect prepared data               │
└──────────────────────────────────────────────────────────────────┘
```

### What you modify vs. what you use as-is

| Layer | You modify | You use as-is |
|-------|-----------|---------------|
| **Notebook** | Embedding dimension, iteration count, analysis parameters ($k$, projection method) | Visualization and segmentation code |
| **Repo config** | `model.conv_embedder.embedding_size`, `train.max_iters`, `logdir` | Everything else in `configs/default.yaml` |
| **Repo code** | Nothing — treat as a library | `train.py`, `evaluate.py`, `datasets.py`, `models.py` |

## 3. Environment setup

Choose **one** of the two paths below. Both result in a working `import tcc` with GPU access.

---

### Path A: Dev container (recommended for full assignment)

The repo ships a complete Docker-based development environment with GPU support, `uv`, and VS Code integration.

**Prerequisites:** Docker with NVIDIA Container Toolkit, VS Code with Dev Containers extension.

**Steps:**

1. Clone the repo locally:
   ```bash
   git clone https://github.com/aegean-ai/tcc && cd tcc
   ```
2. Copy the environment file:
   ```bash
   cp .env.example .env
   # Edit .env to add WANDB_API_KEY and/or HF_TOKEN if needed
   ```
3. Open in VS Code → "Reopen in Container" (or run `docker compose up -d` manually).
4. Inside the container, run:
   ```bash
   make start
   ```
   This creates a `.venv` with `uv`, installs the package in editable mode, and registers a Jupyter kernel.
5. Open this notebook in VS Code or JupyterLab (port 8888) and select the **"Python 3 (tcc)"** kernel.

**Key details:**
- Base image: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`
- Package manager: `uv` (not pip) — the Makefile handles all `uv` calls
- Python: whatever 3.11+ is in the container (typically from conda)
- Workspace: `/workspaces/tcc`
- TensorBoard: port 6006

**Installing extra notebook dependencies** (matplotlib, umap-learn, etc.):
```bash
make install-notebooks
```

---

### Path B: Google Colab (quick start, no local GPU needed)

Use this path if you do not have a local GPU or want a fast start. Colab sessions are ephemeral — save checkpoints to Google Drive to avoid losing training results.

**Steps:**

1. In a Colab notebook, enable GPU: **Runtime → Change runtime type → T4 GPU**.
2. Run the clone and install cells below (Section 3.1–3.2).
3. Colab uses `pip` — the `%pip install` commands handle everything.

**Limitations:**
- Session timeout erases all local files. Mount Google Drive for persistence:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  # Point EXPERIMENT_ROOT and DATA_ROOT to /content/drive/MyDrive/tcc/
  ```
- Colab's default Python may differ from 3.11 — the package should still install but is only tested on 3.11–3.12.

---

### Python version requirement

The repo requires **Python ≥3.11, <3.13** (`pyproject.toml`). The dev container satisfies this automatically. On Colab, check with `!python --version`.


```python
import sys, platform, os, pathlib

print("Python:", sys.version)
print("Platform:", platform.platform())
print("Working directory:", os.getcwd())
```

    Python: 3.11.13 | packaged by conda-forge | (main, Jun  4 2025, 14:48:23) [GCC 13.3.0]
    Platform: Linux-6.17.0-14-generic-x86_64-with-glibc2.35
    Working directory: /workspaces/tcc/notebooks/self-supervised


### 3.1 Clone the repository (Colab / Path B only)

If you are using the **dev container** (Path A), skip this — the repo is already your workspace at `/workspaces/tcc`.


```python
# import subprocess, pathlib

# REPO_URL = "https://github.com/aegean-ai/tcc"
# REPO_DIR = pathlib.Path("tcc")

# if not REPO_DIR.exists():
#     subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=False)
# else:
#     print("Repository already exists:", REPO_DIR)

# print("Repo dir exists:", REPO_DIR.exists())
```

### 3.2 Install the package (Colab / Path B only)

If you are using the **dev container** (Path A), skip this — `make start` already installed the package. Run `make install-notebooks` if you need matplotlib/umap-learn.


```python
# Colab / Path B only — uncomment and run these lines:
# %pip install -e ./tcc
# %pip install matplotlib scikit-learn umap-learn tqdm pyyaml

# Dev container / Path A — these are already installed.
# If you need notebook extras, run in a terminal: make install-notebooks

# import importlib
# try:
#     importlib.import_module("tcc")
#     print("tcc package is available.")
# except ModuleNotFoundError:
#     print("tcc not found. Follow the install instructions for your environment (Path A or B).")
```

### 3.3 Quick repository inspection

Verify the repo structure. In the dev container the repo root is `/workspaces/tcc`; on Colab it is the cloned `tcc/` directory.


```python
import os

# Detect environment: dev container vs Colab
REPO_ROOT = pathlib.Path("/workspaces/tcc") if pathlib.Path("/workspaces/tcc/src/tcc").exists() else REPO_DIR

def walk_top(path, max_depth=2):
    base = os.path.abspath(path)
    label = os.path.basename(base)
    for root, dirs, files in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth <= max_depth:
            print(root.replace(base, label))
            for f in files[:10]:
                print("   ", f)

walk_top(str(REPO_ROOT), max_depth=2)
```

    tcc
        1904.07846v1.pdf
        CLAUDE.md
        pyproject.toml
        .gitignore
        Makefile
        .env
        .~lock.MidTerm_writeup.odt#
        MidTerm_writeup.odt
        README.md
        uv.lock
    tcc/.venv
        .gitignore
        .lock
        pyvenv.cfg
        CACHEDIR.TAG
    tcc/.venv/etc
    tcc/.venv/lib
    tcc/.venv/share
    tcc/.venv/include
    tcc/.venv/bin
        activate_this.py
        jupyter-notebook
        ipython
        jlpm
        typer
        python3
        activate
        pygmentize
        deactivate.bat
        jupyter-console
    tcc/scripts
        execute_notebook.py
        download_pouring_data.sh
    tcc/docker
        Dockerfile.torch.dev.gpu
    tcc/.beads
        dolt-monitor.pid
        config.yaml
        interactions.jsonl
        .gitignore
        dolt-server.port
        metadata.json
        README.md
        dolt-server.activity
    tcc/.beads/hooks
        post-checkout
        post-merge
        pre-push
        prepare-commit-msg
        pre-commit
    tcc/.beads/backup
        labels.jsonl
        config.jsonl
        backup_state.json
        dependencies.jsonl
        issues.jsonl
        comments.jsonl
        events.jsonl
    tcc/docs
        architecture.md
    tcc/.git
        config
        packed-refs
        index
        HEAD
        description
        ORIG_HEAD
        FETCH_HEAD
    tcc/.git/refs
    tcc/.git/objects
    tcc/.git/info
        exclude
    tcc/.git/gk
        config
    tcc/.git/hooks
        fsmonitor-watchman.sample
        applypatch-msg.sample
        post-update.sample
        prepare-commit-msg.sample
        push-to-checkout.sample
        pre-merge-commit.sample
        pre-push.sample
        commit-msg.sample
        pre-commit.sample
        pre-receive.sample
    tcc/.git/branches
    tcc/.git/logs
        HEAD
    tcc/src
    tcc/src/tcc
        datasets.py
        stochastic_alignment.py
        losses.py
        models.py
        train.py
        deterministic_alignment.py
        __init__.py
        evaluate.py
        config.py
        alignment.py
    tcc/configs
        default.yaml
        demo.yaml
    tcc/results
    tcc/results/pca
        PCA trajectory (D=64, clearodwalla_to_clear0_real_view1).png
        PCA trajectory (D=64, clearodwalla_to_clear0_real_view0).png
        PCA trajectory (D=128, clearodwalla_to_clear0_real_view0).png
        PCA trajectory (D=32, clearodwalla_to_clear0_real_view1).png
        PCA trajectory (D=32, clearsoda_to_white0_real_view0).png
        PCA trajectory (D=64, clearsoda_to_white0_real_view0).png
        PCA trajectory (D=128, clearsoda_to_white0_real_view0).png
        PCA trajectory (D=32, clearodwalla_to_clear0_real_view0).png
        PCA trajectory (D=128, clearodwalla_to_clear0_real_view1).png
    tcc/results/scaled
    tcc/results/kmeans
    tcc/results/umap
        UMAP cross-video trajectories (joint projection) 2.png
        UMAP trajectory (D=128, clearodwalla_to_clear0_real_view1).png
        UMAP trajectory (D=128, clearodwalla_to_clear0_real_view0).png
        UMAP cross-video trajectories (joint projection) 3.png
        UMAP cross-video trajectories (joint projection) 1.png
        UMAP trajectory (D=32, clearodwalla_to_clear0_real_view1).png
        UMAP trajectory (D=128, clearsoda_to_white0_real_view0).png
        UMAP trajectory (D=64, clearodwalla_to_clear0_real_view0).png
        UMAP trajectory (D=64, clearsoda_to_white0_real_view0).png
        UMAP trajectory (D=64, clearodwalla_to_clear0_real_view1).png
    tcc/.devcontainer
        devcontainer.json
    tcc/notebooks
        notebook-database.yml
    tcc/notebooks/self-supervised
        tcc_pouring_tutorial_aegean_main-executed.ipynb
        tcc_pouring_tutorial_aegean_main.ipynb
        main.ipynb
        whiteorange_to_clear1_real.tfrecord
    tcc/data
    tcc/data/pouring_processed
    tcc/tests
        test_evaluation.py
        test_datasets.py
        test_dataset_preparation.py
        test_algos.py
        test_train.py
        test_losses.py
        test_models.py
        __init__.py
        test_config.py
    tcc/tests/__pycache__
        __init__.cpython-311.pyc


## 4. Data: the pouring dataset

The multiview pouring dataset is hosted on HuggingFace at [`sermanet/multiview-pouring`](https://huggingface.co/datasets/sermanet/multiview-pouring). It contains TFRecord files with multi-view video sequences of pouring tasks.

### Download from HuggingFace

Use `huggingface_hub` to download the dataset files. The code cell below clones the dataset repository into `data/pouring/`. This is the recommended approach — it downloads all TFRecord files and the recombination script needed for one split file.

### Expected directory layout

After download and conversion, the dataset root must have this structure:

```
data/pouring_processed/pouring/
├── train/
│   ├── video_001/
│   │   ├── frame_0000.png
│   │   ├── frame_0001.png
│   │   └── ...
│   ├── video_002/
│   │   └── ...
│   └── ...
└── val/
    ├── video_050/
    │   └── ...
    └── ...
```

Each video is a directory of sequentially numbered frames. The PyTorch `create_dataset` function expects this layout — it discovers videos by listing subdirectories under `train/` or `val/`, then loads frames in filename-sorted order.


```python
DATA_ROOT = pathlib.Path("data")
RAW_POURING_ROOT = DATA_ROOT / "pouring"
PROCESSED_POURING_ROOT = DATA_ROOT / "pouring_processed"

RAW_POURING_ROOT.mkdir(parents=True, exist_ok=True)
PROCESSED_POURING_ROOT.mkdir(parents=True, exist_ok=True)

print("Raw data dir:", RAW_POURING_ROOT.resolve())
print("Processed data dir:", PROCESSED_POURING_ROOT.resolve())
```

    Raw data dir: /workspaces/tcc/notebooks/self-supervised/data/pouring
    Processed data dir: /workspaces/tcc/notebooks/self-supervised/data/pouring_processed


### 4.1 Download from HuggingFace

The dataset is hosted at [`sermanet/multiview-pouring`](https://huggingface.co/datasets/sermanet/multiview-pouring) and contains TFRecord files organized into `train/`, `val/`, and `test/` splits.

Use `huggingface_hub.snapshot_download` to download the full dataset. This downloads all files (TFRecords, recombination scripts, README) into a local cache and returns the path. We then symlink or copy into our expected `data/pouring/` directory.

> **Note:** One test file (`whiteorange_to_clear1_real`) was split into two parts due to upload size limits. After downloading, run the provided shell script to recombine it. This only affects the test split — training and validation are ready to use immediately.


```python
# from huggingface_hub import snapshot_download

# # Download the full dataset from HuggingFace (progress bars suppressed for clean output)
# hf_cache_path = snapshot_download(
#     repo_id="sermanet/multiview-pouring",
#     repo_type="dataset",
#     local_dir=str(RAW_POURING_ROOT),
# )

# print("Dataset downloaded to:", hf_cache_path)

# # List what was downloaded
# for split_dir in sorted(RAW_POURING_ROOT.iterdir()):
#     if split_dir.is_dir() and not split_dir.name.startswith("."):
#         tfrecords = list(split_dir.glob("*.tfrecord*"))
#         print(f"  {split_dir.name}/: {len(tfrecords)} TFRecord file(s)")
```


```python
# Step 1: Download is handled above via huggingface_hub (cell 4.1)

# Step 2 (optional): Recombine the split test file
# Only needed if you plan to use the test split
# !bash {RAW_POURING_ROOT}/tfrecords/test/whiteorange_to_clear1_real_combining.sh

# Step 3: Convert TFRecords to image-folder layout
# In the dev container terminal:
  # python -m tcc.dataset_preparation.videos_to_dataset \
  #     --input-dir /workspaces/tcc/notebooks/self-supervised/data/pouring/videos/val \
  #     --file-pattern "*.mov" \
  #     --output-dir /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/val \
  #     --name pouring --fps 15 --width 224 --height 224
#
# On Colab, prefix with ! instead:
#   !python -m tcc.dataset_preparation.videos_to_dataset ...

print("After downloading from HuggingFace, convert the TFRecords to image folders.")
```

    After downloading from HuggingFace, convert the TFRecords to image folders.


### 4.2 Expected semantic phases

We will reason about pouring in terms of latent phases such as:

1. reach
2. grasp
3. lift / position
4. tilt
5. pour
6. retract / return

You do **not** need action labels for TCC training.  
These phase names are used only for qualitative interpretation of the learned representation.

## 5. Configuration and training

The current `aegean-ai/tcc` package provides:

- a typed configuration object
- an `alignment` algorithm corresponding to TCC
- a PyTorch training loop

The default configuration is useful to inspect first, because it tells us:

- training algorithm
- dataset name
- image size
- batch size
- embedding size
- checkpoint/logging schedule


```python
from pprint import pprint

try:
    from tcc.config import get_default_config
    cfg = get_default_config()
    print(cfg)
except Exception as e:
    print("Could not import tcc yet:", repr(e))
    cfg = None
```

    TCCConfig(logdir='/tmp/alignment_logs/', datasets=['pouring'], path_to_tfrecords='/tmp/%s_tfrecords/', training_algo='alignment', train=TrainConfig(max_iters=150000, batch_size=2, num_frames=20, visualize_interval=200), eval=EvalConfig(batch_size=2, num_frames=20, val_iters=20, tasks=['algo_loss', 'classification', 'kendalls_tau', 'event_completion', 'few_shot_classification'], frames_per_batch=25, kendalls_tau_stride=5, kendalls_tau_distance='sqeuclidean', classification_fractions=[0.1, 0.5, 1.0], few_shot_num_labeled=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], few_shot_num_episodes=50), model=ModelConfig(embedder_type='conv', base_model=BaseModelConfig(network='resnet50', layer='conv4_block3_out', train_base='only_bn'), conv_embedder=ConvEmbedderConfig(embedding_size=128, num_context_steps=2, conv_layers=[(256, 3, True), (256, 3, True)], fc_layers=[(256, True), (256, True)], capacity_scalar=2, flatten_method='max_pool', base_dropout_rate=0.0, base_dropout_spatial=False, fc_dropout_rate=0.1, dropout_rate=0.1, pooling='max', l2_normalize=False, use_bn=True), convgru_embedder=ConvGRUEmbedderConfig(conv_layers=[(512, 3, True), (512, 3, True)], gru_layers=[128], dropout_rate=0.0, use_bn=True), vggm=VGGMConfig(use_bn=True), train_embedding=True, l2_reg_weight=1e-05, resnet_pretrained_weights='/tmp/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5'), alignment=AlignmentConfig(loss_type='regression_mse_var', similarity_type='l2', temperature=0.1, label_smoothing=0.1, cycle_length=2, stochastic_matching=False, variance_lambda=0.001, huber_delta=0.1, normalize_indices=True, fraction=1.0), sal=SALConfig(dropout_rate=0.0, fc_layers=[(128, True), (64, True), (2, False)], shuffle_fraction=0.75, num_samples=8, label_smoothing=0.0), alignment_sal_tcn=AlignmentSalTcnConfig(alignment_loss_weight=0.33, sal_loss_weight=0.33), classification=ClassificationConfig(label_smoothing=0.0, dropout_rate=0.0), tcn=TCNConfig(positive_window=5, reg_lambda=0.002), optimizer=OptimizerConfig(type='adam', lr=LRConfig(initial_lr=0.0001, decay_type='fixed', exp_decay_rate=0.97, exp_decay_steps=1000, manual_lr_step_boundaries=[5000, 10000], manual_lr_decay_rate=0.1, num_warmup_steps=0), weight_decay=0.0), data=DataConfig(sampling_strategy='offset_uniform', stride=16, num_steps=2, frame_stride=15, image_size=224, augment=True, shuffle_queue_size=0, num_prefetch_batches=1, random_offset=1, frame_labels=True, per_dataset_fraction=1.0, per_class=False, sample_all_stride=1), augmentation=AugmentationConfig(random_flip=True, random_crop=False, brightness=True, brightness_max_delta=0.12549019607843137, contrast=True, contrast_lower=0.5, contrast_upper=1.5, hue=False, hue_max_delta=0.2, saturation=False, saturation_lower=0.5, saturation_upper=1.5), logging=LoggingConfig(report_interval=100), checkpoint=CheckpointConfig(save_interval=1000))


### 5.1 Utility: robust config editing

Research repositories evolve. Rather than assuming one exact config layout, we use helper functions that can set values safely if the corresponding fields exist.

This makes the notebook more resilient to small refactors of the dataclass hierarchy.


```python
def set_if_exists(obj, path, value):
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        if not hasattr(cur, p):
            return False
        cur = getattr(cur, p)
    if hasattr(cur, parts[-1]):
        setattr(cur, parts[-1], value)
        return True
    return False

def get_if_exists(obj, path, default=None):
    parts = path.split(".")
    cur = obj
    for p in parts:
        if not hasattr(cur, p):
            return default
        cur = getattr(cur, p)
    return cur

def summarize_config(cfg):
    keys = [
        "training_algo",
        "datasets",
        "path_to_tfrecords",
        "logdir",
        "train.batch_size",
        "train.max_iters",
        "train.num_frames",
        "eval.batch_size",
        "model.embedder_type",
        "model.conv_embedder.embedding_size",
        "model.base_model.train_base",
        "optimizer.type",
        "optimizer.lr.initial_lr",
        "data.image_size",
        "data.frame_stride",
        "data.num_steps",
    ]
    rows = []
    for k in keys:
        rows.append((k, get_if_exists(cfg, k)))
    return rows

if cfg is not None:
    for k, v in summarize_config(cfg):
        print(f"{k:40s} {v}")
```

    training_algo                            alignment
    datasets                                 ['pouring']
    path_to_tfrecords                        /tmp/%s_tfrecords/
    logdir                                   /tmp/alignment_logs/
    train.batch_size                         2
    train.max_iters                          150000
    train.num_frames                         20
    eval.batch_size                          2
    model.embedder_type                      conv
    model.conv_embedder.embedding_size       128
    model.base_model.train_base              only_bn
    optimizer.type                           adam
    optimizer.lr.initial_lr                  0.0001
    data.image_size                          224
    data.frame_stride                        15
    data.num_steps                           2


### 5.2 Choose experiment settings

The assignment requires an embedding-dimension sweep:

- 32
- 64
- 128

We keep everything else as close as possible to the repo defaults so that the experiment isolates the representation bottleneck dimension.


```python
EMBED_DIMS = [32, 64, 128]
EXPERIMENT_ROOT = pathlib.Path("runs_tutorial")
EXPERIMENT_ROOT.mkdir(exist_ok=True)

print("Experiments will be stored in:", EXPERIMENT_ROOT.resolve())
print("Embedding dims:", EMBED_DIMS)
```

    Experiments will be stored in: /workspaces/tcc/notebooks/self-supervised/runs_tutorial
    Embedding dims: [32, 64, 128]


### 5.3 Build a training config for one run

The training code in `src/tcc/train.py` expects a `TCCConfig`, and the default config already uses:

- `datasets: [pouring]`
- `training_algo: alignment`

We modify:

- embedding size
- log directory
- dataset root
- optionally `train.max_iters` for a shorter tutorial run


```python
def make_run_config(embed_dim=128, max_iters=2000, logdir=None):
    from tcc.config import get_default_config

    cfg = get_default_config()

    set_if_exists(cfg, "training_algo", "alignment")
    set_if_exists(cfg, "datasets", ["pouring"])
    set_if_exists(cfg, "train.max_iters", max_iters)
    set_if_exists(cfg, "model.conv_embedder.embedding_size", embed_dim)

    ''' 
        Models keep getting NaN throughout.  Training takes seconds.
        Changing loss function as options are avialable in losses.py.
        Let's try Huber?
    '''

    # Try Huber loss.
    set_if_exists(cfg, "alignment.loss_type", "regression_huber")

    ds_fmt = str((PROCESSED_POURING_ROOT / "%s").resolve())
    set_if_exists(cfg, "path_to_tfrecords", ds_fmt)

    if logdir is None:
        logdir = str((EXPERIMENT_ROOT / f"pouring_tcc_d{embed_dim}").resolve())
    set_if_exists(cfg, "logdir", logdir)

    return cfg

try:
    demo_cfg = make_run_config(embed_dim=64, max_iters=500)
    for k, v in summarize_config(demo_cfg):
        print(f"{k:40s} {v}")
except Exception as e:
    print("Config construction failed:", repr(e))
```

    training_algo                            alignment
    datasets                                 ['pouring']
    path_to_tfrecords                        /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/%s
    logdir                                   /workspaces/tcc/notebooks/self-supervised/runs_tutorial/pouring_tcc_d64
    train.batch_size                         2
    train.max_iters                          500
    train.num_frames                         20
    eval.batch_size                          2
    model.embedder_type                      conv
    model.conv_embedder.embedding_size       64
    model.base_model.train_base              only_bn
    optimizer.type                           adam
    optimizer.lr.initial_lr                  0.0001
    data.image_size                          224
    data.frame_stride                        15
    data.num_steps                           2


## 6. Training

The training loop in the repo is exposed through `tcc.train.train(cfg)`.

The logic is:

1. instantiate the algorithm corresponding to `cfg.training_algo`
2. build the dataset loader
3. optimize the alignment loss
4. save checkpoints in `cfg.logdir`


```python
'''
    Overnight the process stalled and never began training.  
    In documentation found online I found that Jupyter doesn't work well with num_workers >0 for data loaders.
    https://github.com/pytorch/pytorch/issues/51418
'''

# Check for workers available.
print(f"{os.environ.get('DATALOADER_WORKERS')}")
```

    None



```python
print(f"{cfg.logdir}")
```

    /tmp/alignment_logs/



```python
def run_training(cfg):
    from tcc.train import train
    print("Starting training with logdir:", cfg.logdir)
    train(cfg)

# Example debug run:
cfg_debug = make_run_config(embed_dim=32, max_iters=50)

# Testing
print(f"cfg_debug created.  Now running training.")

run_training(cfg_debug)

print("Uncomment the debug run once the dataset path is ready.")
```

    cfg_debug created.  Now running training.


    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1773590355.064968   11759 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    I0000 00:00:1773590355.108621   11759 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1773590358.000657   11759 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.


    Starting training with logdir: /workspaces/tcc/notebooks/self-supervised/runs_tutorial/pouring_tcc_d32
    Thisn
    cuda
    split_dir: /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/pouring/train
    Train Loader has: 0 workers.
    DataLoader: ['dataset', 'num_workers', 'prefetch_factor', 'pin_memory', 'pin_memory_device', 'timeout', 'worker_init_fn', '_DataLoader__multiprocessing_context', 'in_order', '_dataset_kind', 'batch_size', 'drop_last', 'sampler', 'batch_sampler', 'generator', 'collate_fn', 'persistent_workers', '_DataLoader__initialized', '_IterableDataset_len_called', '_iterator', '__module__', '__annotations__', '__doc__', '__init__', '_get_iterator', 'multiprocessing_context', '__setattr__', '__iter__', '_auto_collation', '_index_sampler', '__len__', 'check_worker_number_rationality', '__orig_bases__', '__dict__', '__weakref__', '__parameters__', '__slots__', '_is_protocol', '__class_getitem__', '__init_subclass__', '__new__', '__repr__', '__hash__', '__str__', '__getattribute__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__reduce_ex__', '__reduce__', '__getstate__', '__subclasshook__', '__format__', '__sizeof__', '__dir__', '__class__']
    Uncomment the debug run once the dataset path is ready.


### 6.1 Full assignment runs

Run three experiments:

- $D=32$
- $D=64$
- $D=128$


```python
for d in EMBED_DIMS:
    cfg_run = make_run_config(embed_dim=d, max_iters=5000)
    run_training(cfg_run)

print("Run the sweep above after verifying the debug run.")
```

    Starting training with logdir: /workspaces/tcc/notebooks/self-supervised/runs_tutorial/pouring_tcc_d32
    Thisn
    cuda
    split_dir: /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/pouring/train
    Train Loader has: 0 workers.
    DataLoader: ['dataset', 'num_workers', 'prefetch_factor', 'pin_memory', 'pin_memory_device', 'timeout', 'worker_init_fn', '_DataLoader__multiprocessing_context', 'in_order', '_dataset_kind', 'batch_size', 'drop_last', 'sampler', 'batch_sampler', 'generator', 'collate_fn', 'persistent_workers', '_DataLoader__initialized', '_IterableDataset_len_called', '_iterator', '__module__', '__annotations__', '__doc__', '__init__', '_get_iterator', 'multiprocessing_context', '__setattr__', '__iter__', '_auto_collation', '_index_sampler', '__len__', 'check_worker_number_rationality', '__orig_bases__', '__dict__', '__weakref__', '__parameters__', '__slots__', '_is_protocol', '__class_getitem__', '__init_subclass__', '__new__', '__repr__', '__hash__', '__str__', '__getattribute__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__reduce_ex__', '__reduce__', '__getstate__', '__subclasshook__', '__format__', '__sizeof__', '__dir__', '__class__']
    Starting training with logdir: /workspaces/tcc/notebooks/self-supervised/runs_tutorial/pouring_tcc_d64
    Thisn
    cuda
    split_dir: /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/pouring/train
    Train Loader has: 0 workers.
    DataLoader: ['dataset', 'num_workers', 'prefetch_factor', 'pin_memory', 'pin_memory_device', 'timeout', 'worker_init_fn', '_DataLoader__multiprocessing_context', 'in_order', '_dataset_kind', 'batch_size', 'drop_last', 'sampler', 'batch_sampler', 'generator', 'collate_fn', 'persistent_workers', '_DataLoader__initialized', '_IterableDataset_len_called', '_iterator', '__module__', '__annotations__', '__doc__', '__init__', '_get_iterator', 'multiprocessing_context', '__setattr__', '__iter__', '_auto_collation', '_index_sampler', '__len__', 'check_worker_number_rationality', '__orig_bases__', '__dict__', '__weakref__', '__parameters__', '__slots__', '_is_protocol', '__class_getitem__', '__init_subclass__', '__new__', '__repr__', '__hash__', '__str__', '__getattribute__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__reduce_ex__', '__reduce__', '__getstate__', '__subclasshook__', '__format__', '__sizeof__', '__dir__', '__class__']
    Starting training with logdir: /workspaces/tcc/notebooks/self-supervised/runs_tutorial/pouring_tcc_d128
    Thisn
    cuda
    split_dir: /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/pouring/train
    Train Loader has: 0 workers.
    DataLoader: ['dataset', 'num_workers', 'prefetch_factor', 'pin_memory', 'pin_memory_device', 'timeout', 'worker_init_fn', '_DataLoader__multiprocessing_context', 'in_order', '_dataset_kind', 'batch_size', 'drop_last', 'sampler', 'batch_sampler', 'generator', 'collate_fn', 'persistent_workers', '_DataLoader__initialized', '_IterableDataset_len_called', '_iterator', '__module__', '__annotations__', '__doc__', '__init__', '_get_iterator', 'multiprocessing_context', '__setattr__', '__iter__', '_auto_collation', '_index_sampler', '__len__', 'check_worker_number_rationality', '__orig_bases__', '__dict__', '__weakref__', '__parameters__', '__slots__', '_is_protocol', '__class_getitem__', '__init_subclass__', '__new__', '__repr__', '__hash__', '__str__', '__getattribute__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__reduce_ex__', '__reduce__', '__getstate__', '__subclasshook__', '__format__', '__sizeof__', '__dir__', '__class__']
    Run the sweep above after verifying the debug run.


## 7. Loading checkpoints and extracting embeddings

The repo provides the pieces we need:

- `get_algo(...)` to instantiate the TCC algorithm
- checkpoint loading utilities from `tcc.train`
- embedding extraction utilities from `tcc.evaluate`


```python
import torch
from pathlib import Path

def latest_checkpoint(logdir):
    candidates = sorted(Path(logdir).glob("checkpoint_*.pt"))
    if not candidates:
        return None
    return str(candidates[-1])

def load_trained_algo(cfg, checkpoint_path=None):
    from tcc.algos.registry import get_algo
    from tcc.train import load_checkpoint

    algo = get_algo(cfg.training_algo, cfg=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = algo.to(device)

    if checkpoint_path is None:
        checkpoint_path = latest_checkpoint(cfg.logdir)

    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {cfg.logdir}")

    _ = load_checkpoint(checkpoint_path, algo, optimizer=None)
    algo.eval()
    return algo, device, checkpoint_path
```

### 7.1 Build the evaluation dataloader

The repo training code internally converts the top-level config into a `DataConfig`.  
We reuse the same helper if available; otherwise we build the `DataConfig` manually.


```python
def make_eval_dataloader(cfg, split="val", mode="eval"):
    """Build an evaluation dataloader from the top-level config.

    Uses the repo's internal _build_data_config helper.  If that helper
    is missing or its signature has changed, the call fails loudly so
    you know the notebook and repo are out of sync.
    """
    from tcc.datasets import create_dataset

    try:
        from tcc.train import _build_data_config
    except ImportError:
        raise ImportError(
            "Cannot import _build_data_config from tcc.train. "
            "The aegean-ai/tcc repo API may have changed. "
            "Check the repo README for the current evaluation interface."
        )

    data_cfg = _build_data_config(cfg)
    loader = create_dataset(split=split, mode=mode, config=data_cfg)
    return loader
```


```python
def extract_embeddings_for_run(cfg, split="val", max_embs=0):
    """Extract embeddings from a trained checkpoint.

    Args:
        cfg: TCCConfig for the run.
        split: dataset split to evaluate ("val" or "train").
        max_embs: maximum number of video embeddings to extract.
                  0 means extract all available videos (no limit).
    """
    from tcc.evaluate import get_embeddings_dataset

    algo, device, checkpoint_path = load_trained_algo(cfg)
    loader = make_eval_dataloader(cfg, split=split, mode="eval")

    # TEsting
    print(f"data loader: {loader}")

    # Assuming 'data_loader' is your PyTorch DataLoader instance
    dataiter = iter(loader)
    batch = next(dataiter)

    # Dataloader is empty?
    print(len(loader.dataset))

    # 'batch' will typically be a list or tuple of tensors (e.g., [features, labels])
    # features, labels = batch

    # print("Features shape:", features.shape)
    # print("Labels shape:", labels.shape)
    # print("First few labels:", labels[:5])

    # Batch returned more than two objects.
    frames, frame_labels, seq_labels, seq_lens, names = batch

    print(f"Sequence Labels:  {seq_labels}")
    print(f"Sequence Lens:  {seq_lens}")
    print(f"Names:  {names}")
    print(f"Frames:  {len(frames)} frames")
    print(f"Frame Labels:  {frame_labels}")
    

    bundle = get_embeddings_dataset(algo, loader, device=device, max_embs=max_embs)

    # Testing
    # print(f"bundle: {bundle}")

    return bundle

# Example:
# cfg64 = make_run_config(embed_dim=64, max_iters=5000)
# emb_bundle = extract_embeddings_for_run(cfg64, split="val")
```


```python
# Create run configs for each dim.
cfg32 = make_run_config(embed_dim=32, max_iters=5000)
cfg64 = make_run_config(embed_dim=64, max_iters=5000)
cfg128 = make_run_config(embed_dim=128, max_iters=5000)

# Create dict to reference configs.
cfgs = {32:cfg32, 64:cfg64, 128:cfg128}

emb_bundle32 = extract_embeddings_for_run(cfg32, split="val")
emb_bundle64 = extract_embeddings_for_run(cfg64, split="val")
emb_bundle128 = extract_embeddings_for_run(cfg128, split="val")

# Create dict to reference embedding bundles.
emb_bundles = {32: emb_bundle32, 64: emb_bundle64, 128: emb_bundle128}
```

    split_dir: /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/pouring/val
    data loader: <torch.utils.data.dataloader.DataLoader object at 0x71df63c99d50>
    34
    Sequence Labels:  tensor([0, 0])
    Sequence Lens:  tensor([133, 165])
    Names:  ['clearodwalla_to_clear0_real_view0', 'clearodwalla_to_clear0_real_view1']
    Frames:  2 frames
    Frame Labels:  tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    split_dir: /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/pouring/val
    data loader: <torch.utils.data.dataloader.DataLoader object at 0x71df63d14590>
    34
    Sequence Labels:  tensor([0, 0])
    Sequence Lens:  tensor([133, 165])
    Names:  ['clearodwalla_to_clear0_real_view0', 'clearodwalla_to_clear0_real_view1']
    Frames:  2 frames
    Frame Labels:  tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    split_dir: /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/pouring/val
    data loader: <torch.utils.data.dataloader.DataLoader object at 0x71df68322e90>
    34
    Sequence Labels:  tensor([0, 0])
    Sequence Lens:  tensor([133, 165])
    Names:  ['clearodwalla_to_clear0_real_view0', 'clearodwalla_to_clear0_real_view1']
    Frames:  2 frames
    Frame Labels:  tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


## 8. Representation diagnostics

Now we test the main scientific claim:

> Do embeddings organize frames by **task phase**?

We use two projection methods and two diagnostic approaches:

1. **PCA** — linear projection preserving global variance; fast and deterministic
2. **UMAP** — nonlinear projection revealing manifold structure; better for fine-grained phase separation

For each, we produce:

- **single-video trajectory plots** colored by time
- **cross-video overlays** in a shared projection space (joint fit, so coordinates are comparable)


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def project_pca(Z):
    return PCA(n_components=2).fit_transform(Z)

def project_umap(Z, seed=0):
    if not HAS_UMAP:
        raise ImportError("Install umap-learn: pip install umap-learn")
    return umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)

def project(Z, method="umap", seed=0):
    if method == "pca":
        return project_pca(Z)
    if method == "umap":
        return project_umap(Z, seed=seed)
    raise ValueError(f"Unknown method: {method}. Use 'pca' or 'umap'.")
```


```python
def plot_single_trajectory(Z, method="umap", title=None):
    Y = project(Z, method=method)
    t = np.arange(len(Y))

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(Y[:, 0], Y[:, 1], c=t, s=12)
    plt.colorbar(sc, label="time")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.title(title or f"{method.upper()} trajectory")
    plt.tight_layout()
    plt.show()

def plot_multiple_trajectories(embeddings_list, names=None, method="umap", max_videos=6):
    """Plot cross-video trajectories in a shared projection space.

    All video embeddings are concatenated, projected once, then split
    back so that the 2D coordinates are comparable across videos.
    """
    n = min(max_videos, len(embeddings_list))
    selected = embeddings_list[:n]
    lengths = [len(Z) for Z in selected]
    Z_all = np.concatenate(selected, axis=0)

    Y_all = project(Z_all, method=method, seed=0)

    splits = np.cumsum(lengths[:-1])
    Y_per_video = np.split(Y_all, splits)

    plt.figure(figsize=(7, 6))
    for i, Y in enumerate(Y_per_video):
        label = names[i] if names else f"video_{i}"
        plt.plot(Y[:, 0], Y[:, 1], alpha=0.8, label=label)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.title(f"{method.upper()} cross-video trajectories (joint projection)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()
```


```python
# # Example usage:
# #
# emb_bundle = extract_embeddings_for_run(cfg64, split="val")
# Z0 = emb_bundle["embeddings_list"][0]
# plot_single_trajectory(Z0, method="pca", title="PCA trajectory")
# plot_single_trajectory(Z0, method="umap", title="UMAP trajectory")
# plot_multiple_trajectories(emb_bundle["embeddings_list"], emb_bundle["names"], method="umap")
```


```python
# Wrapper function to ease in running several plots.
def show_me(emb_bundle, dim):
    Z0 = emb_bundle["embeddings_list"][0]
    plot_single_trajectory(Z0, method="pca", title=f"PCA trajectory D={i}")
    plot_single_trajectory(Z0, method="umap", title=f"UMAP trajectory D={i}")
    plot_multiple_trajectories(emb_bundle["embeddings_list"], emb_bundle["names"], method="umap")

# Iterate through bundles and display.
for i in emb_bundles.keys():
    
    show_me(emb_bundles[i], i)


```


    
![png](../../README_files/../../README_41_0.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_41_2.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_41_4.png)
    



    
![png](../../README_files/../../README_41_5.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_41_7.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_41_9.png)
    



    
![png](../../README_files/../../README_41_10.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_41_12.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_41_14.png)
    


## 9. Temporal segmentation from embedding geometry

This section operationalizes the claim that the embedding has learned latent phase.

We use two complementary segmentation strategies:

### 9.1 Change-point detection

If the representation changes rapidly at phase transitions, then
$$
d_t = \|z_t - z_{t-1}\|
$$
should spike near boundaries. This is a **boundary-detection** approach — it finds *where* phase transitions occur without assigning cluster labels.

### 9.2 KMeans clustering in the native embedding space

If the embedding clusters by phase, KMeans should recover coarse phase labels. We use $k=6$ to match the six expected pouring phases (reach, grasp, lift, tilt, pour, retract). Experiment with different $k$ values to test sensitivity.


```python
from sklearn.cluster import KMeans

def change_point_scores(Z):
    d = np.linalg.norm(Z[1:] - Z[:-1], axis=1)
    d = np.concatenate([[0.0], d])
    return d

def detect_boundaries(d, threshold_quantile=0.98, min_gap=10):
    thr = float(np.quantile(d, threshold_quantile))
    idx = np.where(d >= thr)[0].tolist()

    kept = []
    last = -10**9
    for i in idx:
        if i - last >= min_gap:
            kept.append(i)
            last = i
    return kept, thr

def cluster_kmeans(Z, k=6, seed=0):
    """KMeans with k=6 matching the six expected pouring phases."""
    return KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(Z)
```


```python
def plot_segmentation(Z, labels=None, boundaries=None, title="Segmentation"):
    d = change_point_scores(Z)
    T = len(Z)

    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(T), d)
    if boundaries is not None:
        for b in boundaries:
            plt.axvline(b, linestyle="--")
    plt.title(title + " — change-point score")
    plt.xlabel("frame index")
    plt.ylabel(r"$\|z_t-z_{t-1}\|$")
    plt.tight_layout()
    plt.show()

    if labels is not None:
        plt.figure(figsize=(10, 2))
        plt.plot(np.arange(T), labels, drawstyle="steps-mid")
        if boundaries is not None:
            for b in boundaries:
                plt.axvline(b, linestyle="--")
        plt.title(title + " — cluster labels over time")
        plt.xlabel("frame index")
        plt.ylabel("cluster")
        plt.tight_layout()
        plt.show()
```


```python
# # Example usage:
# #
# Z = emb_bundle["embeddings_list"][0]
# d = change_point_scores(Z)
# boundaries, thr = detect_boundaries(d, threshold_quantile=0.98, min_gap=8)
# labels_km = cluster_kmeans(Z, k=6)

# plot_segmentation(Z, labels=labels_km, boundaries=boundaries, title="KMeans on native embeddings")
```


```python
for key in emb_bundles:
    
    # Get embed bundle.
    emb_bundle = emb_bundles[key]

    Z = emb_bundle["embeddings_list"][0]
    d = change_point_scores(Z)
    boundaries, thr = detect_boundaries(d, threshold_quantile=0.98, min_gap=8)
    labels_km = cluster_kmeans(Z, k=6)

    plot_segmentation(Z, labels=labels_km, boundaries=boundaries, title=f"KMeans on native embeddings  D={key}")

```


    
![png](../../README_files/../../README_46_0.png)
    



    
![png](../../README_files/../../README_46_1.png)
    



    
![png](../../README_files/../../README_46_2.png)
    



    
![png](../../README_files/../../README_46_3.png)
    



    
![png](../../README_files/../../README_46_4.png)
    



    
![png](../../README_files/../../README_46_5.png)
    


## 10. Embedding dimension sweep

The assignment asks you to compare:

- $D=32$
- $D=64$
- $D=128$

This matters because the embedding dimension controls the trade-off between:

- **compression**
- **expressiveness**
- **ease of clustering**
- **risk of overfitting appearance rather than phase**


```python
def run_full_analysis_for_dimension(embed_dim, max_iters=5000, split="val",
                                    max_embs=0, num_videos_to_analyze=3):
    """Run the complete analysis pipeline for one embedding dimension.

    Analyzes up to num_videos_to_analyze videos (not just the first one)
    to ensure results are not artifacts of a single video.
    """
    cfg = make_run_config(embed_dim=embed_dim, max_iters=max_iters)
    bundle = extract_embeddings_for_run(cfg, split=split, max_embs=max_embs)

    print(f"\n===== Dimension {embed_dim} =====")
    print("Number of videos:", len(bundle["embeddings_list"]))
    print("Flat embedding matrix:", bundle["embeddings"].shape)

    if len(bundle["embeddings_list"]) == 0:
        print("No embeddings found.")
        return cfg, bundle

    n_analyze = min(num_videos_to_analyze, len(bundle["embeddings_list"]))

    for vid_idx in range(n_analyze):
        Z = bundle["embeddings_list"][vid_idx]
        vid_name = bundle["names"][vid_idx] if "names" in bundle else f"video_{vid_idx}"
        tag = f"D={embed_dim}, {vid_name}"

        plot_single_trajectory(Z, method="pca", title=f"PCA trajectory ({tag})")
        if HAS_UMAP:
            plot_single_trajectory(Z, method="umap", title=f"UMAP trajectory ({tag})")

        d = change_point_scores(Z)
        boundaries, thr = detect_boundaries(d, threshold_quantile=0.98, min_gap=8)

        # Vary k
        for k in range(6,9,1):
            labels_km = cluster_kmeans(Z, k)
            plot_segmentation(Z, labels=labels_km, boundaries=boundaries,
                          title=f"KMeans k={k} ({tag})")

        # labels_km = cluster_kmeans(Z, k=6)
        # plot_segmentation(Z, labels=labels_km, boundaries=boundaries,
        #                   title=f"KMeans k=6 ({tag})")

    # Cross-video overlay (joint projection)
    if HAS_UMAP and len(bundle["embeddings_list"]) > 1:
        plot_multiple_trajectories(bundle["embeddings_list"], bundle.get("names"),
                                   method="umap")

    return cfg, bundle

# Example:
# cfg64, bundle64 = run_full_analysis_for_dimension(64, max_iters=5000)
```


```python
# Sweep through dimensions and k.
cfg32, bundle32 = run_full_analysis_for_dimension(32, max_iters=5000)
cfg64, bundle64 = run_full_analysis_for_dimension(64, max_iters=5000)
cfg128, bundle128 = run_full_analysis_for_dimension(128, max_iters=5000)

```

    split_dir: /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/pouring/val
    data loader: <torch.utils.data.dataloader.DataLoader object at 0x71df1015f290>
    34
    Sequence Labels:  tensor([0, 0])
    Sequence Lens:  tensor([133, 165])
    Names:  ['clearodwalla_to_clear0_real_view0', 'clearodwalla_to_clear0_real_view1']
    Frames:  2 frames
    Frame Labels:  tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    ===== Dimension 32 =====
    Number of videos: 34
    Flat embedding matrix: (680, 128)



    
![png](../../README_files/../../README_49_1.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_3.png)
    



    
![png](../../README_files/../../README_49_4.png)
    



    
![png](../../README_files/../../README_49_5.png)
    



    
![png](../../README_files/../../README_49_6.png)
    



    
![png](../../README_files/../../README_49_7.png)
    



    
![png](../../README_files/../../README_49_8.png)
    



    
![png](../../README_files/../../README_49_9.png)
    



    
![png](../../README_files/../../README_49_10.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_12.png)
    



    
![png](../../README_files/../../README_49_13.png)
    



    
![png](../../README_files/../../README_49_14.png)
    



    
![png](../../README_files/../../README_49_15.png)
    



    
![png](../../README_files/../../README_49_16.png)
    



    
![png](../../README_files/../../README_49_17.png)
    



    
![png](../../README_files/../../README_49_18.png)
    



    
![png](../../README_files/../../README_49_19.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_21.png)
    



    
![png](../../README_files/../../README_49_22.png)
    



    
![png](../../README_files/../../README_49_23.png)
    



    
![png](../../README_files/../../README_49_24.png)
    



    
![png](../../README_files/../../README_49_25.png)
    



    
![png](../../README_files/../../README_49_26.png)
    



    
![png](../../README_files/../../README_49_27.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_29.png)
    


    split_dir: /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/pouring/val
    data loader: <torch.utils.data.dataloader.DataLoader object at 0x71df5fcf45d0>
    34
    Sequence Labels:  tensor([0, 0])
    Sequence Lens:  tensor([133, 165])
    Names:  ['clearodwalla_to_clear0_real_view0', 'clearodwalla_to_clear0_real_view1']
    Frames:  2 frames
    Frame Labels:  tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    ===== Dimension 64 =====
    Number of videos: 34
    Flat embedding matrix: (680, 128)



    
![png](../../README_files/../../README_49_31.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_33.png)
    



    
![png](../../README_files/../../README_49_34.png)
    



    
![png](../../README_files/../../README_49_35.png)
    



    
![png](../../README_files/../../README_49_36.png)
    



    
![png](../../README_files/../../README_49_37.png)
    



    
![png](../../README_files/../../README_49_38.png)
    



    
![png](../../README_files/../../README_49_39.png)
    



    
![png](../../README_files/../../README_49_40.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_42.png)
    



    
![png](../../README_files/../../README_49_43.png)
    



    
![png](../../README_files/../../README_49_44.png)
    



    
![png](../../README_files/../../README_49_45.png)
    



    
![png](../../README_files/../../README_49_46.png)
    



    
![png](../../README_files/../../README_49_47.png)
    



    
![png](../../README_files/../../README_49_48.png)
    



    
![png](../../README_files/../../README_49_49.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_51.png)
    



    
![png](../../README_files/../../README_49_52.png)
    



    
![png](../../README_files/../../README_49_53.png)
    



    
![png](../../README_files/../../README_49_54.png)
    



    
![png](../../README_files/../../README_49_55.png)
    



    
![png](../../README_files/../../README_49_56.png)
    



    
![png](../../README_files/../../README_49_57.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_59.png)
    


    split_dir: /workspaces/tcc/notebooks/self-supervised/data/pouring_processed/pouring/val
    data loader: <torch.utils.data.dataloader.DataLoader object at 0x71df107e7850>
    34
    Sequence Labels:  tensor([0, 0])
    Sequence Lens:  tensor([133, 165])
    Names:  ['clearodwalla_to_clear0_real_view0', 'clearodwalla_to_clear0_real_view1']
    Frames:  2 frames
    Frame Labels:  tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    ===== Dimension 128 =====
    Number of videos: 34
    Flat embedding matrix: (680, 128)



    
![png](../../README_files/../../README_49_61.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_63.png)
    



    
![png](../../README_files/../../README_49_64.png)
    



    
![png](../../README_files/../../README_49_65.png)
    



    
![png](../../README_files/../../README_49_66.png)
    



    
![png](../../README_files/../../README_49_67.png)
    



    
![png](../../README_files/../../README_49_68.png)
    



    
![png](../../README_files/../../README_49_69.png)
    



    
![png](../../README_files/../../README_49_70.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_72.png)
    



    
![png](../../README_files/../../README_49_73.png)
    



    
![png](../../README_files/../../README_49_74.png)
    



    
![png](../../README_files/../../README_49_75.png)
    



    
![png](../../README_files/../../README_49_76.png)
    



    
![png](../../README_files/../../README_49_77.png)
    



    
![png](../../README_files/../../README_49_78.png)
    



    
![png](../../README_files/../../README_49_79.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_81.png)
    



    
![png](../../README_files/../../README_49_82.png)
    



    
![png](../../README_files/../../README_49_83.png)
    



    
![png](../../README_files/../../README_49_84.png)
    



    
![png](../../README_files/../../README_49_85.png)
    



    
![png](../../README_files/../../README_49_86.png)
    



    
![png](../../README_files/../../README_49_87.png)
    


    /workspaces/tcc/.venv/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(



    
![png](../../README_files/../../README_49_89.png)
    


## 11. Write-up questions

### Q1. TCN vs TCC
Explain, in your own words, the evolution from TCN to TCC. Include the role of the soft nearest-neighbor formulation in making cycle consistency differentiable.

### Q2. Does the learned representation encode phase?
Use your PCA and UMAP plots to justify a claim. Compare single-video trajectories with cross-video overlays.

### Q3. How well does segmentation recover phase structure?
Compare change-point detection and KMeans clustering. Do the detected boundaries align with qualitative phase transitions? Does varying $k$ change the story?

### Q4. What failure modes remain?
Examples:

- appearance variation dominating phase
- pauses causing over-segmentation
- self-similar frames across non-adjacent stages
- collapse of distinct phases into one cluster

### Q1. TCN vs TCC
Explain, in your own words, the evolution from TCN to TCC. Include the role of the soft nearest-neighbor formulation in making cycle consistency differentiable.

	Time-Contrastive Networks (TCN) learns embeddings such that synchronized frames from different views become neighbors in feature space.  That is to say that the use cases it covers are those where recordings are of the same event from different vantage points.  So, the events occurring in the recordings are temporally synchronized.  Any labels we ascribe to one recording can be mapped onto the second, because they are recording the same events at the same time but from different views.
	This means that TCN cannot learn from any recordings which are out of sync.  We cannot use the embeddings from this to analyze other videos, unless those videos are semantically in phase with the training recording.  If they do the exact same things in the exact same order in the exact same time frame we could use our embeddings to label the video.  It also utilizes a loss function which uses a max() selector function.  As this is not differentiable TCN cannot use methods of learning like Stochastic Gradient Decent which require differentiation.
	Temporal Cycle-Consistency Learning (TCC) abandons the assumption of synchronized events.  Instead it learns to map from one scene in a recording to another, and back again.  The loss is calculated between the back-mapped distribution and the original target sequence.  In other words, TCC takes a scene A in a video V, maps it to a scene  B in another recording W regardless of where that scene is in the recording.  It then tries to map B back to A.  By comparing how close in V we are to scene A in the back mapping we compute the loss.  
	By using Soft Nearest Neighbor (SNN) as a loss function we perform two tasks.  First, this function is differentiable, and therefore using it allows us to use techniques like SGD that require differentiable loss functions.  Second, it drives our embedding space such that similar scenes are embedded close together in the embedding space, and dissimilar frames are embedded farther apart. 

### Q2. Does the learned representation encode phase?
Use your PCA and UMAP plots to justify a claim. Compare single-video trajectories with cross-video overlays.

PCA
	The PCA plots show a shift over time in a clear direction between component 1 (C1) and component 2 (C2).  
	For 32 dimensions clearodwalla_to_clear0_real_view0 shows a trajectory over time from bottom right (high C1 and low C2) to upper left (high C2 and low C1).  The other perspective clearodwalla_to_clear0_real_view1 shows a seasawing beginning from left (low C1 and middling C2) to right (high C1 and middling C2) and then back to the middle (zero C1 and zero C2).  Other 32 dim plots show a similar trend of the embeddings shifting over time from low C1 to high C1 clustered around 0 C2.
	64 dimensions showed similar behavior over time, but over a greater spread.  The highs and lows of each component seem to extend as the dimensionality of the embeddings increases.
	
UMAP
	The single-video trajectory UMAP plots appear to show a very even spread of plots over the scaled axes.  This indicates a lack of identified structure.  However, one pattern seen is that the plots appear to be pushed to the edges of the graph, though still spread.  This may indicate a structure favoring one component over the other overall.
	The cross-video trajectory UMAP plots show clusters of events with long connecting streaks between them.  These clusters represent semantic events, and the connection streaks indicate transitions between these events.  The transitions appear to be consistent between the semantic events.
	This indicates that the embeddings have properly found the correct events, and they transition similarly between them despite being out of temporal phase.
	The phase structure was found.  The learned embeddings encoded the phase in the embeddings.

PCA Support Images

[text](results/scaled/pca) ![text](<results/scaled/pca/PCA trajectory (D=32, clearodwalla_to_clear0_real_view0).png>) ![text](<results/scaled/pca/PCA trajectory (D=32, clearodwalla_to_clear0_real_view1).png>) ![text](<results/scaled/pca/PCA trajectory (D=32, clearsoda_to_white0_real_view0).png>) ![text](<results/scaled/pca/PCA trajectory (D=64, clearodwalla_to_clear0_real_view0).png>) ![text](<results/scaled/pca/PCA trajectory (D=64, clearodwalla_to_clear0_real_view1).png>) ![text](<results/scaled/pca/PCA trajectory (D=64, clearsoda_to_white0_real_view0).png>) ![text](<results/scaled/pca/PCA trajectory (D=128, clearodwalla_to_clear0_real_view0).png>) ![text](<results/scaled/pca/PCA trajectory (D=128, clearodwalla_to_clear0_real_view1).png>) ![text](<results/scaled/pca/PCA trajectory (D=128, clearsoda_to_white0_real_view0).png>)

UMAP Support Images

[text](results/scaled/pca) [text](results/scaled/umap) ![text](<results/scaled/umap/UMAP cross-video trajectories (joint projection) 1.png>) ![text](<results/scaled/umap/UMAP cross-video trajectories (joint projection) 2.png>) ![text](<results/scaled/umap/UMAP cross-video trajectories (joint projection) 3.png>) ![text](<results/scaled/umap/UMAP trajectory (D=32, clearodwalla_to_clear0_real_view0).png>) ![text](<results/scaled/umap/UMAP trajectory (D=32, clearodwalla_to_clear0_real_view1).png>) ![text](<results/scaled/umap/UMAP trajectory (D=32, clearsoda_to_white0_real_view0).png>) ![text](<results/scaled/umap/UMAP trajectory (D=64, clearodwalla_to_clear0_real_view0).png>) ![text](<results/scaled/umap/UMAP trajectory (D=64, clearodwalla_to_clear0_real_view1).png>) ![text](<results/scaled/umap/UMAP trajectory (D=64, clearsoda_to_white0_real_view0).png>) ![text](<results/scaled/umap/UMAP trajectory (D=128, clearodwalla_to_clear0_real_view0).png>) ![text](<results/scaled/umap/UMAP trajectory (D=128, clearodwalla_to_clear0_real_view1).png>) ![text](<results/scaled/umap/UMAP trajectory (D=128, clearsoda_to_white0_real_view0).png>)

### Q3. How well does segmentation recover phase structure?
Compare change-point detection and KMeans clustering. Do the detected boundaries align with qualitative phase transitions? Does varying $k$ change the story?
	The detected boundaries seem to align with the qualitative phase transitions even when they are out of temporal phase.  Varying K through 6, 7, 8 did not significantly improve identification of semantic scene change.  However, increasing dimensionality did.  
	This may be because there were only 6 distinct clusters, thus six distinct semantic scene transitions.  I cannot seem to work out how to insert images here correctly.  I apologize for that oversight.  Though, the plots are here in the notebook, and I do not have the time to make the presentation look as it should.
	It appears that even unlabeled semi-supervised learning can and does recover the semantic structure of the video events.  Changing the k in this case does not appear to significantly alter the results, though increasing dimensionality does.  The higher dimension embedding plots seem to capture more of the semantic scene change than lower dimensional embeddings.  Again, the lack of response to alteration of k may mean that the process is more sensitive to dimensionality than k.  Though this may be due to the actual number of separate semantic events in the video.

	Supporting Images

	View 0

![alt text](<results/compare/view0/k6_32_view0.png>) 
![alt text](<results/compare/view0/k6_64_view0.png>)
![alt text](<results/compare/view0/k6_128_view0.png>)
![alt text](<results/compare/view0/k7_64_view0.png>)
![alt text](<results/compare/view0/k8_64_view0.png>)


	View 1

![alt text](results/compare/view1/k6_32_view1.png) ![alt text](results/compare/view1/k6_64_view1.png) ![alt text](results/compare/view1/k7_64_view1.png) ![alt text](results/compare/view1/k8_64_view1.png) ![alt text](results/compare/view1/k8_128_view1.png) [alt text](tcc_pouring_tutorial_aegean_main-executed.ipynb)


### Q4. What failure modes remain?
Depending on what dimensionality and K segmentation hyperparams we use we will see different failure modes.  The TCC seems to be able to properly segment the semantic events, though the clips are short.  The data strongly points to K matching the actual semantic event transition count as a success predictor.  Higher dimensionality lends to better segmentation identification.
With too low dimensionality we would see scenes bleed together.  With too low K we would see similar identification of the scenes as one instead of two.  With too high of k we may frature existing scenes into subscenes.  With too high dimensionality we may see the same.  Though it appears that the higher dimensionality with an appropriate K would combine the granularity provvided from the higher dims into appropriate scene labels with a good K.

We only tested on a few videos, so this is most likely not extensible to other scenes.  Many more scenes of similar semantic action should be used to train more.  With more time for training we could run more epochs and get even better embeddings.  Better embeddings would lead to better semantic identification and class labeling.


## 12. Final checklist

Before submitting, verify that you have:

- trained at least one real TCC run on pouring
- extracted embeddings from a saved checkpoint
- produced PCA and UMAP trajectory plots for multiple videos
- produced a cross-video overlay using joint projection
- run both change-point detection and KMeans segmentation
- compared $D=32,64,128$
- written answers to all four questions (Q1–Q4) in inline markdown cells, with all supporting figures embedded in the notebook

That completes the assignment.
