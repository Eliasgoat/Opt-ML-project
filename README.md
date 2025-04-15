# Opt-ML-project# Optimization Algorithms Benchmarking

This project provides a modular and extensible framework to benchmark various optimization algorithms on multiple deep learning tasks. Our primary objective is to assess the **efficiency** and **accuracy** of optimizers across datasets and model architectures, focusing especially on custom and classical gradient-based methods.

📝 Full methodology and experimental protocol are detailed in the Overleaf report: [Optimization Benchmark Report](https://www.overleaf.com/read/qgyqdcwdnzsx#3a32ed)

---

## 🔧 Project Structure

- `structure.py`: Core logic for loading datasets, initializing models, defining optimizers/schedulers, training and evaluating.
- `visualization.py`: (WIP) Will contain tools for analyzing and visualizing experiment results.
- `Data/`: All experiment results (in `.json` format) are saved here.
- `Results/`: All generated plots and visual figures will be saved here.

---

## 🚀 Quick Start: Run a Full Experiment

We provide utility functions to simplify the setup of model, optimizer, and scheduler configurations.

### Useful Config Builders

- `make_model_config(name, **params)`: describes a model and its hyperparameters.
- `make_optimizer_config(name, **params)`: defines the optimizer type and settings.
- `make_scheduler_config(type, **params)`: (optional) sets up learning rate scheduling.
- `build_optimizer_param_sweep(name, **param_lists)`: lets you scan several values (e.g., different learning rates) for a given optimizer.

### 🔁 Example: Run MLP on Several Datasets with Different Learning Rates (SGD)

```python
from structure import run_experiments, make_model_config, build_optimizer_param_sweep

datasets = ["MNIST", "FashionMNIST", "CIFAR10"]
models = [make_model_config("MLP", input_size=784, hidden_sizes=[128,64], num_classes=10)]
optimizers = build_optimizer_param_sweep("SGD", lr=[0.001, 0.01, 0.1])

run_experiments(
    datasets=datasets,
    models=models,
    optimizers_with_params=optimizers,
    save_results=True,
    save_path="Data/mlp_sgd_lr_scan.json"
)
```

This runs 9 experiments (3 datasets × 3 learning rates), each training an MLP with SGD and logs results into `Data/mlp_sgd_lr_scan.json`.

---

## 🛠️ Extend the Framework

### ➕ Add a New Dataset

1. Edit `get_dataset()` in `structure.py`.
2. Add a new entry using `torchvision.datasets` or a custom loader.
3. Return train/test `DataLoader` instances.

### ➕ Add a New Model

1. Create a new `nn.Module` subclass.
2. Register your model inside `get_model()`.

### ➕ Add a New Optimizer

1. Subclass `torch.optim.Optimizer` with your custom logic.
2. Add it inside `get_optimizer()`.
3. Optionally define projection constraints using `projection_factory` utilities.

---

## 📊 Visualizing Results (WIP)

The upcoming `visualization.py` module will allow:

- Loss and accuracy plotting over epochs
- Optimizer comparisons with error bars
- Best configuration reporting per dataset

All plots will be saved into the `Results/` directory and be compatible with LaTeX figures.

---

## 🧪 Reproducibility

We ensure reproducibility by setting global seeds:

```python
from structure import set_seed
set_seed(42)
```

---

## 📁 Output Structure

- JSON results → `Data/`
- PNG/SVG plots → `Results/`

---

## 📬 Contact

For contributions or questions, feel free to open an issue or submit a pull request!
