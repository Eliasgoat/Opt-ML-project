# How Optimizers Generalize: A Study on Non-Convex Architectures

This repository contains the code, experiments, and visualizations for our mini-project in the course **Optimization for Machine Learning (EPFL, Spring 2025)**. We investigate how different optimization algorithms behave on non-convex deep learning models, across two standard datasets (CIFAR10, SVHN) and two architectures (MLP, CNN).

## 🔍 Objective

We study the convergence and generalization performance of the following optimizers:
- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Momentum
- Nesterov Accelerated Gradient
- Adam
- RMSProp
- Projected Gradient Descent (PGD)
- Partial Gradient Descent (PartialGD)

## 📁 Repository Structure

```

.
├── CNN/                            # Scripts for running CNN experiments
├── MLP/                            # Scripts for running MLP experiments
├── data/                           # All raw .json files from experimental runs
├── PDF/                            # Final report PDF: 'How Optimizers Generalize...'
├── Results/                        # All result plots (.png) are saved here
├── structure.py                    # Main logic: model definitions, training loop, optimizers
├── visualization.py               # Functions for loading results and generating plots
├── \*.ipynb                         # Analysis notebooks: loads data from `data/` and generates plots
└── README.md                       # This file

````

## ⚙️ How to Run Experiments

### 1. Install dependencies
This project uses Python 3 and `torch`, `torchvision`, `numpy`, `matplotlib`. Install via:

```bash
pip install torch torchvision matplotlib numpy
````

### 2. Launch experiments

Experiments are launched using `run_experiments()` from `structure.py`.

Example (inside a script in `CNN/` or `MLP/`):

```python
from structure import run_experiments, make_model_config, build_optimizer_param_sweep

models = [make_model_config("CNN", input_channels=3, ...)]
optimizers = build_optimizer_param_sweep("SGD", lr=[0.01, 0.001])
run_experiments(
    datasets=["CIFAR10"],
    models=models,
    optimizers_with_params=optimizers,
    epochs=30,
    save_results=True,
    save_path="data/cifar10_cnn_sgd.json"
)
```

You can vary the dataset, optimizer, architecture, learning rate and more. Models and optimizers are specified using dictionaries.

### 3. Visualize results

Use `visualization.py` to generate all figures.

#### Plot all loss curves:

```python
from visualization import load_results, plot_losses
results = load_results("cifar10_cnn_sgd")
plot_losses(results)
```

#### Plot grouped metrics:

```python
from visualization import plot_metrics_vs_param_grouped
plot_metrics_vs_param_grouped(
    results,
    x_param="optimizer_params.lr",
    metrics=["accuracies_avg"],
    group_by="optimizer",
    split_by="model",
    save_path="Results/acc_vs_lr"
)
```

## 📊 Notebooks

All `.ipynb` notebooks at the root level are used to:

* Load experimental results from `data/`
* Generate final plots
* Export plots to `Results/`
* These were used to produce the figures in our final paper

## 📄 Final Report

The paper is located in: [How Optimizers Generalize A Study on Non-Convex Architectures.pdf](PDF/How%20Optimizers%20Generalize%20A%20Study%20on%20Non-Convex%20Architectures.pdf)



It summarizes:

* Theoretical background and motivation
* Experimental protocol
* Test loss evolution curves
* Final test accuracies
* Analysis and insights

## 🧠 Authors

This project was conducted as part of the **Optimization for Machine Learning** course at EPFL.

* Mayeul Cassier
* Elias Mir
* Sacha Frankhauser


## 🔗 References

See the final paper (section "References") for a full list of theoretical and empirical studies we built upon.
