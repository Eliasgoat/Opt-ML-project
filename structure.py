import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import json
import time
import tqdm
import pandas as pd


# === Seed setting ==
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# === Model ===
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_channels, conv_layers_config, fc_layers, activation="relu", use_maxpool=True, input_size=(32, 32)):
        super(CNN, self).__init__()
        self.activation_fn = getattr(torch.nn.functional, activation)
        self.use_maxpool = use_maxpool

        layers = []
        in_channels = input_channels

        # --- Build convolutional layers ---
        for layer_conf in conv_layers_config:
            layers.append(nn.Conv2d(in_channels, layer_conf["out_channels"], kernel_size=layer_conf["kernel_size"], padding=1))
            layers.append(nn.BatchNorm2d(layer_conf["out_channels"]))
            layers.append(nn.ReLU(inplace=True))  # ou autre activation

            if use_maxpool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = layer_conf["out_channels"]

        self.conv = nn.Sequential(*layers)

        # --- Determine the output shape after conv ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *input_size)
            conv_output = self.conv(dummy_input)
            conv_output_size = conv_output.view(1, -1).shape[1]  # nombre de features après flatten

        # --- Build fully connected layers ---
        fc = []
        in_features = conv_output_size
        for out_features in fc_layers:
            fc.append(nn.Linear(in_features, out_features))
            fc.append(nn.ReLU(inplace=True))
            in_features = out_features

        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_model(name, **model_params):
    """
    Returns a model based on the name and parameters.

    model_params: dictionary containing the appropriate arguments for the chosen model.
    """
    if name == "MLP":
        return MLP(**model_params)
    elif name == "CNN":
        return CNN(**model_params)
    else:
        raise ValueError(f"Model {name} not supported yet.")
    









# === Dataset ===
def get_dataset(name, batch_size=64):
    """
    Loads a dataset from torchvision.

    Arguments:
    - name: name of the dataset ("MNIST", "FashionMNIST", "KMNIST", "EMNIST_BALANCED", "CIFAR10", "CIFAR100", "SVHN")
    - batch_size: batch size for the dataloaders

    Returns: train_loader, test_loader
    """
    if name in ["MNIST", "FashionMNIST", "KMNIST", "EMNIST_BALANCED"]:
        transform = transforms.ToTensor()

    if name == "MNIST":
        train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    elif name == "FashionMNIST":
        train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    elif name == "EMNIST_BALANCED":
        train = datasets.EMNIST(root="./data", split="balanced", train=True, download=True, transform=transform)
        test = datasets.EMNIST(root="./data", split="balanced", train=False, download=True, transform=transform)

    elif name == "SVHN":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train = datasets.SVHN(root="./data", split="train", download=True, transform=transform)
        test = datasets.SVHN(root="./data", split="test", download=True, transform=transform)

    elif name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    elif name == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        test = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Dataset '{name}' not supported. Try: MNIST, FashionMNIST, KMNIST, EMNIST_BALANCED, CIFAR10, CIFAR100, SVHN")

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False)
    )












# === Custom Optimizer Example ===
from torch.optim import Optimizer
class MySGD(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = {'lr': lr}
        super(MySGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param -= group['lr'] * param.grad
                    
# === Custom Optimizers ===
class GradientDescent(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p -= group['lr'] * p.grad


class PartialGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, update_fraction=0.5):
        defaults = {'lr': lr, 'update_fraction': update_fraction}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            n = len(group['params'])
            k = max(1, int(group['update_fraction'] * n))
            to_update = random.sample(group['params'], k)
            for p in to_update:
                if p.grad is not None:
                    p -= group['lr'] * p.grad

class Momentum(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = {'lr': lr, 'momentum': momentum}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['velocity'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                v = self.state[p]['velocity']
                v.mul_(group['momentum']).add_(p.grad, alpha=-group['lr'])
                p.add_(v)

class Nesterov(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = {'lr': lr, 'momentum': momentum}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['velocity'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                v = self.state[p]['velocity']
                prev_v = v.clone()
                v.mul_(group['momentum']).add_(p.grad, alpha=-group['lr'])
                p.add_(-group['momentum'] * prev_v + (1 + group['momentum']) * v)

class NewtonOptimizer(Optimizer):
    """
    Naive implementation of a Newton's method.
    Uses the inverse of the Hessian to perform: x ← x - H⁻¹g

    ⚠️: very expensive, do not use with large models.
    """
    def __init__(self, params, lr=1.0, epsilon=1e-4):
        defaults = {"lr": lr, "epsilon": epsilon}
        super().__init__(params, defaults)

    # @torch.no_grad()
    def step(self, closure):
        """
        closure() doit recalculer la loss (et ses gradients).
        """
        loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # 🔁 Recalcule le gradient AVEC graphe
                grad = torch.autograd.grad(loss, p, create_graph=True, retain_graph=True)[0].view(-1)

                H_rows = []
                for i in range(grad.numel()):
                    grad_i = torch.autograd.grad(grad[i], p, retain_graph=True)[0].view(-1)
                    H_rows.append(grad_i)
                H = torch.stack(H_rows)

                # 🔽 Résolution de Hx = g
                try:
                    delta = torch.linalg.solve(H + eps * torch.eye(H.size(0), device=H.device), grad)
                    p.data -= lr * delta.view(p.shape)
                except RuntimeError:
                    print("⚠️ Hessian inversion failed — skipping Newton step")

        return loss


                   
class ProjectedGradientDescent(Optimizer):
    """
    PGD-type optimizer: performs gradient descent, then projects the weights into a constrained space.

    Arguments:
    - lr: learning rate.
    - projection: function to apply to the weights after the update. Default: clamp between -1 and 1.
    For example, to project all weights to [0, 1], use: projection=lambda x: x.clamp(0, 1)
    """
    def __init__(self, params, lr=0.01, projection=lambda x: x.clamp(-1, 1)):
        defaults = {'lr': lr, 'projection': projection}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            proj = group['projection']
            for p in group['params']:
                if p.grad is not None:
                    p -= lr * p.grad
                    p.copy_(proj(p))

# === Projection Utils ===


def make_clip_projection(min_val=-1.0, max_val=1.0):
    """
    Projects each weight individually into [min_val, max_val]
    """
    return lambda x: x.clamp(min_val, max_val)

def make_l2_projection(max_norm=5.0):
    """
    Projects all weights onto an L2-norm ball with radius <= max_norm
    """
    def proj(x):
        norm = x.norm()
        return x if norm <= max_norm else x * (max_norm / norm)
    return proj

def make_l1_projection(max_norm=5.0):
    """
    Projects all weights onto an L1-norm ball with radius <= max_norm
    Simple (non-optimal) approach: rescale if it exceeds the limit.
    """
    def proj(x):
        norm = x.abs().sum()
        return x if norm <= max_norm else x * (max_norm / norm)
    return proj

def make_binary_projection():
    """
    Forces the weights to be -1 or +1 (sign-based binary approximation)
    """
    return lambda x: x.sign

def make_unit_sphere_projection():
    """
    Normalizes the weight vector so that it has an L2 norm equal to 1
    """
    return lambda x: x / (x.norm() + 1e-8)


projection_factory = {
    "clip": make_clip_projection,
    "l2": make_l2_projection,
    "l1": make_l1_projection,
    "binary": make_binary_projection,
    "unit": make_unit_sphere_projection,
}

# Example : projection_factory["clip"](min_val=-0.5, max_val=0.5)

# === Optimizer Factory ===
def get_optimizer(name, model_params, optimizer_params):
    if name == "SGD":
        return optim.SGD(model_params, **optimizer_params)
    elif name == "Adam":
        return optim.Adam(model_params, **optimizer_params)
    elif name == "RMSprop":
        return optim.RMSprop(model_params, **optimizer_params)
    elif name == "Adagrad":
        return optim.Adagrad(model_params, **optimizer_params)
    elif name == "Adadelta":
        return optim.Adadelta(model_params, **optimizer_params)
    elif name == "MySGD":
        return MySGD(model_params, **optimizer_params)
    elif name == "GD":
        return GradientDescent(model_params, **optimizer_params)
    elif name == "PGD":
        return ProjectedGradientDescent(model_params, **optimizer_params)
    elif name == "PartialGD":
        return PartialGradientDescent(model_params, **optimizer_params)
    elif name == "Momentum":
        return Momentum(model_params, **optimizer_params)
    elif name == "Newton":
        return NewtonOptimizer(model_params, **optimizer_params)
    elif name == "Nesterov":
        return Nesterov(model_params, **optimizer_params)
    else:
        raise ValueError(f"Optimizer {name} not supported.")













# === Scheduler Factory ===
def get_scheduler(optimizer, scheduler_config):
    scheduler_type = scheduler_config.get("type")
    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_config["params"])
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config["params"])
    else:
        raise ValueError(f"Scheduler {scheduler_type} not supported.")
    return scheduler













# === Training & Evaluation with Scheduler Support ===

# === Training & Evaluation with Scheduler Support ===
def infer_model_params_from_dataset(model_name, dataset_name, model_params=None):
    if model_params is None:
        model_params = {}

    # === Nombre de classes ===
    if "num_classes" not in model_params:
        if dataset_name in ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]:
            model_params["num_classes"] = 10
        elif dataset_name == "CIFAR100":
            model_params["num_classes"] = 100
        elif dataset_name == "EMNIST_BALANCED":
            model_params["num_classes"] = 47
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # === Taille d'entrée pour MLP ===
    if model_name == "MLP" and "input_size" not in model_params:
        if dataset_name in ["MNIST", "FashionMNIST",  "EMNIST_BALANCED"]:
            model_params["input_size"] = 28 * 28
        elif dataset_name in ["CIFAR10", "CIFAR100", "SVHN"]:
            model_params["input_size"] = 3 * 32 * 32
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # === Nombre de canaux pour CNN ===
    if model_name == "CNN" and "in_channels" not in model_params:
        if dataset_name in ["MNIST", "FashionMNIST", "EMNIST_BALANCED"]:
            model_params["in_channels"] = 1
        elif dataset_name in ["CIFAR10", "CIFAR100", "SVHN"]:
            model_params["in_channels"] = 3
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    return model_params


def train_and_evaluate(dataset_name, optimizer_name, model_name, optimizer_params,
                       model_params=None, scheduler_config=None, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataset(dataset_name)

    if model_params is None:
        model_params = {}

    model_params = infer_model_params_from_dataset(model_name, dataset_name, model_params)

    # Création du modèle
    model = get_model(model_name, **model_params).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), optimizer_params)

    scheduler = None
    if scheduler_config:
        scheduler = get_scheduler(optimizer, scheduler_config)

    train_losses, test_losses, test_accuracies = [], [], []
    start_time = time.time() # Start time for training
    duration = [] # List to store time taken for each epoch

    for epoch in tqdm.tqdm(range(epochs), desc="Training", unit="epoch"):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if optimizer_name == "Newton":
                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    return loss
                loss = optimizer.step(closure)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        correct, total = 0, 0
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        avg_test_loss = running_test_loss / len(test_loader)
        accuracy = correct / total
        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(accuracy)
        elif scheduler:
            scheduler.step()

        # print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f} - Accuracy: {accuracy:.4f}")
        epoch_time = time.time() - epoch_start
        duration.append(epoch_time)
    
    
    return train_losses, test_losses, test_accuracies, duration


def run_multiple_seeds(dataset_name, optimizer_name, model_name, optimizer_params,
                       model_params=None, scheduler_config=None, epochs=5, seeds=[0, 1, 2]):
    all_train_losses = []
    all_test_losses = []
    all_test_accuracies = []

    for seed in seeds:
        set_seed(seed)
        train_losses, test_losses, test_accuracies, _ = train_and_evaluate(
            dataset_name,
            optimizer_name,
            model_name,
            optimizer_params,
            model_params=model_params,
            scheduler_config=scheduler_config,
            epochs=epochs
        )
        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)
        all_test_accuracies.append(test_accuracies)

    # Moyennes et écarts-types
    avg_train_loss = np.mean(all_train_losses, axis=0)
    std_train_loss = np.std(all_train_losses, axis=0)
    avg_test_loss = np.mean(all_test_losses, axis=0)
    std_test_loss = np.std(all_test_losses, axis=0)
    avg_test_acc = np.mean(all_test_accuracies, axis=0)
    std_test_acc = np.std(all_test_accuracies, axis=0)

    return {
        "avg_train_loss": avg_train_loss,
        "std_train_loss": std_train_loss,
        "avg_test_loss": avg_test_loss,
        "std_test_loss": std_test_loss,
        "avg_test_acc": avg_test_acc,
        "std_test_acc": std_test_acc,
        "all_test_accuracies": all_test_accuracies  # si tu veux tracer individuellement aussi
    }
def convert_np(obj):
    """Converts numpy objects to basic Python types (for JSON)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


# === Experiment Runner with Param Scan Support ===
import os
import json

def run_experiments(
    datasets,
    models,
    optimizers_with_params,
    scheduler_config=None,
    epochs=5,
    save_results=False,
    save_path="results.json",
    linear=False,
    avg_over_seeds = False,
    seeds = [0, 1, 2, 3, 4]
):
    results = []

    if linear:
        if not (len(datasets) == len(models) == len(optimizers_with_params)):
            raise ValueError("Si linear=True, datasets, models et optimizers_with_params doivent avoir la même taille.")
        configs = zip(datasets, models, optimizers_with_params)
    else:
        configs = (
            (d, m, o)
            for d in datasets
            for m in models
            for o in optimizers_with_params
        )

    def format_params(d):
        return ", ".join(f"{k}={v}" for k, v in d.items())

    for i, (dataset, model_config, opt_dict) in enumerate(configs, 1):
        model_name = model_config["name"]
        model_params = model_config.get("params", {}).copy()
        opt_name = opt_dict["name"]
        opt_params = opt_dict["params"]

        print(f"\n🔹 Run {i}")
        print(f"📊 Dataset     : {dataset}")
        print(f"🧠 Model       : {model_name}")
        if model_params:
            print(f"    ↳ Model Params : {format_params(model_params)}")
        print(f"🛠️ Optimizer   : {opt_name}")
        if opt_params:
            print(f"    ↳ Optim Params : {format_params(opt_params)}")
        if scheduler_config:
            print(f"📉 Scheduler   : {scheduler_config['type']} ({format_params(scheduler_config['params'])})")
        print(f"📈 Epochs      : {epochs}")

        if avg_over_seeds:
          res = run_multiple_seeds(
              dataset_name=dataset,
              optimizer_name=opt_name,
              model_name=model_name,
              optimizer_params=opt_params,
              model_params=model_params,
              scheduler_config=scheduler_config,
              epochs=epochs,
              seeds=seeds
          )
          results.append({
              "dataset": dataset,
              "model": model_name,
              "model_params": model_params,
              "optimizer": opt_name,
              "optimizer_params": opt_params,
              "scheduler": scheduler_config,
              "train_losses_avg": res["avg_train_loss"],
              "train_losses_std": res["std_train_loss"],
              "test_losses_avg": res["avg_test_loss"],
              "test_losses_std": res["std_test_loss"],
              "accuracies_avg": res["avg_test_acc"],
              "accuracies_std": res["std_test_acc"],
              "all_accuracies": res["all_test_accuracies"]
          })
        else:
          train_losses, test_losses, test_accuracies, duration = train_and_evaluate(
              dataset_name=dataset,
              optimizer_name=opt_name,
              model_name=model_name,
              optimizer_params=opt_params,
              model_params=model_params,
              scheduler_config=scheduler_config,
              epochs=epochs
          )
          results.append({
              "dataset": dataset,
              "model": model_name,
              "model_params": model_params,
              "optimizer": opt_name,
              "optimizer_params": opt_params,
              "scheduler": scheduler_config,
              "duration": duration,
              "train_losses": train_losses,
              "test_losses": test_losses,
              "accuracies": test_accuracies
          })

        if save_results:
            temp_path = save_path.replace(".json", "_temp.json")
            with open(temp_path, "w") as f:
                json.dump(results, f, indent=2, default=convert_np)
            print(f"💾 Temp results saved to {temp_path}")
    if save_results:
        os.makedirs("Data", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, default=convert_np)
        print(f"\n✅ Résultats sauvegardés dans '{save_path}'")

    return results


# function to treat results: takes results and returns the best experiment (best lr) for each optimizer
# so we can compare optimizer that are best_tuned
def select_best_lr_per_optimizer(results, metric="accuracies"):
    best_experiments = []
    results_df = pd.DataFrame(results)

    optimizers = results_df["optimizer"].unique()

    for optimizer in optimizers:
        subset = results_df[results_df["optimizer"] == optimizer].copy()
        
        # Prendre la moyenne de la dernière epoch d'accuracy
        subset["final_metric"] = subset[metric].apply(lambda x: x[-1])
        
        # Sélectionner la meilleure expérience
        if metric == "test_losses":
            best_row = subset.loc[subset["final_metric"].idxmin()]
        else:
            best_row = subset.loc[subset["final_metric"].idxmax()]
        best_experiments.append(best_row.to_dict())

    return best_experiments


# just to output a clear ranking of the optimizers
# (best to worst) based on the best results of each optimizer
def rank_optimizers(best_results, metric="accuracies"):
    """
    Ranks the optimizers based on their performance (on the given dataset and model).

    Args:
        best_results (list of dict): List of best experiments per optimizer.
        metric (str): 'accuracies' (default) or 'test_losses'.

    Returns:
        pandas.DataFrame: DataFrame sorted from best to worst optimizer.
    """
    
    df = pd.DataFrame(best_results).copy()

    # Ajouter une colonne pour la performance finale (si ce n'est pas déjà fait)
    if "final_metric" not in df.columns:
        df["final_metric"] = df[metric].apply(lambda x: x[-1])

    # Trier selon le metric
    if metric == "test_losses":
        df = df.sort_values("final_metric", ascending=True)
    else:
        df = df.sort_values("final_metric", ascending=False)
    
    df = df.reset_index(drop=True)
    return df[["optimizer", "lr", "final_metric"]]






###################################################
# === functions that helps to create the config ===
###################################################


# === Parameter Sweep ===
def build_optimizer_param_sweep(name, **param_lists):
    """
    Creates a list of optimizer configurations from parameter lists.
    The lists must be the same size and are combined using `zip`.

    Example:
    build_optimizer_param_sweep("SGD", lr=[0.01, 0.1], weight_decay=[0.0, 1e-4])
    -> [
        {"name": "SGD", "params": {"lr": 0.01, "weight_decay": 0.0}},
        {"name": "SGD", "params": {"lr": 0.1, "weight_decay": 1e-4}}
    ]
    """
    keys = list(param_lists.keys())
    values = zip(*[param_lists[k] for k in keys])
    return [
        {
            "name": name,
            "params": dict(zip(keys, vals))
        } for vals in values
    ]

# === Scheduler and Optimizer Configs ===
def make_scheduler_config(scheduler_type, **params):
    """
    Creates a configuration dictionary for a scheduler.
    Example: make_scheduler_config("StepLR", step_size=2, gamma=0.5)
    """
    return {"type": scheduler_type, "params": params}

def make_optimizer_config(name, **params):
    """
    Creates a configuration dictionary for an optimizer.
    Example: make_optimizer_config("Adam", lr=0.001, weight_decay=1e-4)
    """
    return {"name": name, "params": params}

def make_model_config(name, **params):
    """
    Creates a standardized model configuration.
    Example: make_model_config("MLP", input_size=784, hidden_sizes=[128,64], num_classes=10)
    """
    return {"name": name, "params": params}

