import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import json
import time
import tqdm
import pandas as pd


# === Seed setting ===
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

def get_model(name, **model_params):
    """
    Retourne un modèle en fonction du nom et des paramètres.
    
    model_params : dictionnaire contenant les bons arguments pour le modèle choisi.
    """
    if name == "MLP":
        return MLP(**model_params)
    elif name == "CNN":
        return SimpleCNN(**model_params)
    else:
        raise ValueError(f"Model {name} not supported yet.")
    









# === Dataset ===
def get_dataset(name, batch_size=64):
    """
    Charge un dataset depuis torchvision.
    
    Arguments :
    - name : nom du dataset ("MNIST", "FashionMNIST", "CIFAR10", "CIFAR100")
    - batch_size : taille de batch pour les dataloaders
    
    Retourne : train_loader, test_loader
    """
    if name == "MNIST":
        transform = transforms.ToTensor()
        train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif name == "FashionMNIST":
        transform = transforms.ToTensor()
        train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
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
        raise ValueError(f"Dataset {name} not supported yet.")

    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size, shuffle=False)















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
    Implémentation naïve d'une méthode de Newton.
    Utilise l’inverse de la Hessienne pour faire : x ← x - H⁻¹g

    ⚠️ : très coûteux, ne pas utiliser avec gros modèles.
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
    Optimiseur de type PGD : descend le long du gradient, puis projette les poids dans un espace contraint.
    
    Arguments :
    - lr : taux d'apprentissage.
    - projection : fonction à appliquer sur les poids après mise à jour. Par défaut : clamp entre -1 et 1.
      Par exemple, pour projeter tous les poids à [0, 1], utiliser : projection=lambda x: x.clamp(0, 1)
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
                    # Appliquer la contrainte après la mise à jour
                    p.copy_(proj(p))

# === Projection Utils ===


def make_clip_projection(min_val=-1.0, max_val=1.0):
    """
    Projette chaque poids individuellement dans [min_val, max_val]
    """
    return lambda x: x.clamp(min_val, max_val)

def make_l2_projection(max_norm=5.0):
    """
    Projette l'ensemble des poids sur une boule de norme L2 <= max_norm
    """
    def proj(x):
        norm = x.norm()
        return x if norm <= max_norm else x * (max_norm / norm)
    return proj

def make_l1_projection(max_norm=5.0):
    """
    Projette l'ensemble des poids sur une boule de norme L1 <= max_norm
    Approche simple (non-optimale) : rescale si dépasse.
    """
    def proj(x):
        norm = x.abs().sum()
        return x if norm <= max_norm else x * (max_norm / norm)
    return proj

def make_binary_projection():
    """
    Force les poids à être -1 ou +1 (approximation binaire sign-based)
    """
    return lambda x: x.sign

def make_unit_sphere_projection():
    """
    Normalise le vecteur de poids pour qu’il ait une norme L2 = 1
    """
    return lambda x: x / (x.norm() + 1e-8)


projection_factory = {
    "clip": make_clip_projection,
    "l2": make_l2_projection,
    "l1": make_l1_projection,
    "binary": make_binary_projection,
    "unit": make_unit_sphere_projection,
}

# Exemple : projection_factory["clip"](min_val=-0.5, max_val=0.5)

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

def infer_model_params_from_dataset(model_name, dataset_name, model_params=None):
    """
    Déduit certains paramètres du modèle à partir du dataset, si non spécifiés.

    - num_classes : 10 ou 100 selon le dataset
    - input_size : pour MLP uniquement
    - in_channels : pour CNN uniquement
    """
    if model_params is None:
        model_params = {}

    # Nombre de classes
    if "num_classes" not in model_params:
        if dataset_name in ["MNIST", "FashionMNIST", "CIFAR10"]:
            model_params["num_classes"] = 10
        elif dataset_name == "CIFAR100":
            model_params["num_classes"] = 100
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # Dimensions d'entrée pour MLP
    if model_name == "MLP" and "input_size" not in model_params:
        if dataset_name in ["MNIST", "FashionMNIST"]:
            model_params["input_size"] = 28 * 28
        elif dataset_name in ["CIFAR10", "CIFAR100"]:
            model_params["input_size"] = 3 * 32 * 32

    # Canal d'entrée pour CNN
    if model_name == "CNN" and "in_channels" not in model_params:
        model_params["in_channels"] = 1 if dataset_name in ["MNIST", "FashionMNIST"] else 3

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
    linear=False
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
        model_params = model_config.get("params", {})
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
                json.dump(results, f, indent=2)
            print(f"💾 Temp results saved to {temp_path}")
    if save_results:
        os.makedirs("Data", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
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
    Classe les optimizers en fonction de leur performance (sur le dataset et modèle donnés).
    
    Args:
        best_results (list of dict): Liste des meilleures expériences par optimizer.
        metric (str): 'accuracies' (par défaut) ou 'test_losses'.
        
    Returns:
        pandas.DataFrame: DataFrame trié du meilleur au pire optimizer.
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
    Crée une liste de configurations d'optimiseurs à partir de listes de paramètres.
    Les listes doivent être de même taille et sont combinées par `zip`.
    
    Ex:
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
    Crée un dictionnaire de configuration pour un scheduler
    Ex: make_scheduler_config("StepLR", step_size=2, gamma=0.5)
    """
    return {"type": scheduler_type, "params": params}

def make_optimizer_config(name, **params):
    """
    Crée un dictionnaire de configuration pour un optimiseur
    Ex: make_optimizer_config("Adam", lr=0.001, weight_decay=1e-4)
    """
    return {"name": name, "params": params}

def make_model_config(name, **params):
    """
    Crée une configuration de modèle standardisée.
    Ex: make_model_config("MLP", input_size=784, hidden_sizes=[128,64], num_classes=10)
    """
    return {"name": name, "params": params}

