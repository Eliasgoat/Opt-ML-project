import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.stats import sem
from collections import defaultdict


def plot_losses(results, param_index=None, selected_indices=None, save_path=None):
    """
    Affiche les courbes de pertes (train/test) pour les expériences données.
    
    - Si `param_index` est fourni, trace train+test pour une seule expérience.
    - Sinon, affiche toutes les pertes train puis test.
    - Affiche uniquement les paramètres qui varient.
    - `save_path` (sans extension) permet de sauvegarder les figures dans "Results/"
    """

    def format_params(params):
        return ", ".join(f"{k}={v}" for k, v in params.items())

    def find_varying_params(results):
        all_keys = results[0].keys()
        return [k for k in all_keys if k not in ['train_losses', 'test_losses', 'accuracies', 'duration']
                and len(set(str(exp.get(k)) for exp in results)) > 1]

    if param_index is not None:
        # === Cas 1 : un seul run ===
        exp = results[param_index]
        plt.figure(figsize=(8, 4))
        plt.plot(exp['train_losses'], label="Train Loss", linestyle='--')
        plt.plot(exp['test_losses'], label="Test Loss")
        plt.title("Train/Test Losses\n" + format_params({k: exp[k] for k in find_varying_params(results)}))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        if save_path:
            plt.savefig(os.path.join("Results", f"{save_path}_losses_exp{param_index}.png"), bbox_inches="tight", dpi=300)
        plt.show()

    else:
        # === Cas 2 : plusieurs runs ===
        if selected_indices is None:
            selected_indices = list(range(len(results)))
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))
        varying_params = find_varying_params(results)

        def plot_curve(metric, metric_name):
            plt.figure(figsize=(10, 5))
            for i, idx in enumerate(selected_indices):
                exp = results[idx]
                label = f"Exp {idx+1}: " + ", ".join(f"{k}={exp[k]}" for k in varying_params)
                plt.plot(exp[metric], label=label, color=colors[i])
            plt.title(f"{metric_name} for varying parameters: {', '.join(varying_params)}")
            plt.xlabel("Epochs")
            plt.ylabel(metric_name)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(fontsize=8, loc='best', ncol=2)
            if save_path:
                plt.savefig(os.path.join("Results", f"{save_path}_{metric}.png"), bbox_inches="tight", dpi=300)
            plt.show()

        plot_curve("train_losses", "Train Loss")
        plot_curve("test_losses", "Test Loss")

import matplotlib.pyplot as plt
import numpy as np
import os
import math
from collections import defaultdict
from scipy.stats import sem
from itertools import product

def extract_nested(d, key_path):
    """Extrait une valeur depuis un dictionnaire imbriqué selon un chemin en string 'a.b.c'."""
    for key in key_path.split('.'):
        d = d.get(key, {})
    return d if d != {} else None

def plot_metrics_vs_param_grouped(results, x_param, metrics, group_by=None, split_by=None,
                                   title="", save_path=None, logx=False, logy=False,
                                   grid=True, max_overall=False, subplots=True):
    """
    Générique : supporte group_by, split_by, clés imbriquées.
    """
    if isinstance(group_by, str):
        group_by = [group_by]
    if isinstance(split_by, str):
        split_by = [split_by]

    # Récupérer toutes les combinaisons de split_by
    unique_split_vals = {
        key: sorted(set(extract_nested(r, key) for r in results if extract_nested(r, key) is not None))
        for key in split_by
    }
    all_combos = list(product(*(unique_split_vals[k] for k in split_by))) if split_by else [()]

    split_groups = {}
    for combo in all_combos:
        combo_dict = dict(zip(split_by, combo))
        filtered = [
            r for r in results if all(extract_nested(r, k) == v for k, v in combo_dict.items())
        ]
        split_key = tuple((k, combo_dict[k]) for k in split_by)
        if filtered:
            split_groups[split_key] = filtered

    def plot_one_panel(ax, group_results, color_cycle):
        # Grouping
        grouped = defaultdict(list)
        for exp in group_results:
            key = tuple((k, extract_nested(exp, k)) for k in group_by)
            grouped[key].append(exp)

        for i, (gkey, exps) in enumerate(grouped.items()):
            x_vals = defaultdict(list)
            for exp in exps:
                x = extract_nested(exp, x_param)
                if x is not None:
                    x_vals[x].append(exp)

            sorted_x = sorted(x_vals)
            for metric in metrics:
                y_means, y_errs = [], []
                for x in sorted_x:
                    vals = []
                    for exp in x_vals[x]:
                        values = exp.get(metric)
                        if isinstance(values[0], list):
                            values = [v[-1] for v in values]
                        vals.append(np.mean(values))
                    if max_overall:
                        best = max(vals)
                        y_means.append(best)
                        y_errs.append(0)
                    else:
                        y_means.append(np.mean(vals))
                        y_errs.append(sem(vals) if len(vals) > 1 else 0)

                label = ", ".join(f"{k}={v}" for k, v in gkey) if gkey else metric
                ax.errorbar(sorted_x, y_means, yerr=y_errs, fmt='-o', label=label,
                            color=color_cycle[i % len(color_cycle)], capsize=4)

        ax.set_xlabel(x_param)
        ax.set_ylabel(", ".join(metrics))
        if logx: ax.set_xscale("log")
        if logy: ax.set_yscale("log")
        if grid: ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8)

    # Subplots
    num = len(split_groups)
    ncols = min(2, num)
    nrows = math.ceil(num / ncols)
    color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))

    if subplots:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 5*nrows), squeeze=False)
        for i, (split_key, group) in enumerate(split_groups.items()):
            ax = axes[i//ncols][i%ncols]
            split_label = ", ".join(f"{k}={v}" for k, v in split_key) if split_by else ""
            ax.set_title(f"{title}\n{split_label}" if title else split_label)
            plot_one_panel(ax, group, color_cycle)
        plt.tight_layout()
        if save_path:
            os.makedirs("Results", exist_ok=True)
            plt.savefig(os.path.join("Results", f"{save_path}.png"), bbox_inches="tight", dpi=300)
        plt.show()
    else:
        for split_key, group in split_groups.items():
            plt.figure(figsize=(8, 5))
            split_label = ", ".join(f"{k}={v}" for k, v in split_key) if split_by else ""
            plt.title(f"{title}\n{split_label}" if title else split_label)
            ax = plt.gca()
            plot_one_panel(ax, group, color_cycle)
            plt.tight_layout()
            if save_path:
                suffix = "_".join(f"{k}_{v}" for k, v in split_key)
                os.makedirs("Results", exist_ok=True)
                plt.savefig(os.path.join("Results", f"{save_path}_{suffix}.png"), bbox_inches="tight", dpi=300)
            plt.show()
import json

def load_results(filename):
    """
    Charge un fichier de résultats JSON depuis le dossier 'Data/'.
    
    Paramètres :
    - filename (str) : nom du fichier (avec ou sans extension)

    Retourne :
    - results (list[dict]) : liste des expériences
    """
    if not filename.endswith(".json"):
        filename += ".json"

    full_path = os.path.join("Data", filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Le fichier '{full_path}' n'existe pas.")
    
    with open(full_path, "r") as f:
        results = json.load(f)

    print(f"✅ {len(results)} expériences chargées depuis '{full_path}'")
    return results
