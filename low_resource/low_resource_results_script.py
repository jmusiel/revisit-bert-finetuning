import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from results_utils import get_test_accs, get_test_metrics, get_val_accs, debias

results = pd.read_csv("low_resource_results.csv")

plot_values = {}
filetype = "test_last_log"
datasets = [
    "RTE",
    "MRPC",
    "STS-B",
    "CoLA",
]

for dataset in datasets:
    plot_values[dataset] = {
        "split": [],
        "mean": [],
        "std": [],
    }
    for method in ["reinit_debiased"]:
        for split in [1, 5, 10, 100, 1000]:
            plot_values[dataset]["split"].append(split)
            simulations = []
            for replicate in range(5):
                test_accuracies = get_test_accs(results, dataset, method, filetype, split, replicate)
                val_accuracies = get_val_accs(results, dataset, method, filetype, split, replicate)
                if len(test_accuracies) < 5 or len(val_accuracies) < 5:
                    print("missing data for dataset:"+str(dataset) + " method:" + str(method) + " split:" + str(split) + " rep:" + str(replicate))
                else:
                    simulations.append(debias(val_accuracies, test_accuracies))
            plot_values[dataset]["mean"].append(np.mean(simulations))
            plot_values[dataset]["std"].append(np.std(simulations))

fig, axs = plt.subplots(1, 4, figsize=(20, 4))

test_metrics = get_test_metrics()

for dataset, ax in zip(datasets, axs):
    test_metric = test_metrics[dataset]
    test_metric = test_metric[0][test_metric[1]]
    ax.set_title(dataset, size=20)
    ax.set_ylabel(test_metric, size=20)
    ax.set_xlabel("Size of Split", size=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # if dataset == "RTE":
    #     ylimits = [0.5, 0.75]
    # elif dataset == "MRPC":
    #     ylimits = [0.8, 0.95]
    # elif dataset == "STS-B":
    #     ylimits = [0.85, 0.9]
    # elif dataset == "CoLA":
    #     ylimits = [0.4, 0.7]
    ylimits = [0, 1]
    ax.set_ylim(ylimits)

    for method in ["reinit_debiased"]:
        # if method == "not_debiased":
        #     label = "No Correction"
        #     color = "indianred"
        #     fillcolor = "lightcoral"
        if method == "reinit_debiased":
            label = "Correction"
            color = "cornflowerblue"
            fillcolor = "lightsteelblue"

        x = np.array(plot_values[dataset]["split"])
        y = np.array(plot_values[dataset]["mean"])
        std = np.array(plot_values[dataset]["std"])
        ax.semilogx(x, y, color=color)
        ax.fill_between(x, y - std, y + std, color=fillcolor, alpha=0.5)
    # ax.legend(fontsize=20)
fig.tight_layout()
plt.savefig("figure_1_lowresource.png")