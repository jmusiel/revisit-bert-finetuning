import pandas as pd
import numpy as np
from replicate_results_utils import get_test_accs
from scipy.stats import ttest_ind_from_stats

results = pd.read_csv("replicate_results.csv")

# get results of table 1
plot_values = {}
filetype = "test_last_log"
datasets = [
    "RTE",
    "MRPC",
    "STS-B",
    "CoLA",
]

paper_results = {
    "RTE": {
        "Standard": {
            "mean": 69.5,
            "std": 2.5,
        },
        "Re-init": {
            "mean": 72.6,
            "std": 1.6,
        },
    },
    "MRPC": {
        "Standard": {
            "mean": 90.8,
            "std": 1.3,
        },
        "Re-init": {
            "mean": 91.4,
            "std": 0.8,
        },
    },
    "STS-B": {
        "Standard": {
            "mean": 89.0,
            "std": 0.6,
        },
        "Re-init": {
            "mean": 89.4,
            "std": 0.2,
        },
    },
    "CoLA": {
        "Standard": {
            "mean": 63.0,
            "std": 1.5,
        },
        "Re-init": {
            "mean": 63.9,
            "std": 1.9,
        },
    },
}

for dataset in datasets:
    for method in ["standard_debiased", "reinit_debiased"]:
        test_accuracies = get_test_accs(results, dataset, method, filetype)

        if dataset not in plot_values:
            plot_values[dataset] = {}
        plot_values[dataset][method] = {
            "x": [i for i in range(1, 51)],
            "y": [],
            "std": [],
        }

        if method == "reinit_debiased":
            method_str = "Re-init"
        if method == "standard_debiased":
            method_str = "Standard"
        acc_avg = np.mean(test_accuracies[-20:]) * 100
        acc_std = np.std(test_accuracies[-20:]) * 100

        ttest = ttest_ind_from_stats(
            mean1=paper_results[dataset][method_str]["mean"],
            std1=paper_results[dataset][method_str]["std"],
            nobs1=20,
            mean2=acc_avg,
            std2=acc_std,
            nobs2=20,
        )

        print_str = "Dataset: " + str(dataset) + " 3 Epochs, " + method_str + ", " + str(acc_avg) + " +- " + str(acc_std) + ", p-value: " + str(ttest.pvalue)
        if ttest.pvalue / 2 < 0.05:
            print_str += "*"
        print(print_str)
