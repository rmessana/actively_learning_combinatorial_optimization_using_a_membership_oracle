import re

from os import makedirs, path

import matplotlib.pyplot as plt
import numpy as np


max_n_queries = 2000

algorithms = [
    ("SEP", "CUT"),
    ("SVM", "CUT"),
    ("SVM", "SIM")
]

dimension = 100
instance_numbers = list(range(100, 1000, 30))
instance_types = ["uncorrelated", "weakly-correlated", "strongly-correlated"]

output_directory = "KNAP_output/graphs"
if not path.exists(output_directory):
    makedirs(output_directory)

for instance_type in instance_types:

    cumulative_errors = {algorithm: np.zeros(max_n_queries) for algorithm in algorithms}

    for instance_number in instance_numbers:

        for separator_name, strategy_name in algorithms:

            errors = []

            with open("KNAP_output/logs/{}+{}/{}/{}.log".format(separator_name, strategy_name, instance_type,
                                                                instance_number), "r") as f:

                lines = f.readlines()

                exact_best_objective = eval(lines[3][len("Exact best objective value: "):])

                for line in lines:
                    if "best objective: " in line:
                        match = re.search("best objective: \d*\.?\d*", line)
                        best_objective = eval(line[match.span()[0]:match.span()[1]][16:])
                        errors.append(100 * (exact_best_objective - best_objective) / exact_best_objective)

            errors = errors + [errors[-1]] * (max_n_queries - len(errors))
            cumulative_errors[(separator_name, strategy_name)] += errors

    fig, ax = plt.subplots(figsize=(16, 4))

    for separator_name, strategy_name in algorithms:
        ax.plot(range(max_n_queries), cumulative_errors[(separator_name, strategy_name)] / len(instance_numbers))
    plt.xlabel("Number of oracle calls")
    plt.ylabel("Average relative objective error (%)")

    ax.set_ylim({"uncorrelated": (0, 16), "weakly-correlated": (0, 16), "strongly-correlated": (0, 16)}[instance_type])

    if instance_type == "uncorrelated":
        zoom_ax = fig.add_axes([0.4425, 0.3, 0.3525, 0.5])
    elif instance_type == "weakly-correlated":
        zoom_ax = fig.add_axes([0.5925, 0.3, 0.2025, 0.5])
    else:
        zoom_ax = fig.add_axes([0.4425, 0.3, 0.3525, 0.5])

    for separator_name, strategy_name in algorithms:
        zoom_ax.plot(range(max_n_queries), cumulative_errors[(separator_name, strategy_name)] / len(instance_numbers))

    if instance_type == "uncorrelated":
        zoom_ax.set_xlim(600, 2000)
        zoom_ax.set_ylim(0, 0.15)
    elif instance_type == "weakly-correlated":
        zoom_ax.set_xlim(1200, 2000)
        zoom_ax.set_ylim(0, 0.15)
    else:  # instance_type == "strongly-correlated"
        zoom_ax.set_xlim(600, 2000)
        zoom_ax.set_ylim(0, 0.15)

    ax.legend(["{} + {}".format({"SEP": "SEP", "SVM": "SVM"}[separator_name], strategy_name)
               for separator_name, strategy_name in algorithms])

    plt.savefig("{}/average_{}.png".format(output_directory, instance_type))
