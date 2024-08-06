from os import makedirs, path

import gurobipy as gp
import numpy as np

from KNAP_generator import BinaryKnapsackGenerator
from KNAP_optimizer import Optimizer
from oracle import Oracle
from separator import SVMLinearSeparator

dimension = 100
instance_types = ["uncorrelated", "weakly-correlated", "strongly-correlated"]
instance_numbers = range(100, 1000, 30)

n_random_samples = 2000

cumulative_error_value = 0

for instance_type in instance_types:

    print("Instance type: {}".format(instance_type))

    output_directory = "KNAP_output/logs/baseline/{}".format(instance_type)
    if not path.exists(output_directory):
        makedirs(output_directory)

    cumulative_error_value = 0

    for instance_number in instance_numbers:

        # knapsack instance --------------------------------------------

        knapsack_instance = BinaryKnapsackGenerator(n_items=dimension, coefficient_range=10000).generate(
            instance_type=instance_type, instance_number=instance_number)
        values = knapsack_instance["values"]
        weights = knapsack_instance["weights"]
        capacity = knapsack_instance["capacity"]

        model = gp.Model("model")
        variables = model.addVars(range(dimension), vtype=gp.GRB.BINARY)
        constraint = model.addConstr(gp.quicksum(weights[j] * variables[j]
                                                 for j in range(dimension)) <= capacity)
        model.setObjective(gp.quicksum(values[j] * variables[j] for j in range(dimension)))
        model.setParam("OutputFlag", 0)
        model.setParam("Threads", 4)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.optimize()
        exact_optimal_solution = np.array([variables[j].X for j in range(dimension)])
        exact_objective_value = np.dot(values, exact_optimal_solution)

        # oracle -------------------------------------------------------

        oracle = Oracle(weights, capacity)

        # data ---------------------------------------------------------

        samples = np.empty((0, dimension))

        while samples.shape[0] < 2000:
            sample = np.random.randint(2, size=dimension)
            if not np.any(np.all(samples == sample, axis=1)):
                samples = np.concatenate([samples, [sample]], axis=0)

        points = {"feasible": np.empty((0, dimension)), "infeasible": np.empty((0, dimension))}
        for sample in samples:
            if oracle.query(sample) > 0.5:
                points["feasible"] = np.concatenate([points["feasible"], [sample]])
            else:
                points["infeasible"] = np.concatenate([points["infeasible"], [sample]])

        if points["feasible"].shape[0] == 0:
            best_point = None
            best_objective_value = 0.0
        else:
            best_point = points["feasible"][np.argmax([np.dot(values, point) for point in points["feasible"]])]
            best_objective_value = np.dot(values, best_point)

        # separator ----------------------------------------------------

        if points["feasible"].shape[0] > 0 and points["infeasible"].shape[0] > 0:

            separator = SVMLinearSeparator(data=points)
            separation_solution = separator.solve()
            b = separation_solution["b"]
            w = separation_solution["w"]

            # optimizer ----------------------------------------------------

            optimizer = Optimizer()
            status = {"values": values, "b": b, "w": w, "points": points}
            point, message = optimizer.step(status)
            if oracle.query(point) > 0.5:
                objective_value = np.dot(values, point)
                if objective_value > best_objective_value:
                    best_point = point.copy()
                    best_objective_value = objective_value

        # print("{}. E: {}, B: {}, err: {}".format(
        #     instance_number, exact_objective_value, best_objective_value,
        #     100 * (exact_objective_value - best_objective_value) / exact_objective_value))

        cumulative_error_value += 100 * (exact_objective_value - best_objective_value) / exact_objective_value

    print("Average relative error value (%): {}".format(cumulative_error_value / len(instance_numbers)))
