import json
import time

from os import makedirs, path

import gurobipy as gp
import numpy as np

from KNAP_generator import BinaryKnapsackGenerator
from functions import log
from KNAP_optimizer import Optimizer
from oracle import Oracle
from separator import SEPLinearSeparator, SVMLinearSeparator
from KNAP_strategy import KNAPCUTStrategy, KNAPSIMStrategy


algorithms = [
    ("SEP", "CUT"),
    ("SVM", "CUT"),
    ("SVM", "SIM")
]

dimension = 100
instance_types = ["uncorrelated", "weakly-correlated", "strongly-correlated"]
instance_numbers = range(100, 1000, 30)

max_n_queries = 2000
n_query_checkpoints = [8, 40, 200, 1000]

for separator_name, strategy_name in algorithms:

    for instance_type in instance_types:

        output_directory = "KNAP_output/logs/{}+{}/{}".format(separator_name, strategy_name, instance_type)
        if not path.exists(output_directory):
            makedirs(output_directory)

        # --------------------------------------------------------------

        results = []
        checkpoint_results = {n: [] for n in n_query_checkpoints}

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

            # initial data -------------------------------------------------

            points = {"feasible": np.array([np.zeros(dimension)]), "infeasible": np.array([np.ones(dimension)])}
            new_points = {"feasible": np.empty((0, dimension)), "infeasible": np.empty((0, dimension))}

            best_point = points["feasible"][np.argmax([np.dot(values, point) for point in points["feasible"]])]
            best_objective_value = np.dot(values, best_point)

            # separator ----------------------------------------------------

            if separator_name == "SEP":
                separator = SEPLinearSeparator(data=points)
            elif separator_name == "SVM":
                separator = SVMLinearSeparator(data=points)
            else:
                separator = None

            b = None
            w = None

            # optimizer ----------------------------------------------------

            optimizer = Optimizer()

            # strategy -----------------------------------------------------

            if strategy_name == "CUT":
                strategy = KNAPCUTStrategy(dimension=dimension)
            elif strategy_name == "SIM":
                strategy = KNAPSIMStrategy(dimension=dimension)
            else:
                strategy = None

            # metrics ------------------------------------------------------

            separation_cumulative_elapsed_time = 0.0
            strategy_cumulative_elapsed_time = 0.0
            optimization_cumulative_elapsed_time = 0.0
            cumulative_relative_error = 0.0
            cumulative_error = 0.0
            relative_error = 1.0
            error = exact_objective_value

            # --------------------------------------------------------------

            status = {"values": values, "b": b, "w": w, "points": points, "best_point": best_point}

            with (open("{}/{}.log".format(output_directory, instance_number), 'w') as f):

                log(f, "Separator: {}, strategy: {}, dimension: {}, instance type: {}, "
                       "instance number: {}".format(separator_name, strategy_name, dimension, instance_type,
                                                    instance_number))

                log(f, "\nExact optimal solution: {}".format(''.join([str(round(n))
                                                                      for n in exact_optimal_solution])))
                log(f, "Exact best objective value: {}\n".format(exact_objective_value))
                log(f, "iteration\tpoint\tobjective\tlabel\tinfo")

                iteration = 0
                start_time = time.time()

                while True:

                    iteration += 1
                    iteration_start_time = time.time()
                    new_points = {"feasible": np.empty((0, dimension)), "infeasible": np.empty((0, dimension))}

                    # separation ---------------------------------------------------

                    separation_start_time = time.time()
                    separation_solution = separator.solve()
                    b = separation_solution["b"]
                    w = separation_solution["w"]
                    separation_elapsed_time = time.time() - separation_start_time
                    separation_cumulative_elapsed_time += separation_elapsed_time

                    log(f, "{}\t\t\tseparation\t({} feasible, {} infeasible) "
                           "(weight distance: {}) (elapsed time: {})".format(
                            iteration, points["feasible"].shape[0], points["infeasible"].shape[0],
                            np.linalg.norm(weights / capacity + w / b), separation_elapsed_time))

                    log(f, "{}\t\t\tseparation\t(weights: {})".format(iteration, list(-w / b)))

                    status = {"values": values, "b": b, "w": w, "points": points, "best_point": best_point}

                    # strategy -----------------------------------------------------

                    strategy_start_time = time.time()
                    point, message = strategy.step(status)
                    strategy_elapsed_time = time.time() - strategy_start_time
                    strategy_cumulative_elapsed_time += strategy_elapsed_time

                    label = None

                    if point is None:

                        log(f, "{}\t\t\t".format(iteration) + message)

                    elif list(point) in points["feasible"].tolist() or \
                            list(point) in points["infeasible"].tolist():

                        log(f, "{}\t\t\t{} already in dataset (strategy: {})".format(
                            iteration, ''.join([str(round(n)) for n in point]), strategy.name))
                        point = None

                    else:

                        label = oracle.query(point)
                        objective_value = np.dot(values, point)

                        if label == 1:
                            new_points["feasible"] = np.concatenate([new_points["feasible"], [point]],
                                                                    axis=0)
                            if objective_value >= best_objective_value:
                                best_objective_value = objective_value
                                best_point = point
                                error = exact_objective_value - best_objective_value
                                relative_error = error / exact_objective_value
                        else:
                            new_points["infeasible"] = np.concatenate([new_points["infeasible"],
                                                                       [point]], axis=0)
                        cumulative_relative_error += relative_error
                        cumulative_error += error

                        status = {"values": values, "b": b, "w": w, "points": points, "best_point": best_point}

                        log(f, "{}\t\t\t{}\t{}\t{}\t(best objective: {}) {} (elapsed time: {})".format(
                            iteration, ''.join([str(round(n)) for n in point]), round(objective_value, 4),
                            {1: "feas", 0: "infeas"}[label], best_objective_value, message,
                            strategy_elapsed_time))

                        if exact_objective_value == best_objective_value or oracle.n_queries >= max_n_queries:
                            break

                        if oracle.n_queries in n_query_checkpoints:
                            last_error = exact_objective_value - best_objective_value
                            last_relative_error = last_error / exact_objective_value
                            checkpoint_separation_solution = separator.solve_nomod(new_points)
                            checkpoint_b = checkpoint_separation_solution["b"]
                            checkpoint_w = checkpoint_separation_solution["w"]
                            checkpoint_results[oracle.n_queries].append(
                                {"instance_number": instance_number,
                                 "solved": False,
                                 "elapsed_time": time.time() - start_time,
                                 "n_queries": oracle.n_queries,
                                 "n_iterations": iteration,
                                 "last_error": last_error,
                                 "last_relative_error": last_relative_error,
                                 "cumulative_error": cumulative_error,
                                 "cumulative_relative_error": cumulative_relative_error,
                                 "n_feasible": points["feasible"].shape[0] + new_points["feasible"].shape[0],
                                 "n_infeasible": points["infeasible"].shape[0] +
                                    new_points["infeasible"].shape[0],
                                 "separation_cumulative_elapsed_time": separation_cumulative_elapsed_time,
                                 "strategy_cumulative_elapsed_time": strategy_cumulative_elapsed_time,
                                 "optimization_cumulative_elapsed_time": optimization_cumulative_elapsed_time,
                                 "approximate_weights": list(- checkpoint_w / checkpoint_b)})

                    # optimization -------------------------------------------------

                    optimization_start_time = time.time()
                    point, message = optimizer.step(status)
                    optimization_elapsed_time = time.time() - optimization_start_time
                    optimization_cumulative_elapsed_time += optimization_elapsed_time

                    label = None

                    if point is None:

                        log(f, "{}\t\t\t".format(iteration) + message)

                    elif list(point) in points["feasible"].tolist() or \
                            list(point) in points["infeasible"].tolist():

                        log(f, "{}\t\t\t{} already in dataset (strategy: optimization)".format(
                            iteration, ''.join([str(round(n)) for n in point]), strategy.name))
                        point = None

                    else:

                        label = oracle.query(point)
                        objective_value = np.dot(values, point)

                        if label == 1:
                            new_points["feasible"] = np.concatenate([new_points["feasible"], [point]],
                                                                    axis=0)
                            if objective_value >= best_objective_value:
                                best_objective_value = objective_value
                                best_point = point
                                error = exact_objective_value - best_objective_value
                                relative_error = error / exact_objective_value
                        else:
                            new_points["infeasible"] = np.concatenate([new_points["infeasible"],
                                                                       [point]], axis=0)
                        cumulative_relative_error += relative_error
                        cumulative_error += error

                        status = {"values": values, "b": b, "w": w, "points": points, "best_point": best_point}

                        log(f, "{}\t\t\t{}\t{}\t{}\t(best objective: {}) {} (elapsed time: {})".format(
                            iteration, ''.join([str(round(n)) for n in point]), round(objective_value, 4),
                            {1: "feas", 0: "infeas"}[label], best_objective_value, message,
                            optimization_elapsed_time))

                        if exact_objective_value == best_objective_value or oracle.n_queries >= max_n_queries:
                            break

                        if oracle.n_queries in n_query_checkpoints:
                            last_error = exact_objective_value - best_objective_value
                            last_relative_error = last_error / exact_objective_value
                            checkpoint_separation_solution = separator.solve_nomod(new_points)
                            checkpoint_b = checkpoint_separation_solution["b"]
                            checkpoint_w = checkpoint_separation_solution["w"]
                            checkpoint_results[oracle.n_queries].append(
                                {"instance_number": instance_number,
                                 "solved": False,
                                 "elapsed_time": time.time() - start_time,
                                 "n_queries": oracle.n_queries,
                                 "n_iterations": iteration,
                                 "last_error": last_error,
                                 "last_relative_error": last_relative_error,
                                 "cumulative_error": cumulative_error,
                                 "cumulative_relative_error": cumulative_relative_error,
                                 "n_feasible": points["feasible"].shape[0] + new_points["feasible"].shape[0],
                                 "n_infeasible": points["infeasible"].shape[0] +
                                    new_points["infeasible"].shape[0],
                                 "separation_cumulative_elapsed_time": separation_cumulative_elapsed_time,
                                 "strategy_cumulative_elapsed_time": strategy_cumulative_elapsed_time,
                                 "optimization_cumulative_elapsed_time": optimization_cumulative_elapsed_time,
                                 "approximate_weights": list(- checkpoint_w / checkpoint_b)})

                    # dataset update -----------------------------------------------

                    if new_points["feasible"].shape[0] == 0 and new_points["infeasible"].shape[0] == 0:
                        break
                    new_points["feasible"] = np.unique(new_points["feasible"], axis=0)
                    points["feasible"] = np.concatenate([points["feasible"], new_points["feasible"]], axis=0)
                    new_points["infeasible"] = np.unique(new_points["infeasible"], axis=0)
                    points["infeasible"] = np.concatenate([points["infeasible"], new_points["infeasible"]],
                                                          axis=0)

                    # separator update ---------------------------------------------

                    separator.update(new_points)

                if new_points["feasible"].shape[0] > 0 or new_points["infeasible"].shape[0] > 0:

                    # dataset update -----------------------------------------------

                    new_points["feasible"] = np.unique(new_points["feasible"], axis=0)
                    points["feasible"] = np.concatenate([points["feasible"], new_points["feasible"]], axis=0)
                    new_points["infeasible"] = np.unique(new_points["infeasible"], axis=0)
                    points["infeasible"] = np.concatenate([points["infeasible"], new_points["infeasible"]],
                                                          axis=0)

                    # separator update ---------------------------------------------
                    separator.update(new_points)

                    # separation ---------------------------------------------------

                    separation_start_time = time.time()
                    separation_solution = separator.solve()
                    b = separation_solution["b"]
                    w = separation_solution["w"]
                    separation_elapsed_time = time.time() - separation_start_time
                    separation_cumulative_elapsed_time += separation_elapsed_time

                    log(f, "{}\t\t\tseparation\t({} feasible, {} infeasible) "
                           "(weight distance: {}) (elapsed time: {})".format(
                            iteration, points["feasible"].shape[0], points["infeasible"].shape[0],
                            np.linalg.norm(weights / capacity + w / b), separation_elapsed_time))

                    log(f, "{}\t\t\tseparation\t(weights: {})".format(iteration, list(-w / b)))

                # end ----------------------------------------------------------

                elapsed_time = time.time() - start_time

                # output -------------------------------------------------------

                if exact_objective_value == best_objective_value:

                    log(f, "\nSolved in {} queries and {} iterations".format(oracle.n_queries, iteration))
                    instance_solved_flag = True

                else:

                    log(f, "\nUnsolved after {} queries and {} iterations".format(oracle.n_queries, iteration))
                    instance_solved_flag = False

                last_error = exact_objective_value - best_objective_value
                last_relative_error = last_error / exact_objective_value

                log(f, "Elapsed time: {}".format(elapsed_time))
                log(f, "Exact best objective value: {}".format(exact_objective_value))
                log(f, "Last error: {}".format(last_error))
                log(f, "Last relative error: {}".format(last_relative_error))
                log(f, "Cumulative relative error: {}".format(cumulative_relative_error))
                log(f, "Feasible points: {}".format(points["feasible"].shape[0]))
                log(f, "Infeasible points: {}".format(points["infeasible"].shape[0]))
                log(f, "Separation cumulative elapsed time: {}".format(separation_cumulative_elapsed_time))
                log(f, "Strategy cumulative elapsed time: {}".format(strategy_cumulative_elapsed_time))
                log(f, "Optimization cumulative elapsed time: {}".format(optimization_cumulative_elapsed_time))

                new_results = {"instance_number": instance_number,
                               "solved": instance_solved_flag,
                               "elapsed_time": elapsed_time,
                               "n_queries": oracle.n_queries,
                               "n_iterations": iteration,
                               "last_error": last_error,
                               "last_relative_error": last_relative_error,
                               "cumulative_error": cumulative_error,
                               "cumulative_relative_error": cumulative_relative_error,
                               "n_feasible": points["feasible"].shape[0],
                               "n_infeasible": points["infeasible"].shape[0],
                               "separation_cumulative_elapsed_time": separation_cumulative_elapsed_time,
                               "strategy_cumulative_elapsed_time": strategy_cumulative_elapsed_time,
                               "optimization_cumulative_elapsed_time": optimization_cumulative_elapsed_time,
                               "approximate_weights": list(- w / b)}

                results.append(new_results)
                for n_queries in n_query_checkpoints:
                    if oracle.n_queries <= n_queries:
                        checkpoint_results[n_queries].append(new_results)

        for n_queries in n_query_checkpoints:
            with open("{}/{}_results.json".format(output_directory, n_queries), "w") as f:
                json.dump(checkpoint_results[n_queries], f, indent=4)

        with open("{}/{}_results.json".format(output_directory, max_n_queries), "w") as f:
            json.dump(results, f, indent=4)
